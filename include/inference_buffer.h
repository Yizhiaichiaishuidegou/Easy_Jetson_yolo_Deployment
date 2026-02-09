/**
 * @file inference_buffer.h
 * @brief 推理缓冲区组 - 支持多缓冲流水线
 * @description 每个缓冲区组包含完整的输入/输出缓冲区，支持独立的CUDA流处理
 */

#ifndef INFERENCE_BUFFER_H
#define INFERENCE_BUFFER_H

#include "cuda_utils.h"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <vector>
#include <memory>

namespace jetson {

// 缓冲区状态
enum class BufferState {
    EMPTY,          // 空闲，可接收新帧
    PREPROCESSING,  // 正在预处理
    READY_INFER,    // 预处理完成，等待推理
    INFERRING,      // 正在推理
    READY_POST,     // 推理完成，等待后处理
    POSTPROCESSING, // 正在后处理
    READY_OUTPUT,   // 后处理完成，等待输出
    OUTPUT          // 正在输出
};

/**
 * @struct InferenceBuffer
 * @brief 单个推理缓冲区组 - 包含完整的输入/输出数据
 */
struct InferenceBuffer {
    // 缓冲区标识
    int buffer_id = -1;
    
    // 帧信息
    uint64_t frame_id = 0;
    int src_width = 0;
    int src_height = 0;
    
    // 仿射变换矩阵 (用于坐标映射)
    float affine_i2d[6];  // image to detection
    float affine_d2i[6];  // detection to image
    
    // 设备端缓冲区 (使用指针以支持移动)
    std::unique_ptr<DeviceBuffer<uint8_t>> src_device;     // 原始图像 (GPU)
    std::unique_ptr<DeviceBuffer<float>> input_tensor;      // 预处理后的输入张量
    std::unique_ptr<DeviceBuffer<float>> output_tensor;     // 推理输出张量
    std::unique_ptr<DeviceBuffer<float>> decode_output;     // 解码后的检测结果
    
    std::unique_ptr<ZeroCopyBuffer<float>> detection_results;  // 最终检测结果
    
    // CUDA 事件 (用于流同步)
    std::unique_ptr<CudaEvent> preprocess_done;
    std::unique_ptr<CudaEvent> inference_done;
    std::unique_ptr<CudaEvent> postprocess_done;
    
    // 状态
    std::atomic<BufferState> state{BufferState::EMPTY};
    
    // 原始帧引用 (用于绘制结果)
    cv::Mat src_frame;
    cv::Mat result_frame;
    
    // 检测结果
    std::vector<float> detections;  // [x1,y1,x2,y2,conf,class] * N
    int detection_count = 0;
    
    // 时间戳 (用于性能分析)
    double timestamp_push = 0;
    double timestamp_preprocess_start = 0;
    double timestamp_preprocess_end = 0;
    double timestamp_inference_start = 0;
    double timestamp_inference_end = 0;
    double timestamp_postprocess_start = 0;
    double timestamp_postprocess_end = 0;
    
    InferenceBuffer() {
        for (int i = 0; i < 6; ++i) {
            affine_i2d[i] = 0;
            affine_d2i[i] = 0;
        }
    }
    
    void initialize(int id, int input_h, int input_w, int output_size, int max_detections) {
        buffer_id = id;
        
        // 分配设备内存
        src_device = std::unique_ptr<DeviceBuffer<uint8_t>>(new DeviceBuffer<uint8_t>());
        input_tensor = std::unique_ptr<DeviceBuffer<float>>(new DeviceBuffer<float>(3 * input_h * input_w));
        output_tensor = std::unique_ptr<DeviceBuffer<float>>(new DeviceBuffer<float>(output_size));
        decode_output = std::unique_ptr<DeviceBuffer<float>>(new DeviceBuffer<float>(32 + max_detections * 8));
        
        // 分配零拷贝内存用于结果
        detection_results = std::unique_ptr<ZeroCopyBuffer<float>>(new ZeroCopyBuffer<float>(1 + max_detections * 8));
        
        // 创建CUDA事件
        preprocess_done = std::unique_ptr<CudaEvent>(new CudaEvent(cudaEventDisableTiming));
        inference_done = std::unique_ptr<CudaEvent>(new CudaEvent(cudaEventDisableTiming));
        postprocess_done = std::unique_ptr<CudaEvent>(new CudaEvent(cudaEventDisableTiming));
        
        state = BufferState::EMPTY;
    }
    
    void reset() {
        frame_id = 0;
        src_width = 0;
        src_height = 0;
        detection_count = 0;
        detections.clear();
        src_frame.release();
        result_frame.release();
        state = BufferState::EMPTY;
    }
    
    bool isReady() const {
        return state == BufferState::EMPTY;
    }
};

/**
 * @class BufferPool
 * @brief 缓冲区池 - 管理多个推理缓冲区实现流水线
 */
class BufferPool {
public:
    BufferPool(int num_buffers, int input_h, int input_w, int output_size, int max_detections)
        : num_buffers_(num_buffers) {
        
        // 使用指针数组避免拷贝问题
        for (int i = 0; i < num_buffers; ++i) {
            buffers_.push_back(std::unique_ptr<InferenceBuffer>(new InferenceBuffer()));
            buffers_.back()->initialize(i, input_h, input_w, output_size, max_detections);
        }

        std::cout << "BufferPool created with " << num_buffers << " buffers" << std::endl;
    }
    
    // 获取空闲缓冲区
    InferenceBuffer* acquireBuffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& buffer : buffers_) {
            if (buffer->state == BufferState::EMPTY) {
                buffer->state = BufferState::PREPROCESSING;
                return buffer.get();
            }
        }
        return nullptr;  // 没有空闲缓冲区
    }
    
    // 获取指定ID的缓冲区
    InferenceBuffer* getBuffer(int id) {
        if (id >= 0 && id < num_buffers_) {
            return buffers_[id].get();
        }
        return nullptr;
    }
    
    // 获取处于特定状态的缓冲区
    InferenceBuffer* getBufferByState(BufferState state) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& buffer : buffers_) {
            if (buffer->state == state) {
                return buffer.get();
            }
        }
        return nullptr;
    }
    
    // 释放缓冲区
    void releaseBuffer(InferenceBuffer* buffer) {
        if (buffer) {
            std::lock_guard<std::mutex> lock(mutex_);
            buffer->reset();
        }
    }
    
    int size() const { return num_buffers_; }
    
    // 获取所有缓冲区状态统计
    std::string getStatusString() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string status = "BufferPool status: ";
        for (const auto& buffer : buffers_) {
            status += "[" + std::to_string(buffer->buffer_id) + ":";
            switch (buffer->state) {
                case BufferState::EMPTY: status += "EMPTY"; break;
                case BufferState::PREPROCESSING: status += "PREP"; break;
                case BufferState::READY_INFER: status += "R_INF"; break;
                case BufferState::INFERRING: status += "INF"; break;
                case BufferState::READY_POST: status += "R_POST"; break;
                case BufferState::POSTPROCESSING: status += "POST"; break;
                case BufferState::READY_OUTPUT: status += "R_OUT"; break;
                case BufferState::OUTPUT: status += "OUT"; break;
            }
            status += "] ";
        }
        return status;
    }

private:
    std::vector<std::unique_ptr<InferenceBuffer>> buffers_;
    int num_buffers_;
    mutable std::mutex mutex_;
};

} // namespace jetson

#endif // INFERENCE_BUFFER_H
