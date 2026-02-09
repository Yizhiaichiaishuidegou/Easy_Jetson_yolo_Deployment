/**
 * @file yolo_engine.h
 * @brief 高性能 YOLO 推理引擎
 * @description 支持多缓冲区、多CUDA流、零拷贝内存的推理引擎
 */

#ifndef YOLO_ENGINE_H
#define YOLO_ENGINE_H

#include "cuda_utils.h"
#include "inference_buffer.h"
#include "config_loader.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <NvInferRuntime.h>

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>


#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace jetson {

// 预处理参数
struct PreprocessParams {
    float scale = 1.0f / 255.0f;
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {1.0f, 1.0f, 1.0f};
    bool swap_rb = true;
    uint8_t pad_value = 114;
};

// 后处理参数
struct PostprocessParams {
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = 1024;
};

// TensorRT Logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

/**
 * @class YoloEngine
 * @brief 高性能 YOLO 推理引擎
 * 
 * 特性:
 * - 支持多缓冲区并行处理
 * - 使用多CUDA流实现GPU流水线
 * - 预处理、推理、后处理可在不同缓冲区并行执行
 * - 支持零拷贝内存减少数据传输开销
 * - 支持 DLA 加速 (Jetson Orin NX)
 */
class YoloEngine {
public:
    YoloEngine(const ModelConfig& config,
                PreprocessBackend backend = PreprocessBackend::CUDA,
                bool use_dla = false, int dla_core = 0);
    ~YoloEngine();
    
    // 禁止拷贝
    YoloEngine(const YoloEngine&) = delete;
    YoloEngine& operator=(const YoloEngine&) = delete;
    
    // 初始化引擎
    bool initialize();
    
    // 在指定的缓冲区和流上执行预处理
    void preprocess(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream);
    
    // 在指定的缓冲区和流上执行推理
    void inference(InferenceBuffer& buffer, CudaStream& stream);
    
    // 在指定的缓冲区和流上执行后处理
    void postprocess(InferenceBuffer& buffer, CudaStream& stream);
    
    // 在帧上绘制检测结果
    void drawResults(InferenceBuffer& buffer, const std::vector<std::string>& class_names);
    
    // 获取参数
    int getInputHeight() const { return input_h_; }
    int getInputWidth() const { return input_w_; }
    int getNumClasses() const { return num_classes_; }
    int getOutputSize() const { return output_size_; }
    PreprocessBackend getPreprocessBackend() const { return preprocess_backend_; }
    
    // 设置参数
    void setPreprocessParams(const PreprocessParams& params) { preprocess_params_ = params; }
    void setPostprocessParams(const PostprocessParams& params) { postprocess_params_ = params; }
    void setPreprocessBackend(PreprocessBackend backend) { preprocess_backend_ = backend; }

private:
    // 加载TensorRT引擎
    bool loadEngine(const std::string& engine_path);
    
    // 计算仿射变换矩阵
    void computeAffineMatrix(int src_w, int src_h, float* i2d, float* d2i);

    // 不同预处理后端实现
    void preprocessCuda(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream);
    void preprocessVpiCuda(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream);
    void preprocessVpiVic(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream);
    
    // 配置
    ModelConfig config_;
    PreprocessParams preprocess_params_;
    PostprocessParams postprocess_params_;
    PreprocessBackend preprocess_backend_ = PreprocessBackend::CUDA;
    // NV12 预处理 (用于 NVDEC + VIC 硬件加速路径)
    void preprocessNV12(InferenceBuffer& buffer, 
                        const uint8_t* y_plane, const uint8_t* uv_plane,
                        int width, int height, int y_pitch, int uv_pitch,
                        CudaStream& stream);
    // DLA 配置
    bool use_dla_ = false;
    int dla_core_ = 0;
    // 模型参数
    int input_h_ = 640;
    int input_w_ = 640;
    int num_classes_ = 10;
    int num_boxes_ = 8400;
    int output_size_ = 0;

    int last_src_w_ = -1;
    int last_src_h_ = -1;
    float cached_i2d_[6];
    float cached_d2i_[6];


    
    // TensorRT 组件
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // 绑定索引
    int input_index_ = -1;
    int output_index_ = -1;
    
    // VPI 资源
    VPIStream vpi_stream_ = nullptr;
    VPIImage vpi_input_bgr_ = nullptr;     // 原始 BGR 输入
    VPIImage vpi_input_rgb_ = nullptr;     // 转换后的 RGB
    VPIImage vpi_rescaled_ = nullptr;  
    
    // 缓存上次的分辨率，避免重复分配 VPI 图像
    int last_vpi_w_ = -1;
    int last_vpi_h_ = -1;
    
    bool initialized_ = false;
};

} // namespace jetson

#endif // YOLO_ENGINE_H
