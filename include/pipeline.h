/**
 * @file pipeline.h
 * @brief 高性能异步推理流水线
 * @description 利用多缓冲区和多CUDA流实现真正的并行处理
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include "yolo_engine.h"
#include "inference_buffer.h"
#include "cuda_utils.h"
#include "config_loader.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>

namespace jetson {

/**
 * @struct PipelineConfig
 * @brief 流水线配置
 */
struct PipelineConfig {
    int num_buffers = 3;          // 缓冲区数量 (建议3个实现最佳流水线效果)
    int num_streams = 3;          // CUDA流数量
    bool enable_profiling = false; // 是否启用性能分析
    int max_pending_frames = 5;    // 最大待处理帧数
    std::vector<int> cpu_ids;      // 新增：CPU 绑定列表
    PreprocessBackend preprocess_backend = PreprocessBackend::CUDA;  // 预处理后端
    // DLA 配置
    bool use_dla = false;         // 是否使用 DLA 加速
    int dla_core = 0;             // DLA 核心 ID (0 或 1)
};

/**
 * @struct FrameResult
 * @brief 帧处理结果
 */
struct FrameResult {
    uint64_t frame_id = 0;
    cv::Mat result_frame;
    std::vector<float> detections;  // [x1,y1,x2,y2,conf,class] * N
    int detection_count = 0;
    double processing_time_ms = 0;
    bool valid = false;
};

/**
 * @class Pipeline
 * @brief 高性能异步推理流水线
 * 
 * 设计原理:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    三级流水线 (Triple Pipeline)                   │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ 时间 T0   │ Buffer0: Preprocess  │                              │
 * │ 时间 T1   │ Buffer0: Inference   │ Buffer1: Preprocess          │
 * │ 时间 T2   │ Buffer0: Postprocess │ Buffer1: Inference   │ Buffer2: Preprocess   │
 * │ 时间 T3   │ Buffer0: Output      │ Buffer1: Postprocess │ Buffer2: Inference    │
 * │ ...       │ (循环复用)            │ ...                   │ ...                   │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * 特性:
 * - 真正的并行: 预处理、推理、后处理可同时在不同帧上执行
 * - 多CUDA流: 每个缓冲区使用独立的CUDA流
 * - 零拷贝内存: 结果直接在CPU/GPU共享内存中
 * - 异步回调: 处理完成后自动调用回调函数
 */
class Pipeline {
public:
    using ResultCallback = std::function<void(FrameResult&)>;
    
    Pipeline(const ModelConfig& model_config, const PipelineConfig& pipeline_config);
    ~Pipeline();
    
    // 禁止拷贝
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    
    // 初始化流水线
    bool initialize();
    
    // 提交帧进行处理 (非阻塞)
    bool submitFrame(const cv::Mat& frame, uint64_t frame_id = 0);
    
    // 获取处理结果 (阻塞等待)
    bool getResult(FrameResult& result, int timeout_ms = 1000);
    
    // 尝试获取结果 (非阻塞)
    bool tryGetResult(FrameResult& result);
    
    // 设置结果回调 (可选)
    void setResultCallback(ResultCallback callback);
    
    // 获取当前FPS
    double getFPS() const { return current_fps_.load(); }
    
    // 获取待处理帧数
    size_t getPendingCount() const;
    
    // 停止流水线
    void stop();
    
    // 是否正在运行
    bool isRunning() const { return running_.load(); }
    
    // 获取性能统计
    struct Stats {
        double avg_preprocess_ms = 0;
        double avg_inference_ms = 0;
        double avg_postprocess_ms = 0;
        double avg_total_ms = 0;
        uint64_t total_frames = 0;
    };
    Stats getStats() const;

private:
    // 工作线程
    void inputWorker();      // 输入处理线程
    void processingWorker(); // GPU处理协调线程
    void outputWorker();     // 输出处理线程
    
    // 配置
    ModelConfig model_config_;
    PipelineConfig pipeline_config_;
    
    // 核心组件
    std::unique_ptr<YoloEngine> engine_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unique_ptr<StreamPool> stream_pool_;
    
    // 输入队列
    struct InputFrame {
        cv::Mat frame;
        uint64_t frame_id;
        std::chrono::high_resolution_clock::time_point submit_time;
    };
    std::queue<InputFrame> input_queue_;
    std::mutex input_mutex_;
    std::condition_variable input_cv_;
    
    // 输出队列
    std::queue<FrameResult> output_queue_;
    std::mutex output_mutex_;
    std::condition_variable output_cv_;
    
    // 处理中的缓冲区队列
    std::queue<int> processing_queue_;  // buffer_id
    std::mutex processing_mutex_;
    
    // 线程
    std::thread input_thread_;
    std::thread processing_thread_;
    std::thread output_thread_;
    
    // 状态
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};
    
    // 回调
    ResultCallback result_callback_;
    std::mutex callback_mutex_;
    
    // 统计
    std::atomic<double> current_fps_{0};
    std::atomic<uint64_t> frame_counter_{0};
    std::atomic<uint64_t> processed_counter_{0}; // 新增：已处理帧计数
    
    // 性能统计 (移动平均)
    std::atomic<double> avg_preprocess_ms_{0};
    std::atomic<double> avg_inference_ms_{0};
    std::atomic<double> avg_postprocess_ms_{0};
    std::atomic<double> avg_total_ms_{0};
    
    // FPS计算
    std::chrono::high_resolution_clock::time_point last_fps_time_;
    uint64_t last_fps_frame_count_ = 0;
};

/**
 * @class SimplePipeline
 * @brief 简化版流水线 - 单线程同步处理
 * @description 适用于调试或低延迟场景
 */
class SimplePipeline {
public:
    SimplePipeline(const ModelConfig& model_config, 
                   PreprocessBackend backend = PreprocessBackend::CUDA,
                    bool use_dla = false, int dla_core = 0);
    ~SimplePipeline();
    
    bool initialize();
    
    // 同步处理单帧
    bool process(const cv::Mat& frame, FrameResult& result);
    
    double getLastProcessingTime() const { return last_processing_time_; }

private:
    ModelConfig model_config_;
    PreprocessBackend preprocess_backend_;
    bool use_dla_ = false;
    int dla_core_ = 0;
    std::unique_ptr<YoloEngine> engine_;
    std::unique_ptr<CudaStream> stream_;
    std::unique_ptr<InferenceBuffer> buffer_;
    
    double last_processing_time_ = 0;
    bool initialized_ = false;
};

} // namespace jetson

#endif // PIPELINE_H
