/**
 * @file pipeline.cpp
 * @brief 高性能异步推理流水线实现
 */

#include "pipeline.h"
#include <iostream>
#include <pthread.h>

namespace jetson {

void setThreadAffinity(const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) return;
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int id : cpu_ids) {
        CPU_SET(id, &cpuset);
    }
    
    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "Warning: Failed to set thread affinity" << std::endl;
    }
}

// ============================================================================
// Pipeline 实现
// ============================================================================

Pipeline::Pipeline(const ModelConfig& model_config, const PipelineConfig& pipeline_config)
    : model_config_(model_config)
    , pipeline_config_(pipeline_config)
{
    last_fps_time_ = std::chrono::high_resolution_clock::now();
    
    // 初始化滑动窗口
    preprocess_times_.reserve(WINDOW_SIZE);
    inference_times_.reserve(WINDOW_SIZE);
    postprocess_times_.reserve(WINDOW_SIZE);
    wait_times_.reserve(WINDOW_SIZE);
    total_times_.reserve(WINDOW_SIZE);
    frame_times_.reserve(WINDOW_SIZE);
}

Pipeline::~Pipeline() {
    stop();
    // 按正确顺序销毁资源
    buffer_pool_.reset();
    stream_pool_.reset();
    engine_.reset();
}

bool Pipeline::initialize() {
    if (initialized_) return true;
    
    try {
        // 创建YOLO引擎
        std::cout << "DEBUG: Creating YoloEngine..." << std::endl << std::flush;
        engine_ = std::make_unique<YoloEngine>(model_config_,
                                                pipeline_config_.preprocess_backend,
                                                pipeline_config_.use_dla,
                                                pipeline_config_.dla_core);
        std::cout << "DEBUG: Initializing YoloEngine..." << std::endl << std::flush;
        if (!engine_->initialize()) {
            std::cerr << "Failed to initialize YOLO engine" << std::endl;
            return false;
        }
        std::cout << "DEBUG: YoloEngine initialized, creating CUDA stream..." << std::endl << std::flush;
        
        // 创建CUDA流池
        stream_pool_ = std::make_unique<StreamPool>(pipeline_config_.num_streams);
        std::cout << "DEBUG: Creating buffer..." << std::endl << std::flush;
        
        // 创建缓冲区池
        buffer_pool_ = std::make_unique<BufferPool>(
            pipeline_config_.num_buffers,
            engine_->getInputHeight(),
            engine_->getInputWidth(),
            engine_->getOutputSize(),
            1024  // max detections
        );
        std::cout << "DEBUG: Buffer initialized" << std::endl << std::flush;
        initialized_ = true;
        
        // 启动工作线程
        running_ = true;
        input_thread_ = std::thread(&Pipeline::inputWorker, this);
        processing_thread_ = std::thread(&Pipeline::processingWorker, this);
        output_thread_ = std::thread(&Pipeline::outputWorker, this);
        
        std::cout << "Pipeline initialized with " 
                  << pipeline_config_.num_buffers << " buffers and "
                  << pipeline_config_.num_streams << " streams" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Pipeline initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void Pipeline::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // 通知所有等待的线程
    input_cv_.notify_all();
    output_cv_.notify_all();
    
    // 等待线程结束
    if (input_thread_.joinable()) input_thread_.join();
    if (processing_thread_.joinable()) processing_thread_.join();
    if (output_thread_.joinable()) output_thread_.join();
    
    // 同步所有CUDA流
    if (stream_pool_) {
        stream_pool_->synchronizeAll();
    }
    
    std::cout << "Pipeline stopped" << std::endl;
}

bool Pipeline::submitFrame(const cv::Mat& frame, uint64_t frame_id) {
    if (!running_ || !initialized_) return false;
    
    std::unique_lock<std::mutex> lock(input_mutex_);
    
    // 检查队列是否已满
    if (static_cast<int>(input_queue_.size()) >= pipeline_config_.max_pending_frames) {
        return false;  // 丢弃帧
    }
    
    InputFrame input_frame;
    input_frame.frame = frame.clone();
    input_frame.frame_id = (frame_id == 0) ? frame_counter_++ : frame_id;
    if (frame_id != 0) frame_counter_++; 
    input_frame.submit_time = std::chrono::high_resolution_clock::now();
    
    input_queue_.push(std::move(input_frame));
    lock.unlock();
    
    input_cv_.notify_one();
    return true;
}

bool Pipeline::getResult(FrameResult& result, int timeout_ms) {
    std::unique_lock<std::mutex> lock(output_mutex_);
    
    if (output_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [this] { return !output_queue_.empty() || !running_; })) {
        if (!output_queue_.empty()) {
            result = std::move(output_queue_.front());
            output_queue_.pop();
            return true;
        }
    }
    
    return false;
}

bool Pipeline::tryGetResult(FrameResult& result) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    
    if (!output_queue_.empty()) {
        result = std::move(output_queue_.front());
        output_queue_.pop();
        return true;
    }
    
    return false;
}

void Pipeline::setResultCallback(ResultCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    result_callback_ = std::move(callback);
}

size_t Pipeline::getPendingCount() const {
    // 这是一个近似值，因为没有加锁
    return input_queue_.size() + processing_queue_.size();
}

Pipeline::Stats Pipeline::getStats() const {
    Stats stats;
    stats.avg_preprocess_ms = avg_preprocess_ms_.load();
    stats.avg_inference_ms = avg_inference_ms_.load();
    stats.avg_postprocess_ms = avg_postprocess_ms_.load();
    stats.avg_wait_ms = avg_wait_ms_.load();
    stats.avg_total_ms = avg_total_ms_.load();
    stats.total_frames = frame_counter_.load();
    return stats;
}

void Pipeline::inputWorker() {
    setThreadAffinity(pipeline_config_.cpu_ids);
    std::cout << "Input worker started" << std::endl;
    
    while (running_) {
        InputFrame input_frame;
        
        // 等待输入
        {
            std::unique_lock<std::mutex> lock(input_mutex_);
            input_cv_.wait(lock, [this] { 
                return !input_queue_.empty() || !running_; 
            });
            
            if (!running_ && input_queue_.empty()) break;
            if (input_queue_.empty()) continue;
            
            input_frame = std::move(input_queue_.front());
            input_queue_.pop();
        }
        
        // 获取空闲缓冲区
        InferenceBuffer* buffer = nullptr;
        while (running_ && !buffer) {
            buffer = buffer_pool_->acquireBuffer();
            if (!buffer) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        if (!buffer || !running_) continue;
        
        // 设置帧信息
        buffer->frame_id = input_frame.frame_id;
        buffer->timestamp_push = std::chrono::duration<double, std::milli>(
            input_frame.submit_time.time_since_epoch()).count();
        
        // 执行预处理
        auto& stream = stream_pool_->get(buffer->buffer_id);
        
        buffer->timestamp_preprocess_start = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        engine_->preprocess(*buffer, input_frame.frame, stream);
        
        buffer->state = BufferState::READY_INFER;
        
        // 添加到处理队列
        {
            std::lock_guard<std::mutex> lock(processing_mutex_);
            processing_queue_.push(buffer->buffer_id);
        }
    }
    
    std::cout << "Input worker stopped" << std::endl;
}

void Pipeline::processingWorker() {
    setThreadAffinity(pipeline_config_.cpu_ids);
    std::cout << "Processing worker started" << std::endl;
    
    while (running_) {
        int buffer_id = -1;
        
        // 检查处理队列
        {
            std::lock_guard<std::mutex> lock(processing_mutex_);
            if (!processing_queue_.empty()) {
                buffer_id = processing_queue_.front();
                processing_queue_.pop();
            }
        }
        
        if (buffer_id < 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        
        InferenceBuffer* buffer = buffer_pool_->getBuffer(buffer_id);
        if (!buffer) continue;
        
        auto& stream = stream_pool_->get(buffer_id);
        
        // 执行推理
        if (buffer->state == BufferState::READY_INFER) {
            buffer->state = BufferState::INFERRING;
            buffer->timestamp_inference_start = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            engine_->inference(*buffer, stream);
            
            buffer->state = BufferState::READY_POST;
        }
        
        // 执行后处理
        if (buffer->state == BufferState::READY_POST) {
            buffer->state = BufferState::POSTPROCESSING;
            buffer->timestamp_postprocess_start = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            engine_->postprocess(*buffer, stream);
            
            // 同步流确保后处理完成
            stream.synchronize();
            
            buffer->timestamp_postprocess_end = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            // 绘制结果
            engine_->drawResults(*buffer, model_config_.class_names);
            
            buffer->state = BufferState::READY_OUTPUT;
            
            // 创建输出结果
            FrameResult result;
            result.frame_id = buffer->frame_id;
            result.result_frame = buffer->result_frame.clone();
            result.detections = buffer->detections;
            result.detection_count = buffer->detection_count;
            result.processing_time_ms = buffer->timestamp_postprocess_end - buffer->timestamp_push;
            result.valid = true;
            processed_counter_++;
            // 计算各阶段时间
            double preprocess_time = buffer->timestamp_inference_start - buffer->timestamp_preprocess_start;
            double inference_time = buffer->timestamp_postprocess_start - buffer->timestamp_inference_start;
            double postprocess_time = buffer->timestamp_postprocess_end - buffer->timestamp_postprocess_start;
            double wait_time = buffer->timestamp_preprocess_start - buffer->timestamp_push;
            double total_time = result.processing_time_ms;
            
            // 更新滑动窗口统计
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                
                // 更新时间窗口
                preprocess_times_.push_back(preprocess_time);
                inference_times_.push_back(inference_time);
                postprocess_times_.push_back(postprocess_time);
                wait_times_.push_back(wait_time);
                total_times_.push_back(total_time);
                
                // 保持窗口大小
                if (preprocess_times_.size() > WINDOW_SIZE) {
                    preprocess_times_.erase(preprocess_times_.begin());
                    inference_times_.erase(inference_times_.begin());
                    postprocess_times_.erase(postprocess_times_.begin());
                    wait_times_.erase(wait_times_.begin());
                    total_times_.erase(total_times_.begin());
                }
                
                // 更新帧时间窗口
                frame_times_.push_back(std::chrono::high_resolution_clock::now());
                if (frame_times_.size() > WINDOW_SIZE) {
                    frame_times_.erase(frame_times_.begin());
                }
                
                // 计算滑动窗口平均值
                auto calculateAverage = [](const std::vector<double>& times) {
                    if (times.empty()) return 0.0;
                    double sum = 0.0;
                    for (double time : times) sum += time;
                    return sum / times.size();
                };
                
                avg_preprocess_ms_ = calculateAverage(preprocess_times_);
                avg_inference_ms_ = calculateAverage(inference_times_);
                avg_postprocess_ms_ = calculateAverage(postprocess_times_);
                avg_wait_ms_ = calculateAverage(wait_times_);
                avg_total_ms_ = calculateAverage(total_times_);
                
                // 使用滑动窗口计算FPS
                if (frame_times_.size() >= 2) {
                    double window_duration = std::chrono::duration<double>(
                        frame_times_.back() - frame_times_.front()).count();
                    if (window_duration > 0) {
                        current_fps_ = (frame_times_.size() - 1) / window_duration;
                    }
                }
            }
            
            // 释放缓冲区
            buffer_pool_->releaseBuffer(buffer);
            
            // 添加到输出队列
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                output_queue_.push(std::move(result));
            }
            output_cv_.notify_one();
            
            // 调用回调
            {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (result_callback_) {
                    FrameResult callback_result = output_queue_.back();
                    result_callback_(callback_result);
                }
            }
        }
    }
    
    std::cout << "Processing worker stopped" << std::endl;
}

void Pipeline::outputWorker() {
    setThreadAffinity(pipeline_config_.cpu_ids);
    std::cout << "Output worker started" << std::endl;
    
    // 输出工作线程目前只用于监控，主要输出通过 getResult() 获取
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (pipeline_config_.enable_profiling) {
                auto stats = getStats();
                std::cout << "Pipeline Stats - "
                          << "FPS: " << current_fps_.load() 
                          << " Prep: " << stats.avg_preprocess_ms << "ms"
                          << " Infer: " << stats.avg_inference_ms << "ms"
                          << " Post: " << stats.avg_postprocess_ms << "ms"
                          << " Wait: " << stats.avg_wait_ms << "ms"
                          << " Total: " << stats.avg_total_ms << "ms"
                          << std::endl;
            }
    }
    
    std::cout << "Output worker stopped" << std::endl;
}

// ============================================================================
// SimplePipeline 实现
// ============================================================================

SimplePipeline::SimplePipeline(const ModelConfig& model_config, PreprocessBackend backend,
                               bool use_dla, int dla_core)
    : model_config_(model_config), preprocess_backend_(backend), 
      use_dla_(use_dla), dla_core_(dla_core)
{
    // 初始化滑动窗口
    preprocess_times_.reserve(WINDOW_SIZE);
    inference_times_.reserve(WINDOW_SIZE);
    postprocess_times_.reserve(WINDOW_SIZE);
    total_times_.reserve(WINDOW_SIZE);
    frame_times_.reserve(WINDOW_SIZE);
}

SimplePipeline::~SimplePipeline() {
    // 按正确顺序销毁资源
    // 1. 先同步 CUDA 流，确保所有操作完成
    if (stream_) {
        stream_->synchronize();
    }
    // 2. 销毁 buffer（可能包含 CUDA 事件等资源）
    buffer_.reset();
    // 3. 销毁 stream
    stream_.reset();
    // 4. 最后销毁 engine
    engine_.reset();
}

bool SimplePipeline::initialize() {
    if (initialized_) return true;
    
    try {
        // 创建YOLO引擎
        std::cout << "DEBUG: SimplePipeline creating YoloEngine..." << std::endl << std::flush;
        engine_ = std::make_unique<YoloEngine>(model_config_, preprocess_backend_, use_dla_, dla_core_);
        std::cout << "DEBUG: SimplePipeline initializing YoloEngine..." << std::endl << std::flush;
        if (!engine_->initialize()) {
            std::cerr << "Failed to initialize YOLO engine" << std::endl;
            return false;
        }
        std::cout << "DEBUG: SimplePipeline YoloEngine initialized" << std::endl << std::flush;
        
        // 创建CUDA流
        stream_ = std::make_unique<CudaStream>();
        
        // 创建缓冲区
        buffer_ = std::make_unique<InferenceBuffer>();
        buffer_->initialize(
            0,
            engine_->getInputHeight(),
            engine_->getInputWidth(),
            engine_->getOutputSize(),
            1024
        );
        
        initialized_ = true;
        std::cout << "SimplePipeline initialized" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SimplePipeline initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool SimplePipeline::process(const cv::Mat& frame, FrameResult& result) {
    if (!initialized_) {
        std::cerr << "SimplePipeline not initialized" << std::endl;
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 重置缓冲区
    buffer_->reset();
    buffer_->state = BufferState::PREPROCESSING;
    
    // 预处理
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    engine_->preprocess(*buffer_, frame, *stream_);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    // 推理
    buffer_->state = BufferState::INFERRING;
    auto inference_start = std::chrono::high_resolution_clock::now();
    engine_->inference(*buffer_, *stream_);
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    // 后处理
    buffer_->state = BufferState::POSTPROCESSING;
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    engine_->postprocess(*buffer_, *stream_);
    
    // 同步
    stream_->synchronize();
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    
    // 绘制结果
    engine_->drawResults(*buffer_, model_config_.class_names);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 计算各阶段时间
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    double total_time = last_processing_time_;
    
    // 更新滑动窗口统计
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        // 更新时间窗口
        preprocess_times_.push_back(preprocess_time);
        inference_times_.push_back(inference_time);
        postprocess_times_.push_back(postprocess_time);
        total_times_.push_back(total_time);
        
        // 保持窗口大小
        if (preprocess_times_.size() > WINDOW_SIZE) {
            preprocess_times_.erase(preprocess_times_.begin());
            inference_times_.erase(inference_times_.begin());
            postprocess_times_.erase(postprocess_times_.begin());
            total_times_.erase(total_times_.begin());
        }
        
        // 更新帧时间窗口
        frame_times_.push_back(end);
        if (frame_times_.size() > WINDOW_SIZE) {
            frame_times_.erase(frame_times_.begin());
        }
        
        // 计算滑动窗口平均值
        auto calculateAverage = [](const std::vector<double>& times) {
            if (times.empty()) return 0.0;
            double sum = 0.0;
            for (double time : times) sum += time;
            return sum / times.size();
        };
        
        avg_preprocess_ms_ = calculateAverage(preprocess_times_);
        avg_inference_ms_ = calculateAverage(inference_times_);
        avg_postprocess_ms_ = calculateAverage(postprocess_times_);
        avg_total_ms_ = calculateAverage(total_times_);
        
        // 使用滑动窗口计算FPS
        if (frame_times_.size() >= 2) {
            double window_duration = std::chrono::duration<double>(
                frame_times_.back() - frame_times_.front()).count();
            if (window_duration > 0) {
                current_fps_ = (frame_times_.size() - 1) / window_duration;
            }
        }
    }
    
    // 填充结果
    result.frame_id = buffer_->frame_id;
    result.result_frame = buffer_->result_frame.clone();
    result.detections = buffer_->detections;
    result.detection_count = buffer_->detection_count;
    result.processing_time_ms = last_processing_time_;
    result.valid = true;
    
    return true;
}

SimplePipeline::Stats SimplePipeline::getStats() const {
    Stats stats;
    stats.avg_preprocess_ms = avg_preprocess_ms_;
    stats.avg_inference_ms = avg_inference_ms_;
    stats.avg_postprocess_ms = avg_postprocess_ms_;
    stats.avg_total_ms = avg_total_ms_;
    return stats;
}

} // namespace jetson
