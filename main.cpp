/**
 * @file main.cpp
 * @brief YOLO 推理系统主程序 - 使用高性能流水线架构
 * @description 针对 Jetson Orin NX 优化的视频目标检测系统
 */

#include "pipeline.h"
#include "config_loader.h"
#include "gst_decoder.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <atomic>
#include <sys/stat.h>
#include <sys/types.h>

// NVIDIA Nsight Systems 性能分析头文件
#ifdef NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif

// 全局停止标志
std::atomic<bool> g_stop{false};

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", stopping..." << std::endl;
    g_stop = true;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --config <path>   Configuration file path (default: ../config/default.yaml)" << std::endl;
    std::cout << "  -s, --simple          Use simple (synchronous) pipeline" << std::endl;
    std::cout << "  -p, --profile         Enable performance profiling" << std::endl;
    std::cout << "  -h, --help            Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 解析命令行参数
    std::string config_path = "../config/default.yaml";
    bool use_simple_pipeline = false;
    bool enable_profiling = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_path = argv[++i];
            }
        } else if (arg == "-s" || arg == "--simple") {
            use_simple_pipeline = true;
        } else if (arg == "-p" || arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Jetson YOLO Inference System v2.0    " << std::endl;
    std::cout << "  Optimized for Jetson Orin NX         " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载配置
    ConfigLoader config_loader;
    if (!config_loader.loadConfig(config_path)) {
        std::cerr << "Failed to load config: " << config_path << std::endl;
        return -1;
    }
    
    const auto& model_config = config_loader.getModelConfig();
    const auto& input_config = config_loader.getInputConfig();
    const auto& output_config = config_loader.getOutputConfig();
    const auto& display_config = config_loader.getDisplayConfig();
    const auto& performance_config = config_loader.getPerformanceConfig();
    
    // 设置GPU
    cudaSetDevice(performance_config.gpu_id);
    
    // 打印GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, performance_config.gpu_id);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    
    // 初始化视频捕获
    cv::VideoCapture cap;
    // if (input_config.type == "camera") {
    //     cap.open(input_config.camera_id);
    // } else if (input_config.type == "video") {
    //     cap.open(input_config.video_path);
    // } else {
    //     std::cerr << "Unsupported input type: " << input_config.type << std::endl;
    //     return -1;
    // }
    
    // if (!cap.isOpened()) {
    //     std::cerr << "Failed to open input source" << std::endl;
    //     return -1;
    // }
    
    // int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // double input_fps = cap.get(cv::CAP_PROP_FPS);
    std::unique_ptr<jetson::GstDecoder> gst_decoder;
    bool use_gstreamer = false;
    int frame_width = 0, frame_height = 0;
    double input_fps = 25.0;
    // 调试：显示解码器配置
    std::cout << "Decoder config: " << (input_config.decoder == DecoderType::GSTREAMER_NVDEC ? "gstreamer_nvdec" : "opencv") << std::endl;
    
#ifdef USE_GSTREAMER
    std::cout << "GStreamer support: ENABLED" << std::endl;
    // 检查是否使用 GStreamer 解码器
    if (input_config.decoder == DecoderType::GSTREAMER_NVDEC) {
        std::cout << "Checking GStreamer availability..." << std::endl;
        if (jetson::checkGstreamerAvailable()) {
            std::cout << "Using GStreamer + NVDEC decoder" << std::endl;
            jetson::printGstreamerInfo();
            
            // 创建 GStreamer 解码器 (输出 BGR 格式用于兼容)
            gst_decoder = std::make_unique<jetson::GstDecoder>(false);
            
            if (input_config.type == "video") {
                if (!gst_decoder->openFile(input_config.video_path)) {
                    std::cerr << "Failed to open video with GStreamer, falling back to OpenCV" << std::endl;
                    gst_decoder.reset();
                } else {
                    use_gstreamer = true;
                    frame_width = gst_decoder->getWidth();
                    frame_height = gst_decoder->getHeight();
                    input_fps = gst_decoder->getFps();
                }
            } else if (input_config.type == "camera") {
                if (!gst_decoder->openCamera(input_config.camera_id, 1920, 1080, 30)) {
                    std::cerr << "Failed to open camera with GStreamer, falling back to OpenCV" << std::endl;
                    gst_decoder.reset();
                } else {
                    use_gstreamer = true;
                    frame_width = gst_decoder->getWidth();
                    frame_height = gst_decoder->getHeight();
                    input_fps = gst_decoder->getFps();
                }
            } else if (input_config.type == "rtsp") {
                if (!gst_decoder->openRtsp(input_config.rtsp_url)) {
                    std::cerr << "Failed to open RTSP stream with GStreamer, falling back to OpenCV" << std::endl;
                    gst_decoder.reset();
                } else {
                    use_gstreamer = true;
                    frame_width = gst_decoder->getWidth();
                    frame_height = gst_decoder->getHeight();
                    input_fps = gst_decoder->getFps();
                }
            } else if (input_config.type == "rtmp") {
                if (!gst_decoder->openRtmp(input_config.rtmp_url)) {
                    std::cerr << "Failed to open RTMP stream with GStreamer, falling back to OpenCV" << std::endl;
                    gst_decoder.reset();
                } else {
                    use_gstreamer = true;
                    frame_width = gst_decoder->getWidth();
                    frame_height = gst_decoder->getHeight();
                    input_fps = gst_decoder->getFps();
                }
            }
        } else {
            std::cout << "GStreamer NVDEC not available, falling back to OpenCV" << std::endl;
        }
    }
#endif
    
    // 如果不使用 GStreamer，使用 OpenCV
    if (!use_gstreamer) {
        std::cout << "Using OpenCV decoder" << std::endl;
        if (input_config.type == "camera") {
            cap.open(input_config.camera_id);
        } else if (input_config.type == "video") {
            cap.open(input_config.video_path);
        } else if (input_config.type == "rtsp") {
            cap.open(input_config.rtsp_url);
        } else if (input_config.type == "rtmp") {
            cap.open(input_config.rtmp_url);
        } else {
            std::cerr << "Unsupported input type: " << input_config.type << std::endl;
            return -1;
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Failed to open input source" << std::endl;
            return -1;
        }
        
        frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        input_fps = cap.get(cv::CAP_PROP_FPS);
    }
    
    std::cout << "Input: " << frame_width << "x" << frame_height << " @ " << input_fps << " fps" << std::endl;
    
    // 创建输出目录结构
    std::string output_root = "output";
    std::string experiment_dir;
    int experiment_id = 0;
    
    // 确保输出根目录存在
    struct stat st;
    if (stat(output_root.c_str(), &st) == -1) {
        mkdir(output_root.c_str(), 0755);
    }
    
    // 查找最新的实验编号
    while (true) {
        experiment_dir = output_root + "/experiment_" + std::to_string(experiment_id);
        if (stat(experiment_dir.c_str(), &st) == -1) {
            mkdir(experiment_dir.c_str(), 0755);
            break;
        }
        experiment_id++;
    }
    
    std::cout << "Experiment directory: " << experiment_dir << std::endl;
    
    // 初始化视频写入器
    cv::VideoWriter video_writer;
    if (output_config.save_video) {
        std::string video_path = experiment_dir + "/" + output_config.output_path;
        int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
        video_writer.open(video_path, fourcc, input_fps, 
                          cv::Size(frame_width, frame_height));
        if (!video_writer.isOpened()) {
            std::cerr << "Warning: Failed to open video writer" << std::endl;
        }
    }
    
    // FPS日志文件
    std::ofstream fps_log;
    if (output_config.fps_log) {
        std::string fps_path = experiment_dir + "/" + output_config.fps_log_path;
        fps_log.open(fps_path);
    }
    
    // 检测结果文件
    std::ofstream detections_log;
    if (output_config.save_detections) {
        std::string detections_path = experiment_dir + "/" + output_config.detections_path;
        detections_log.open(detections_path);
    }
    
    // ========================================================================
    // 根据配置选择流水线模式
    // ========================================================================
    
    if (use_simple_pipeline || !performance_config.async_mode) {
        // 简单同步流水线
        std::cout << "Using Simple (Synchronous) Pipeline" << std::endl;
        if (performance_config.use_dla) {
            std::cout << "DLA acceleration: Enabled (core " << performance_config.dla_core << ")" << std::endl;
        }
        
        jetson::SimplePipeline pipeline(model_config, 
                                         performance_config.preprocess_backend,
                                         performance_config.use_dla,
                                         performance_config.dla_core);
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize pipeline" << std::endl;
            return -1;
        }
        
        cv::Mat frame;
        jetson::FrameResult result;
        uint64_t frame_count = 0;
        double total_time = 0;
                // 帧读取 lambda
        auto readFrame = [&]() -> bool {
#ifdef USE_GSTREAMER
            if (use_gstreamer && gst_decoder) {
                jetson::GstFrame gst_frame;
                if (!gst_decoder->read(gst_frame)) {
                    return false;
                }
                frame = gst_frame.bgr_frame;
                return !frame.empty();
            }
#endif
            return cap.read(frame);
        };
        
        std::cout << "Processing started. Press '" << display_config.quit_key << "' to quit." << std::endl;
        
        // NVIDIA Nsight Systems 性能分析标记
        #ifdef NVTX_ENABLED
        nvtxRangePushA("Main Processing Loop");
        #endif
        
        while (!g_stop) {
            // NVIDIA Nsight Systems 性能分析标记 - 读取帧
            #ifdef NVTX_ENABLED
            nvtxRangePushA("Read Frame");
            #endif
            
            if (!readFrame()) {
                // NVIDIA Nsight Systems 性能分析标记 - 读取帧结束
                #ifdef NVTX_ENABLED
                nvtxRangePop();
                #endif
                
                std::cout << "End of input" << std::endl;
                break;
            }
            
            // NVIDIA Nsight Systems 性能分析标记 - 读取帧结束
            #ifdef NVTX_ENABLED
            nvtxRangePop();
            
            // NVIDIA Nsight Systems 性能分析标记 - 处理帧
            nvtxRangePushA("Process Frame");
            #endif
            
            if (!pipeline.process(frame, result)) {
                // NVIDIA Nsight Systems 性能分析标记 - 处理帧结束
                #ifdef NVTX_ENABLED
                nvtxRangePop();
                #endif
                
                std::cerr << "Processing failed" << std::endl;
                continue;
            }
            
            // NVIDIA Nsight Systems 性能分析标记 - 处理帧结束
            #ifdef NVTX_ENABLED
            nvtxRangePop();
            #endif
            
            frame_count++;
            total_time += result.processing_time_ms;
            double avg_fps = 1000.0 * frame_count / total_time;
            
            // 在帧上绘制FPS
            cv::putText(result.result_frame, 
                        cv::format("FPS: %.1f | Detections: %d", avg_fps, result.detection_count),
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                        cv::Scalar(0, 255, 0), 2);
            
            // 显示检测结果
            if (display_config.show_video) {
                // 显示视频图像检测结果
                cv::imshow(display_config.imshow_name, result.result_frame);
            } else {
                // 只展示检测结果和检测时间
                std::cout << "Frame " << frame_count << ": " 
                          << result.detection_count << " detections, " 
                          << result.processing_time_ms << " ms" << std::endl;
            }
            
            // 保存视频
            if (video_writer.isOpened()) {
                video_writer.write(result.result_frame);
            }
            
            // FPS日志
            if (fps_log.is_open()) {
                fps_log << frame_count << "," << avg_fps << "," 
                        << result.processing_time_ms << std::endl;
            }
            
            // 检测结果日志
            if (detections_log.is_open()) {
                detections_log << "Frame " << frame_count << ": " 
                              << result.detection_count << " detections, " 
                              << result.processing_time_ms << " ms" << std::endl;
            }
            
            // 检查退出
            int key = 0;
            if (display_config.show_video) {
                key = cv::waitKey(1) & 0xFF;
            } else {
                // 非视频模式下，每秒检查一次退出
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                if (g_stop) {
                    break;
                }
            }
            
            if (key == display_config.quit_key || key == 27) {
                break;
            }
        }
        
        // NVIDIA Nsight Systems 性能分析标记
        #ifdef NVTX_ENABLED
        nvtxRangePop();
        #endif
        
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Average FPS: " << (1000.0 * frame_count / total_time) << std::endl;
        
    } else {
        // 高性能异步流水线
        std::cout << "Using High-Performance Async Pipeline" << std::endl;
        if (performance_config.use_dla) {
            std::cout << "DLA acceleration: Enabled (core " << performance_config.dla_core << ")" << std::endl;
        }
        
        // NVIDIA Nsight Systems 性能分析标记 - 异步流水线
        #ifdef NVTX_ENABLED
        nvtxRangePushA("Async Pipeline Processing");
        #endif
        jetson::PipelineConfig pipeline_config;
        pipeline_config.num_buffers = performance_config.queue_size;
        pipeline_config.num_streams = performance_config.queue_size;
        pipeline_config.enable_profiling = enable_profiling;
        pipeline_config.max_pending_frames = performance_config.queue_size + 2;
        pipeline_config.cpu_ids = performance_config.cpu_ids;
        pipeline_config.preprocess_backend = performance_config.preprocess_backend;
        pipeline_config.use_dla = performance_config.use_dla;
        pipeline_config.dla_core = performance_config.dla_core;
        
        jetson::Pipeline pipeline(model_config, pipeline_config);
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize pipeline" << std::endl;
            return -1;
        }
        
        std::cout << "Processing started. Press '" << display_config.quit_key << "' to quit." << std::endl;
        
        cv::Mat frame;
        uint64_t submitted_frames = 0;
        uint64_t processed_frames = 0;
        // 帧读取 lambda (异步流水线版本)
        auto readFrameAsync = [&]() -> bool {
#ifdef USE_GSTREAMER
            if (use_gstreamer && gst_decoder) {
                jetson::GstFrame gst_frame;
                if (!gst_decoder->read(gst_frame)) {
                    return false;
                }
                frame = gst_frame.bgr_frame;
                return !frame.empty();
            }
#endif
            return cap.read(frame);
        };
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (!g_stop) {
            // 读取并提交帧
            if (readFrameAsync()) {
                if (pipeline.submitFrame(frame, submitted_frames)) {
                    submitted_frames++;
                } else {
                    // 队列满，稍微等待，而不是立即丢弃
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            } else {
                // 视频结束
                std::cout << "End of input" << std::endl;
                
                // 等待剩余帧处理完成
                while (pipeline.getPendingCount() > 0) {
                    jetson::FrameResult result;
                    if (pipeline.getResult(result, 100)) {
                        processed_frames++;
                        
                        // 显示和保存
                        cv::putText(result.result_frame, 
                                    cv::format("FPS: %.1f | Detections: %d", 
                                               pipeline.getFPS(), result.detection_count),
                                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                                    cv::Scalar(0, 255, 0), 2);
                        
                        cv::imshow(display_config.imshow_name, result.result_frame);
                        
                        if (video_writer.isOpened()) {
                            video_writer.write(result.result_frame);
                        }
                        
                        cv::waitKey(1);
                    }
                }
                break;
            }
            
            // 获取处理结果（非阻塞）
            jetson::FrameResult result;
            while (pipeline.tryGetResult(result)) {
                processed_frames++;
                
                // 在帧上绘制信息
                cv::putText(result.result_frame, 
                            cv::format("FPS: %.1f | Detections: %d | Latency: %.1fms", 
                                       pipeline.getFPS(), result.detection_count, result.processing_time_ms),
                            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                            cv::Scalar(0, 255, 0), 2);
                
                // 显示检测结果
                if (display_config.show_video) {
                    // 显示视频图像检测结果
                    cv::imshow(display_config.imshow_name, result.result_frame);
                } else {
                    // 只展示检测结果和检测时间
                    std::cout << "Frame " << result.frame_id << ": " 
                              << result.detection_count << " detections, " 
                              << result.processing_time_ms << " ms" << std::endl;
                }
                
                // 保存视频
                if (video_writer.isOpened()) {
                    video_writer.write(result.result_frame);
                }
                
                // FPS日志
                if (fps_log.is_open()) {
                    fps_log << result.frame_id << "," << pipeline.getFPS() << "," 
                            << result.processing_time_ms << std::endl;
                }
                
                // 检测结果日志
                if (detections_log.is_open()) {
                    detections_log << "Frame " << result.frame_id << ": " 
                                  << result.detection_count << " detections, " 
                                  << result.processing_time_ms << " ms" << std::endl;
                }
            }
            
            // 检查退出
            int key = 0;
            if (display_config.show_video) {
                key = cv::waitKey(1) & 0xFF;
                if (key == display_config.quit_key || key == 27) {
                    break;
                }
            } else {
                // 非视频模式下，检查全局停止标志
                if (g_stop) {
                    break;
                }
            }
        }
        
        // 停止流水线
        pipeline.stop();
        
        // NVIDIA Nsight Systems 性能分析标记 - 异步流水线结束
        #ifdef NVTX_ENABLED
        nvtxRangePop();
        #endif
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "========================================" << std::endl;
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Submitted frames: " << submitted_frames << std::endl;
        std::cout << "  Processed frames: " << processed_frames << std::endl;
        std::cout << "  Total time: " << total_seconds << " s" << std::endl;
        std::cout << "  Overall FPS: " << processed_frames / total_seconds << std::endl;
        
        auto stats = pipeline.getStats();
        std::cout << "  Avg preprocess: " << stats.avg_preprocess_ms << " ms" << std::endl;
        std::cout << "  Avg inference: " << stats.avg_inference_ms << " ms" << std::endl;
        std::cout << "  Avg postprocess: " << stats.avg_postprocess_ms << " ms" << std::endl;
        std::cout << "  Avg total: " << stats.avg_total_ms << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    // 清理
    cap.release();
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    if (fps_log.is_open()) {
        fps_log.close();
    }
    cv::destroyAllWindows();
    
    std::cout << "Done." << std::endl;
    return 0;
}
