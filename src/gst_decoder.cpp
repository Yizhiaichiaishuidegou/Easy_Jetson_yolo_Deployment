/**
 * @file gst_decoder.cpp
 * @brief GStreamer + NVDEC 硬件解码器实现
 */

#include "gst_decoder.h"

#ifdef USE_GSTREAMER

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <iostream>
#include <sstream>



namespace jetson {

GstDecoder::GstDecoder(bool output_nv12)
    : output_nv12_(output_nv12) {
    // 初始化 GStreamer (如果还没初始化)
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
}

GstDecoder::~GstDecoder() {
    release();
}

bool GstDecoder::openFile(const std::string& file_path) {
    std::ostringstream pipeline_ss;
    
    if (output_nv12_) {
        // 输出 NV12 到 GPU 内存 (用于 VIC 加速)
        pipeline_ss << "filesrc location=\"" << file_path << "\" ! "
                    << "qtdemux ! h264parse ! "
                    << "nvv4l2decoder ! "
                    << "nvvidconv ! "
                    << "video/x-raw(memory:NVMM),format=NV12 ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    } else {
        // 输出 BGR 到 CPU 内存
        pipeline_ss << "filesrc location=\"" << file_path << "\" ! "
                    << "qtdemux ! h264parse ! "
                    << "nvv4l2decoder ! "
                    << "nvvidconv ! "
                    << "video/x-raw,format=BGRx ! "
                    << "videoconvert ! "
                    << "video/x-raw,format=BGR ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    return createPipeline(pipeline_ss.str());
}

bool GstDecoder::openRtsp(const std::string& rtsp_url) {
    std::ostringstream pipeline_ss;
    
    if (output_nv12_) {
        // RTSP -> NVDEC -> NV12 (GPU)
        pipeline_ss << "rtspsrc location=\"" << rtsp_url << "\" latency=100 ! "
                    << "rtph264depay ! h264parse ! "
                    << "nvv4l2decoder ! "
                    << "nvvidconv ! "
                    << "video/x-raw(memory:NVMM),format=NV12 ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    } else {
        // RTSP -> NVDEC -> BGR (CPU)
        pipeline_ss << "rtspsrc location=\"" << rtsp_url << "\" latency=100 ! "
                    << "rtph264depay ! h264parse ! "
                    << "nvv4l2decoder ! "
                    << "nvvidconv ! "
                    << "video/x-raw,format=BGRx ! "
                    << "videoconvert ! "
                    << "video/x-raw,format=BGR ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    return createPipeline(pipeline_ss.str());
}

bool GstDecoder::openRtmp(const std::string& rtmp_url) {
    std::ostringstream pipeline_ss;
    
    if (output_nv12_) {
        // RTMP -> NVDEC -> NV12 (GPU)
        pipeline_ss << "rtmpsrc location=\"" << rtmp_url << "\" ! "
                    << "flvdemux ! "
                    << "queue ! "
                    << "h264parse ! "
                    << "queue ! "
                    << "nvv4l2decoder ! "
                    << "queue ! "
                    << "nvvidconv ! "
                    << "video/x-raw(memory:NVMM),format=NV12 ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    } else {
        // RTMP -> NVDEC -> BGR (CPU)
        pipeline_ss << "rtmpsrc location=\"" << rtmp_url << "\" ! "
                    << "flvdemux ! "
                    << "queue ! "
                    << "h264parse ! "
                    << "queue ! "
                    << "nvv4l2decoder ! "
                    << "queue ! "
                    << "nvvidconv ! "
                    << "video/x-raw,format=BGRx ! "
                    << "videoconvert ! "
                    << "video/x-raw,format=BGR ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    return createPipeline(pipeline_ss.str());
}

bool GstDecoder::openCamera(int camera_id, int width, int height, int fps) {
    std::ostringstream pipeline_ss;
    
    // 使用 nvarguscamerasrc (CSI 摄像头) 或 v4l2src (USB 摄像头)
    if (output_nv12_) {
        pipeline_ss << "nvarguscamerasrc sensor-id=" << camera_id << " ! "
                    << "video/x-raw(memory:NVMM),width=" << width << ",height=" << height 
                    << ",format=NV12,framerate=" << fps << "/1 ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    } else {
        pipeline_ss << "nvarguscamerasrc sensor-id=" << camera_id << " ! "
                    << "video/x-raw(memory:NVMM),width=" << width << ",height=" << height 
                    << ",format=NV12,framerate=" << fps << "/1 ! "
                    << "nvvidconv ! "
                    << "video/x-raw,format=BGRx ! "
                    << "videoconvert ! "
                    << "video/x-raw,format=BGR ! "
                    << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    width_ = width;
    height_ = height;
    fps_ = fps;
    
    return createPipeline(pipeline_ss.str());
}

bool GstDecoder::createPipeline(const std::string& pipeline_str) {
    GError* error = nullptr;
    
    std::cout << "Creating GStreamer pipeline: " << pipeline_str << std::endl;
    
    GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        std::cerr << "GStreamer pipeline error: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    if (!pipeline) {
        std::cerr << "Failed to create GStreamer pipeline" << std::endl;
        return false;
    }
    pipeline_ = pipeline;
    
    // 获取 appsink
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
        std::cerr << "Failed to get appsink" << std::endl;
        gst_object_unref(pipeline);
        pipeline_ = nullptr;
        return false;
    }
    appsink_ = appsink;
    
    // 配置 appsink
    g_object_set(appsink, 
                 "emit-signals", TRUE,
                 "sync", FALSE,
                 "max-buffers", 2,
                 "drop", TRUE,
                 nullptr);
    
    // 启动 pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start GStreamer pipeline" << std::endl;
        release();
        return false;
    }
    
    // 等待 pipeline 就绪并获取视频信息
    GstState state;
    ret = gst_element_get_state(pipeline, &state, nullptr, 5 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Pipeline failed to reach playing state" << std::endl;
        release();
        return false;
    }
    
    // 尝试获取第一帧来确定分辨率
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), GST_SECOND);
    if (sample) {
        GstCaps* caps = gst_sample_get_caps(sample);
        if (caps) {
            GstStructure* s = gst_caps_get_structure(caps, 0);
            gst_structure_get_int(s, "width", &width_);
            gst_structure_get_int(s, "height", &height_);
            
            gint fps_n = 0, fps_d = 1;
            if (gst_structure_get_fraction(s, "framerate", &fps_n, &fps_d)) {
                fps_ = static_cast<double>(fps_n) / fps_d;
            }
        }
        
        // 处理第一帧
        processSampleInternal(sample);
        gst_sample_unref(sample);
    }
    
    is_opened_ = true;
    std::cout << "GStreamer pipeline opened: " << width_ << "x" << height_ << " @ " << fps_ << " fps" << std::endl;
    
    return true;
}

bool GstDecoder::read(GstFrame& frame) {
    if (!is_opened_ || eos_received_) {
        return false;
    }
    
    // 先检查队列
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (!frame_queue_.empty()) {
            frame = std::move(frame_queue_.front());
            frame_queue_.pop();
            return true;
        }
    }
    
    GstElement* appsink = static_cast<GstElement*>(appsink_);
    
    // 从 appsink 拉取样本
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 100 * GST_MSECOND);
    if (!sample) {
        // 检查是否到达流末尾
        if (gst_app_sink_is_eos(GST_APP_SINK(appsink))) {
            eos_received_ = true;
            return false;
        }
        return false;
    }
    
    // 处理样本
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    
    if (buffer && caps) {
        // GstMapInfo map;
        // if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        //     frame.width = width_;
        //     frame.height = height_;
        //     frame.pts = GST_BUFFER_PTS(buffer);
        //     frame.is_nv12 = output_nv12_;
            
        //     if (output_nv12_) {
        //         // NV12 格式
        //         // TODO: 实现零拷贝 GPU 内存访问
        //         // 目前暂时拷贝到 CPU 然后再上传
        //         frame.nv12_size = map.size;
        //         frame.nv12_gpu_data = nullptr;  // 需要实现 GPU 内存管理
        //     } else {
        //         // BGR 格式
        frame.width = width_;
        frame.height = height_;
        frame.pts = GST_BUFFER_PTS(buffer);
        frame.is_nv12 = output_nv12_;
        

        if (output_nv12_) {
             static bool warned = false;
            if (!warned) {
                std::cerr << "Warning: NV12 NVMM zero-copy not yet implemented, using BGR path" << std::endl;
                warned = true;
            }
            // 回退：让 GStreamer 输出 BGR
            frame.is_nv12 = false;
        }
        
        // BGR 格式
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            frame.bgr_frame = cv::Mat(height_, width_, CV_8UC3);
            memcpy(frame.bgr_frame.data, map.data, map.size);
            gst_buffer_unmap(buffer, &map);
        }
        
    }
    
    gst_sample_unref(sample);
    return true;
}

void GstDecoder::processSampleInternal(void* sample_ptr) {
    GstSample* sample = static_cast<GstSample*>(sample_ptr);
    if (!sample) return;
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) return;
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return;
    
    GstFrame frame;
    frame.width = width_;
    frame.height = height_;
    frame.pts = GST_BUFFER_PTS(buffer);
    frame.is_nv12 = output_nv12_;
    
    if (!output_nv12_) {
        frame.bgr_frame = cv::Mat(height_, width_, CV_8UC3);
        memcpy(frame.bgr_frame.data, map.data, map.size);
    }
    
    gst_buffer_unmap(buffer, &map);
    
    // 添加到队列
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() < MAX_QUEUE_SIZE) {
        frame_queue_.push(std::move(frame));
    }
}

void GstDecoder::release() {
    if (pipeline_) {
        GstElement* pipeline = static_cast<GstElement*>(pipeline_);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline_ = nullptr;
    }
    
    if (appsink_) {
        GstElement* appsink = static_cast<GstElement*>(appsink_);
        gst_object_unref(appsink);
        appsink_ = nullptr;
    }
    
    is_opened_ = false;
    eos_received_ = false;
    
    // 清空队列
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!frame_queue_.empty()) {
        frame_queue_.pop();
    }
}

bool checkGstreamerAvailable() {
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    // 检查 NVDEC 插件
    GstElementFactory* factory = gst_element_factory_find("nvv4l2decoder");
    if (!factory) {
        std::cerr << "nvv4l2decoder plugin not found" << std::endl;
        return false;
    }
    gst_object_unref(factory);
    
    // 检查 nvvidconv 插件
    factory = gst_element_factory_find("nvvidconv");
    if (!factory) {
        std::cerr << "nvvidconv plugin not found" << std::endl;
        return false;
    }
    gst_object_unref(factory);
    
    return true;
}

void printGstreamerInfo() {
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    guint major, minor, micro, nano;
    gst_version(&major, &minor, &micro, &nano);
    
    std::cout << "GStreamer version: " << major << "." << minor << "." << micro;
    if (nano == 1) std::cout << " (CVS)";
    else if (nano == 2) std::cout << " (Prerelease)";
    std::cout << std::endl;
    
    // 检查关键插件
    const char* plugins[] = {
        "nvv4l2decoder",
        "nvvidconv",
        "nvarguscamerasrc",
        "rtspsrc",
        "rtmpsrc",
        "qtdemux",
        "h264parse"
    };
    
    std::cout << "Available NVIDIA plugins:" << std::endl;
    for (const char* plugin_name : plugins) {
        GstElementFactory* factory = gst_element_factory_find(plugin_name);
        if (factory) {
            std::cout << "  [OK] " << plugin_name << std::endl;
            gst_object_unref(factory);
        } else {
            std::cout << "  [--] " << plugin_name << " (not found)" << std::endl;
        }
    }
}

} // namespace jetson

#else  // USE_GSTREAMER not defined

// 当 GStreamer 不可用时的占位实现
namespace jetson {

GstDecoder::GstDecoder(bool output_nv12) : output_nv12_(output_nv12) {
    std::cerr << "GStreamer support not compiled. Rebuild with -DUSE_GSTREAMER=ON" << std::endl;
}

GstDecoder::~GstDecoder() {}

bool GstDecoder::openFile(const std::string&) { return false; }
bool GstDecoder::openRtsp(const std::string&) { return false; }
bool GstDecoder::openCamera(int, int, int, int) { return false; }
bool GstDecoder::read(GstFrame&) { return false; }
void GstDecoder::release() {}
bool GstDecoder::createPipeline(const std::string&) { return false; }
void GstDecoder::processSampleInternal(void*) {}

bool checkGstreamerAvailable() { return false; }
void printGstreamerInfo() {
    std::cout << "GStreamer support not compiled" << std::endl;
}

} // namespace jetson

#endif  // USE_GSTREAMER
