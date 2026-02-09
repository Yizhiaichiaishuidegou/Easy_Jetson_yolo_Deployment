/**
 * @file gst_decoder.h
 * @brief GStreamer + NVDEC 硬件解码器
 * @description 使用 NVDEC 硬件加速解码视频，输出 NV12 格式到 GPU 内存
 */

#ifndef GST_DECODER_H
#define GST_DECODER_H

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// GStreamer 前向声明
// typedef struct _GstElement GstElement;
// typedef struct _GstSample GstSample;
// typedef struct _GstBuffer GstBuffer;
// typedef struct _GstCaps GstCaps;
// typedef struct _GMainLoop GMainLoop;

namespace jetson {

/**
 * @struct GstFrame
 * @brief 解码后的帧数据
 */
struct GstFrame {
    cv::Mat bgr_frame;          // BGR 格式帧
    
    // NV12 GPU 数据 (用于 VIC 加速)
    void* nv12_y_data = nullptr;   // Y 平面 CUDA 指针
    void* nv12_uv_data = nullptr;  // UV 平面 CUDA 指针  
    int y_pitch = 0;               // Y 平面行步长
    int uv_pitch = 0;              // UV 平面行步长
    int dmabuf_fd = -1;            // DMA-BUF 文件描述符
    
    int width = 0;
    int height = 0;
    int64_t pts = 0;               // 时间戳
    bool is_nv12 = false;          // 是否为 NV12 格式
    size_t nv12_size = 0;          // NV12 数据大小
};

/**
 * @class GstDecoder
 * @brief GStreamer + NVDEC 硬件解码器
 * 
 * 特性:
 * - 使用 NVDEC 硬件加速解码
 * - 支持 MP4、RTSP 等多种输入源
 * - 可输出 NV12 格式到 GPU 内存供 VPI/VIC 使用
 * - 也可输出 BGR 格式用于 CPU 处理
 */
class GstDecoder {
public:
    /**
     * @brief 构造函数
     * @param output_nv12 是否输出 NV12 格式 (true: NV12 GPU, false: BGR CPU)
     */
    explicit GstDecoder(bool output_nv12 = false);
    ~GstDecoder();
    
    // 禁止拷贝
    GstDecoder(const GstDecoder&) = delete;
    GstDecoder& operator=(const GstDecoder&) = delete;
    
    /**
     * @brief 打开视频文件
     */
    bool openFile(const std::string& file_path);
    
    /**
     * @brief 打开 RTSP 流
     */
    bool openRtsp(const std::string& rtsp_url);
    
    /**
     * @brief 打开 RTMP 流
     */
    bool openRtmp(const std::string& rtmp_url);
    
    /**
     * @brief 打开摄像头
     */
    bool openCamera(int camera_id, int width = 1920, int height = 1080, int fps = 30);
    
    /**
     * @brief 读取一帧
     * @return 是否成功读取
     */
    bool read(GstFrame& frame);
    
    /**
     * @brief 是否已打开
     */
    bool isOpened() const { return is_opened_.load(); }
    
    /**
     * @brief 获取视频宽度
     */
    int getWidth() const { return width_; }
    
    /**
     * @brief 获取视频高度
     */
    int getHeight() const { return height_; }
    
    /**
     * @brief 获取帧率
     */
    double getFps() const { return fps_; }
    
    /**
     * @brief 释放资源
     */
    void release();

private:
    // 创建 GStreamer pipeline
    bool createPipeline(const std::string& pipeline_str);
    
    // GStreamer 回调
    // static GstFlowReturn onNewSample(GstElement* sink, void* user_data);
    // static void onEos(GstElement* sink, void* user_data);
    
    // 处理样本
    // 处理样本 (内部实现)
    void processSampleInternal(void* sample);
    
    // // 成员变量
    // GstElement* pipeline_ = nullptr;
    // GstElement* appsink_ = nullptr;

    void* pipeline_ = nullptr;   // GstElement*
    void* appsink_ = nullptr;    // GstElement*

    std::atomic<bool> is_opened_{false};
    std::atomic<bool> eos_received_{false};
    
    bool output_nv12_;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0;
    
    // 帧队列
    std::queue<GstFrame> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static constexpr size_t MAX_QUEUE_SIZE = 5;
};

/**
 * @brief 检查 GStreamer 和 NVDEC 是否可用
 * @return true 如果可用
 */
bool checkGstreamerAvailable();

/**
 * @brief 打印 GStreamer 版本和可用插件信息
 */
void printGstreamerInfo();

} // namespace jetson

#endif // GST_DECODER_H
