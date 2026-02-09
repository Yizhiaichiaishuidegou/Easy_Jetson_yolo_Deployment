#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct ModelConfig {
    std::string engine_path;
    std::vector<std::string> class_names;
    float confidence_threshold=0.25f;
    float nms_threshold = 0.45f;
    int input_width=640;
    int input_height=640;
    int num_box=8400;  // 添加缺失的 num_box 成员
};

// 解码器类型枚举
enum class DecoderType {
    OPENCV,           // OpenCV/FFMPEG 解码 (CPU)
    GSTREAMER_NVDEC   // GStreamer + NVDEC 硬件解码
};

// 预处理后端枚举
enum class PreprocessBackend {
    CUDA,             // 原始 CUDA kernel 预处理
    VPI_CUDA,         // VPI + CUDA 后端
    VPI_VIC           // VPI + VIC 硬件加速 (需要 NV12 输入)
};

struct InputConfig {
    std::string type="video";  // "video", "image", "camera", "rtsp", "rtmp"
    std::string video_path;
    std::string image_path;
    std::string rtsp_url;
    std::string rtmp_url;
    int camera_id=0;
    std::string video_id;
    DecoderType decoder = DecoderType::OPENCV; // 解码器类型
};

struct OutputConfig {
    bool save_video;
    std::string output_path;
    bool save_detections;
    std::string detections_path;
    bool fps_log;
    std::string fps_log_path;
};

struct DisplayConfig {
    bool show_fps;
    std::string imshow_name;
    int window_width;
    int window_height;
    char quit_key;
    bool show_video; // 是否展示视频图像检测结果
};

struct PerformanceConfig {
    bool async_mode = true;
    int queue_size =3;
    int gpu_id =0;
    int max_fps=30;
    std::vector<int> cpu_ids;
    PreprocessBackend preprocess_backend = PreprocessBackend::CUDA;  // 预处理后端

    bool use_dla = false;
    int dla_core = 0;
};

struct DebugConfig {
    std::string log_level;
    bool save_debug_images;
    std::string debug_image_path;
};

class ConfigLoader {
private:
    ModelConfig model_config_;
    InputConfig input_config_;
    OutputConfig output_config_;
    DisplayConfig display_config_;
    PerformanceConfig performance_config_;
    DebugConfig debug_config_;

public:
    bool loadConfig(const std::string& config_path);
    
    // Getters
    const ModelConfig& getModelConfig() const { return model_config_; }
    const InputConfig& getInputConfig() const { return input_config_; }
    const OutputConfig& getOutputConfig() const { return output_config_; }
    const DisplayConfig& getDisplayConfig() const { return display_config_; }
    const PerformanceConfig& getPerformanceConfig() const { return performance_config_; }
    const DebugConfig& getDebugConfig() const { return debug_config_; }
};

#endif