/**
 * @file yolo_engine.cu
 * @brief 高性能 YOLO 推理引擎实现
 * @description CUDA 核函数和引擎逻辑
 */

#include "yolo_engine.h"
#include <fstream>
#include <chrono>


namespace jetson {

// ============================================================================
// CUDA 核函数
// ============================================================================

#define GPU_BLOCK_THREADS 256
#define NUM_BOX_ELEMENT 8

// 预处理核函数: 仿射变换 + 归一化 + 通道转换
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    float m0, float m1, float m2,  // 仿射矩阵第一行
    float m3, float m4, float m5,  // 仿射矩阵第二行
    float scale, uint8_t pad_value, bool swap_rb
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_width || dy >= dst_height) return;
    
    // 计算源图像坐标
    float src_x = m0 * dx + m1 * dy + m2;
    float src_y = m3 * dx + m4 * dy + m5;
    
    float c0, c1, c2;
    
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        // 超出边界，使用填充值
        c0 = c1 = c2 = pad_value * scale;
    } else {
        // 双线性插值
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;
        
        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;
        
        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;
        
        auto get_pixel = [&](int x, int y, int c) -> float {
            if (x < 0 || x >= src_width || y < 0 || y >= src_height) {
                return pad_value;
            }
            return src[y * src_width * 3 + x * 3 + c];
        };
        
        #define GET_PIXEL(x, y, ch) \
            (((x) < 0 || (x) >= src_width || (y) < 0 || (y) >= src_height) ? \
             (float)pad_value : (float)src[(y) * src_width * 3 + (x) * 3 + (ch)])
        
        c0 = w1 * GET_PIXEL(x_low, y_low, 0) + w2 * GET_PIXEL(x_high, y_low, 0) +
             w3 * GET_PIXEL(x_low, y_high, 0) + w4 * GET_PIXEL(x_high, y_high, 0);
        c1 = w1 * GET_PIXEL(x_low, y_low, 1) + w2 * GET_PIXEL(x_high, y_low, 1) +
             w3 * GET_PIXEL(x_low, y_high, 1) + w4 * GET_PIXEL(x_high, y_high, 1);
        c2 = w1 * GET_PIXEL(x_low, y_low, 2) + w2 * GET_PIXEL(x_high, y_low, 2) +
             w3 * GET_PIXEL(x_low, y_high, 2) + w4 * GET_PIXEL(x_high, y_high, 2);
        
        #undef GET_PIXEL
        
        // 归一化
        c0 *= scale;
        c1 *= scale;
        c2 *= scale;
    }
    
    // BGR -> RGB (如果需要)
    if (swap_rb) {
        float t = c0;
        c0 = c2;
        c2 = t;
    }
    
    // 写入 NCHW 格式
    int area = dst_width * dst_height;
    dst[0 * area + dy * dst_width + dx] = c0;
    dst[1 * area + dy * dst_width + dx] = c1;
    dst[2 * area + dy * dst_width + dx] = c2;
}

// 专门用于 VPI 输出后的归一化和 NCHW 转换 Kernel
__global__ void vpi_to_nchw_kernel(
    const float3* __restrict__ src,
    float* __restrict__ dst,
    int width, int height,
    float scale
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= width || dy >= height) return;
    
    float3 pixel = src[dy * width + dx];
    
    int area = width * height;
    dst[0 * area + dy * width + dx] = pixel.x * scale;
    dst[1 * area + dy * width + dx] = pixel.y * scale;
    dst[2 * area + dy * width + dx] = pixel.z * scale;
}
// NV12 Letterbox + YUV->RGB + 归一化 + NCHW 核函数
// 输入: NV12 格式 (Y 平面 + UV 平面)
// 输出: NCHW 格式 RGB float 张量
__global__ void nv12_letterbox_normalize_kernel(
    const uint8_t* __restrict__ y_plane,    // Y 平面数据
    const uint8_t* __restrict__ uv_plane,   // UV 交错平面数据
    float* __restrict__ dst,                 // 输出 NCHW 张量
    int src_width, int src_height,          // 源图像尺寸
    int y_pitch, int uv_pitch,              // 平面行步长
    int dst_width, int dst_height,          // 目标尺寸 (640x640)
    int offset_x, int offset_y,             // letterbox 偏移
    int scaled_w, int scaled_h,             // 缩放后的尺寸
    float scale_x, float scale_y,           // 缩放比例
    float normalize_scale,                  // 归一化系数 (1/255)
    uint8_t pad_value                       // 填充值
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_width || dy >= dst_height) return;
    
    float r, g, b;
    
    // 计算是否在有效区域内
    int rx = dx - offset_x;
    int ry = dy - offset_y;
    
    if (rx < 0 || rx >= scaled_w || ry < 0 || ry >= scaled_h) {
        // 填充区域
        r = g = b = pad_value * normalize_scale;
    } else {
        // 计算源图像坐标
        float src_x = rx * scale_x;
        float src_y = ry * scale_y;
        
        int ix = __float2int_rd(src_x);
        int iy = __float2int_rd(src_y);
        
        ix = min(max(ix, 0), src_width - 1);
        iy = min(max(iy, 0), src_height - 1);
        
        // 读取 Y 值
        uint8_t Y = y_plane[iy * y_pitch + ix];
        
        // 读取 UV 值 (UV 平面是 Y 分辨率的一半)
        int uvx = ix / 2;
        int uvy = iy / 2;
        int uv_idx = uvy * uv_pitch + uvx * 2;
        uint8_t U = uv_plane[uv_idx];
        uint8_t V = uv_plane[uv_idx + 1];
        
        // YUV to RGB 转换 (BT.601)
        float Yf = (float)Y;
        float Uf = (float)U - 128.0f;
        float Vf = (float)V - 128.0f;
        
        r = Yf + 1.402f * Vf;
        g = Yf - 0.344f * Uf - 0.714f * Vf;
        b = Yf + 1.772f * Uf;
        
        // 钳制并归一化
        r = fminf(fmaxf(r, 0.0f), 255.0f) * normalize_scale;
        g = fminf(fmaxf(g, 0.0f), 255.0f) * normalize_scale;
        b = fminf(fmaxf(b, 0.0f), 255.0f) * normalize_scale;
    }
    
    // 写入 NCHW 格式 (RGB 顺序)
    int area = dst_width * dst_height;
    dst[0 * area + dy * dst_width + dx] = r;
    dst[1 * area + dy * dst_width + dx] = g;
    dst[2 * area + dy * dst_width + dx] = b;
}

// VIC 加速后的 Letterbox + 归一化 + NCHW 转换核函数
// 输入: VIC Rescale 后的 BGR8 图像 (已缩放但未 padding)
// 输出: NCHW 格式的 float 张量 (带 letterbox padding)
__global__ void letterbox_normalize_kernel(
    const uint8_t* __restrict__ src,    // VIC 缩放后的 RGB 图像
    float* __restrict__ dst,             // 输出 NCHW 张量
    int scaled_w, int scaled_h,          // 缩放后的尺寸
    int src_pitch,
    int dst_w, int dst_h,                // 目标尺寸 (640x640)
    int offset_x, int offset_y,          // letterbox 偏移
    float scale, uint8_t pad_value,
    bool swap_rb
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_w || dy >= dst_h) return;
    
    float r, g, b;
    
    // 计算在缩放图像中的坐标
    int src_x = dx - offset_x;
    int src_y = dy - offset_y;
    
    if (src_x < 0 || src_x >= scaled_w || src_y < 0 || src_y >= scaled_h) {
        // padding 区域
        r = g = b = pad_value * scale;
    } else {
        // 从 VIC 输出读取 (已经是 RGB 格式)
        int idx = src_y * src_pitch + src_x * 3;
        if (swap_rb) {
            // BGR -> RGB
            b = src[idx + 0] * scale;
            g = src[idx + 1] * scale;
            r = src[idx + 2] * scale;
        } else {
            r = src[idx + 0] * scale;
            g = src[idx + 1] * scale;
            b = src[idx + 2] * scale;
        }
    }
    
    // 写入 NCHW 格式
    int area = dst_w * dst_h;
    dst[0 * area + dy * dst_w + dx] = r;
    dst[1 * area + dy * dst_w + dx] = g;
    dst[2 * area + dy * dst_w + dx] = b;
}
// 解码核函数: YOLOv8 输出格式
__global__ void decode_kernel(
    const float* __restrict__ predict,
    float* __restrict__ output,
    int num_boxes, int num_classes,
    float conf_threshold,
    float d2i_m0, float d2i_m1, float d2i_m2,
    float d2i_m3, float d2i_m4, float d2i_m5,
    int max_detections
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // YOLOv8 输出格式: [x, y, w, h, class0_conf, class1_conf, ...]
    const float* pitem = predict + idx * (4 + num_classes);
    
    // 找到最大类别置信度
    float max_conf = 0;
    int max_class = 0;
    for (int c = 0; c < num_classes; ++c) {
        float conf = pitem[4 + c];
        if (conf > max_conf) {
            max_conf = conf;
            max_class = c;
        }
    }
    
    if (max_conf < conf_threshold) return;
    
    // 原子操作获取输出索引
    int out_idx = atomicAdd(reinterpret_cast<int*>(output), 1);
    if (out_idx >= max_detections) return;
    
    // 解析边界框
    float cx = pitem[0];
    float cy = pitem[1];
    float w = pitem[2];
    float h = pitem[3];
    
    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;
    
    // 仿射变换到原图坐标
    float ox1 = d2i_m0 * x1 + d2i_m1 * y1 + d2i_m2;
    float oy1 = d2i_m3 * x1 + d2i_m4 * y1 + d2i_m5;
    float ox2 = d2i_m0 * x2 + d2i_m1 * y2 + d2i_m2;
    float oy2 = d2i_m3 * x2 + d2i_m4 * y2 + d2i_m5;
    
    // 写入结果: [x1, y1, x2, y2, conf, class, keep_flag, position]
    float* pout = output + 1 + out_idx * NUM_BOX_ELEMENT;
    pout[0] = ox1;
    pout[1] = oy1;
    pout[2] = ox2;
    pout[3] = oy2;
    pout[4] = max_conf;
    pout[5] = static_cast<float>(max_class);
    pout[6] = 1.0f;  // keep flag
    pout[7] = static_cast<float>(idx);  // position
}

// NMS 核函数
__global__ void nms_kernel(
    float* __restrict__ boxes,
    int max_detections,
    float nms_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = min(*(reinterpret_cast<int*>(boxes)), max_detections);
    if (idx >= count) return;
    
    float* pcur = boxes + 1 + idx * NUM_BOX_ELEMENT;
    if (pcur[6] == 0) return;  // 已被抑制
    
    float x1 = pcur[0], y1 = pcur[1], x2 = pcur[2], y2 = pcur[3];
    float conf = pcur[4];
    int cls = static_cast<int>(pcur[5]);
    
    for (int i = 0; i < count; ++i) {
        if (i == idx) continue;
        
        float* pitem = boxes + 1 + i * NUM_BOX_ELEMENT;
        if (pitem[6] == 0) continue;  // 已被抑制
        if (static_cast<int>(pitem[5]) != cls) continue;  // 不同类别
        
        // 只抑制置信度较低的
        if (pitem[4] > conf || (pitem[4] == conf && i < idx)) continue;
        
        // 计算IoU
        float ix1 = fmaxf(x1, pitem[0]);
        float iy1 = fmaxf(y1, pitem[1]);
        float ix2 = fminf(x2, pitem[2]);
        float iy2 = fminf(y2, pitem[3]);
        
        float iw = fmaxf(0.0f, ix2 - ix1);
        float ih = fmaxf(0.0f, iy2 - iy1);
        float inter = iw * ih;
        
        float area1 = (x2 - x1) * (y2 - y1);
        float area2 = (pitem[2] - pitem[0]) * (pitem[3] - pitem[1]);
        float iou = inter / (area1 + area2 - inter + 1e-6f);
        
        if (iou > nms_threshold) {
            pcur[6] = 0;  // 抑制当前框
            return;
        }
    }
}

// ============================================================================
// YoloEngine 实现
// ============================================================================

YoloEngine::YoloEngine(const ModelConfig& config, PreprocessBackend backend,
                        bool use_dla, int dla_core) 
    : config_(config), preprocess_backend_(backend), use_dla_(use_dla), dla_core_(dla_core) {
    std::cout << "  YoloEngine: Reading config values..." << std::endl << std::flush;   
    input_h_ = config.input_height;
    input_w_ = config.input_width;
    std::cout << "  YoloEngine: input size = " << input_w_ << "x" << input_h_ << std::endl << std::flush;
    num_classes_ = static_cast<int>(config.class_names.size());
    std::cout << "  YoloEngine: num_classes = " << num_classes_ << std::endl << std::flush;
    num_boxes_ = config.num_box;
    std::cout << "  YoloEngine: num_boxes = " << num_boxes_ << std::endl << std::flush;
    output_size_ = num_boxes_ * (num_classes_ + 4);
    
    postprocess_params_.conf_threshold = config.confidence_threshold;
    postprocess_params_.nms_threshold = config.nms_threshold;
    
    preprocess_params_.swap_rb = true;
    preprocess_params_.scale = 1.0f / 255.0f;

    const char* backend_names[] = {"CUDA", "VPI_CUDA", "VPI_VIC"};
    
    std::cout << "  Preprocess backend: " << backend_names[static_cast<int>(backend)] << std::endl << std::flush;
        int backend_idx = static_cast<int>(backend);
    if (backend_idx >= 0 && backend_idx < 3) {
        std::cout << "  Preprocess backend: " << backend_names[backend_idx] << std::endl << std::flush;
    } else {
        std::cerr << "  WARNING: Invalid preprocess backend value: " << backend_idx 
                  << ", using CUDA" << std::endl << std::flush;
        preprocess_backend_ = PreprocessBackend::CUDA;
    }
        // 打印 DLA 配置
    if (use_dla_) {
        std::cout << "  DLA: Enabled (core " << dla_core_ << ")" << std::endl << std::flush;
    }
}

YoloEngine::~YoloEngine() {
    // 重要：按正确顺序显式销毁 TensorRT 资源
    // context 必须在 engine 之前销毁，engine 必须在 runtime 之前销毁
    context_.reset();
    engine_.reset();
    runtime_.reset();
    
    // 销毁 VPI 资源（如果存在）
    if (vpi_rescaled_) {
        vpiImageDestroy(vpi_rescaled_);
        vpi_rescaled_ = nullptr;
    }
    if (vpi_input_rgb_) {
        vpiImageDestroy(vpi_input_rgb_);
        vpi_input_rgb_ = nullptr;
    }
    if (vpi_input_bgr_) {
        vpiImageDestroy(vpi_input_bgr_);
        vpi_input_bgr_ = nullptr;
    }
    if (vpi_stream_) {
        vpiStreamDestroy(vpi_stream_);
        vpi_stream_ = nullptr;
    }
    // TensorRT 资源会通过 unique_ptr 自动释放
}

bool YoloEngine::initialize() {
    if (initialized_) return true;
    std::cout << "  YoloEngine::initialize() starting..." << std::endl << std::flush;
    // 初始化 TensorRT 插件
    std::cout << "  Initializing TensorRT plugins..." << std::endl << std::flush;
    initLibNvInferPlugins(&logger_, "");
    std::cout << "  TensorRT plugins initialized" << std::endl << std::flush;

    // 只在需要 VPI 后端时初始化 VPI 流
    if (preprocess_backend_ == PreprocessBackend::VPI_CUDA || 
        preprocess_backend_ == PreprocessBackend::VPI_VIC) {
        std::cout << "  Creating VPI stream for VPI backend..." << std::endl << std::flush;
        VPIStatus status = vpiStreamCreate(0, &vpi_stream_);
        if (status != VPI_SUCCESS) {
            std::cerr << "Failed to create VPI stream: " << vpiStatusGetName(status) << std::endl;
            return false;
        }
        std::cout << "  VPI stream created for VIC acceleration" << std::endl << std::flush;
    } else {
        std::cout << "  Using CUDA backend, skipping VPI initialization" << std::endl << std::flush;
    }
    
    // 加载引擎
    if (!loadEngine(config_.engine_path)) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return false;
    }
    

    initialized_ = true;
    std::cout << "YoloEngine initialized successfully" << std::endl;
    std::cout << "  Input: " << input_w_ << "x" << input_h_ << std::endl;
    std::cout << "  Classes: " << num_classes_ << std::endl;
    std::cout << "  Boxes: " << num_boxes_ << std::endl;
    std::cout << std::flush;
    return true;
}

bool YoloEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    // 创建 runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    // 配置 DLA (如果启用)
    if (use_dla_) {
        int nbDLACores = runtime_->getNbDLACores();
        std::cout << "Available DLA cores: " << nbDLACores << std::endl;
        
        if (nbDLACores > 0) {
            if (dla_core_ >= nbDLACores) {
                std::cerr << "Warning: DLA core " << dla_core_ << " not available, using core 0" << std::endl;
                dla_core_ = 0;
            }
            runtime_->setDLACore(dla_core_);
            std::cout << "Using DLA core: " << dla_core_ << std::endl;
        } else {
            std::cerr << "Warning: No DLA cores available, falling back to GPU" << std::endl;
            use_dla_ = false;
        }
    }
    // 反序列化引擎
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }
    // 检查引擎是否支持 DLA
    if (use_dla_) {
        std::cout << "Note: DLA acceleration requested. Engine must be built with --useDLACore flag." << std::endl;
        std::cout << "  If engine doesn't support DLA, inference will run on GPU." << std::endl;
    }
    
    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // 获取绑定索引
    input_index_ = engine_->getBindingIndex("images");
    output_index_ = engine_->getBindingIndex("output0");
    
    if (input_index_ < 0 || output_index_ < 0) {
        std::cerr << "Invalid binding names" << std::endl;
        return false;
    }
    
    std::cout << "Engine loaded: " << engine_path << std::endl;
    if (use_dla_) {
        std::cout << "  DLA acceleration: ENABLED (core " << dla_core_ << ")" << std::endl;
    }
    return true;
}

void YoloEngine::computeAffineMatrix(int src_w, int src_h, float* i2d, float* d2i) {
    float scale_x = static_cast<float>(input_w_) / src_w;
    float scale_y = static_cast<float>(input_h_) / src_h;
    float scale = std::min(scale_x, scale_y);
    
    // image to detection (i2d)
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * src_w * 0.5f + input_w_ * 0.5f;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * src_h * 0.5f + input_h_ * 0.5f;
    
    // detection to image (d2i) - 逆矩阵
    float D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0 ? 1.0f / D : 0;
    
    float A11 = i2d[4] * D;
    float A22 = i2d[0] * D;
    float A12 = -i2d[1] * D;
    float A21 = -i2d[3] * D;
    
    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = -A11 * i2d[2] - A12 * i2d[5];
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = -A21 * i2d[2] - A22 * i2d[5];
}

void YoloEngine::preprocess(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream) {
    if (frame.empty() || frame.channels() != 3) {
        std::cerr << "Invalid input frame" << std::endl;
        return;
    }
    
    buffer.src_width = frame.cols;
    buffer.src_height = frame.rows;
    buffer.src_frame = frame.clone();
    
    // --- 优化部分：检查分辨率是否变化 ---
    if (frame.cols != last_src_w_ || frame.rows != last_src_h_) {
        // 只有在分辨率改变时才计算
        computeAffineMatrix(frame.cols, frame.rows, cached_i2d_, cached_d2i_);
        last_src_w_ = frame.cols;
        last_src_h_ = frame.rows;
        std::cout << "Resolution changed or initialized: " << last_src_w_ << "x" << last_src_h_ 
                  << ", recalculated affine matrix." << std::endl;
    }
    
    // 将缓存的矩阵拷贝到当前 buffer
    std::memcpy(buffer.affine_i2d, cached_i2d_, sizeof(float) * 6);
    std::memcpy(buffer.affine_d2i, cached_d2i_, sizeof(float) * 6);
    
     // 根据后端配置选择不同的预处理实现
    switch (preprocess_backend_) {
        case PreprocessBackend::VPI_CUDA:
            preprocessVpiCuda(buffer, frame, stream);
            break;
        case PreprocessBackend::VPI_VIC:
            preprocessVpiVic(buffer, frame, stream);
            break;
        case PreprocessBackend::CUDA:
        default:
            preprocessCuda(buffer, frame, stream);
            break;
    }
}

// ============================================================================
// 原始 CUDA Kernel 预处理 (默认方案)
// ============================================================================

void YoloEngine::preprocessCuda(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream) {
    // 分配源图像设备内存
    size_t src_size = frame.cols * frame.rows * 3;
    if (buffer.src_device->bytes() < src_size) {
        buffer.src_device->allocate(src_size);
    }
    
    // 拷贝图像到设备
    CUDA_CHECK(cudaMemcpyAsync(
        buffer.src_device->get(),
        frame.data,
        src_size,
        cudaMemcpyHostToDevice,
        stream
    ));
    
    // 启动预处理核函数
    dim3 block(32, 32);
    dim3 grid((input_w_ + 31) / 32, (input_h_ + 31) / 32);
    
    preprocess_kernel<<<grid, block, 0, stream>>>(
        buffer.src_device->get(),
        buffer.input_tensor->get(),
        frame.cols, frame.rows,
        input_w_, input_h_,
        buffer.affine_d2i[0], buffer.affine_d2i[1], buffer.affine_d2i[2],
        buffer.affine_d2i[3], buffer.affine_d2i[4], buffer.affine_d2i[5],
        preprocess_params_.scale,
        preprocess_params_.pad_value,
        preprocess_params_.swap_rb
    );
    
    // 记录预处理完成事件
    buffer.preprocess_done->record(stream);
}

// ============================================================================
// VPI + CUDA 后端预处理
// ============================================================================

void YoloEngine::preprocessVpiCuda(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream) {
    // 计算缩放比例和 letterbox 参数
    float scale_x = static_cast<float>(input_w_) / frame.cols;
    float scale_y = static_cast<float>(input_h_) / frame.rows;
    float scale = std::min(scale_x, scale_y);
    
    int scaled_w = static_cast<int>(frame.cols * scale);
    int scaled_h = static_cast<int>(frame.rows * scale);
    int offset_x = (input_w_ - scaled_w) / 2;
    int offset_y = (input_h_ - scaled_h) / 2;
    
    // 分辨率变化时重新创建 VPI 图像
    if (frame.cols != last_vpi_w_ || frame.rows != last_vpi_h_) {
        // 释放旧资源
        if (vpi_input_bgr_) { vpiImageDestroy(vpi_input_bgr_); vpi_input_bgr_ = nullptr; }
        if (vpi_input_rgb_) { vpiImageDestroy(vpi_input_rgb_); vpi_input_rgb_ = nullptr; }
        if (vpi_rescaled_) { vpiImageDestroy(vpi_rescaled_); vpi_rescaled_ = nullptr; }
        
        // 创建新的 VPI 图像 (使用 CUDA 后端)
        VPIStatus status;
        
        status = vpiImageCreate(frame.cols, frame.rows, VPI_IMAGE_FORMAT_BGR8, 
                               VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &vpi_input_bgr_);
        if (status != VPI_SUCCESS) {
            std::cerr << "Failed to create vpi_input_bgr_: " << vpiStatusGetName(status) << std::endl;
            return;
        }
        
        status = vpiImageCreate(scaled_w, scaled_h, VPI_IMAGE_FORMAT_BGR8, 
                               VPI_BACKEND_CUDA, &vpi_rescaled_);
        if (status != VPI_SUCCESS) {
            std::cerr << "Failed to create vpi_rescaled_: " << vpiStatusGetName(status) << std::endl;
            return;
        }
        
        vpi_input_rgb_ = nullptr;
        last_vpi_w_ = frame.cols;
        last_vpi_h_ = frame.rows;
        
        std::cout << "VPI images created (CUDA backend): " 
                  << frame.cols << "x" << frame.rows << " -> " 
                  << scaled_w << "x" << scaled_h << std::endl;
    }
    
    // 将 OpenCV Mat 数据拷贝到 VPI 图像
    VPIImageData imgData;
    VPIStatus lockStatus = vpiImageLockData(vpi_input_bgr_, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData);
    if (lockStatus != VPI_SUCCESS) {
        std::cerr << "Failed to lock vpi_input_bgr_: " << vpiStatusGetName(lockStatus) << std::endl;
        return;
    }
    
    cv::Mat vpiMat(frame.rows, frame.cols, CV_8UC3, imgData.buffer.pitch.planes[0].data, 
                   imgData.buffer.pitch.planes[0].pitchBytes);
    frame.copyTo(vpiMat);
    vpiImageUnlock(vpi_input_bgr_);
    
    // VPI/CUDA: Rescale 缩放
    VPIStatus rescaleStatus = vpiSubmitRescale(vpi_stream_, VPI_BACKEND_CUDA, 
                                                vpi_input_bgr_, vpi_rescaled_, 
                                                VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0);
    if (rescaleStatus != VPI_SUCCESS) {
        std::cerr << "Failed to submit Rescale: " << vpiStatusGetName(rescaleStatus) << std::endl;
        return;
    }
    
    // 等待 VPI 完成
    VPIStatus syncStatus = vpiStreamSync(vpi_stream_);
    if (syncStatus != VPI_SUCCESS) {
        std::cerr << "Failed to sync VPI stream: " << vpiStatusGetName(syncStatus) << std::endl;
        return;
    }
    
    // 获取缩放后的数据
    VPIImageData scaledData;
    VPIStatus scaledLockStatus = vpiImageLockData(vpi_rescaled_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &scaledData);
    if (scaledLockStatus != VPI_SUCCESS) {
        std::cerr << "Failed to lock vpi_rescaled_: " << vpiStatusGetName(scaledLockStatus) << std::endl;
        return;
    }
    
    int src_pitch = static_cast<int>(scaledData.buffer.pitch.planes[0].pitchBytes);
    
    dim3 block(16, 16);
    dim3 grid((input_w_ + 15) / 16, (input_h_ + 15) / 16);

    letterbox_normalize_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(scaledData.buffer.pitch.planes[0].data),
        buffer.input_tensor->get(),
        scaled_w, scaled_h,
        src_pitch,
        input_w_, input_h_,
        offset_x, offset_y,
        preprocess_params_.scale,
        preprocess_params_.pad_value,
        preprocess_params_.swap_rb
    );
    
    cudaStreamSynchronize(stream);
    vpiImageUnlock(vpi_rescaled_);
    buffer.preprocess_done->record(stream);
}

// ============================================================================
// VPI + VIC 后端预处理 (需要 NV12 输入，用于 GStreamer + NVDEC)
// ============================================================================

void YoloEngine::preprocessVpiVic(InferenceBuffer& buffer, const cv::Mat& frame, CudaStream& stream) {
    // TODO: 实现 VIC 后端预处理 (需要 NV12 输入)
    // 当使用 GStreamer + NVDEC 时，输入将是 NV12 格式，可以直接使用 VIC 加速
    // 目前回退到 CUDA 方案
    static bool warned = false;
    if (!warned) {
        std::cerr << "VPI_VIC backend requires NV12 input. Falling back to CUDA for this session." << std::endl;
        warned = true;
    }
    preprocessCuda(buffer, frame, stream);
}

// ============================================================================
// NV12 直接预处理 (用于 NVDEC + VIC 完整硬件加速路径)
// ============================================================================

void YoloEngine::preprocessNV12(InferenceBuffer& buffer, 
                                 const uint8_t* y_plane, const uint8_t* uv_plane,
                                 int src_w, int src_h, int y_pitch, int uv_pitch,
                                 CudaStream& stream) {
    // 检查分辨率变化
    if (src_w != last_src_w_ || src_h != last_src_h_) {
        float d2i[6];
        computeAffineMatrix(src_w, src_h, cached_i2d_, d2i);
        last_src_w_ = src_w;
        last_src_h_ = src_h;
        std::cout << "NV12 resolution: " << src_w << "x" << src_h << std::endl;
    }
    
    // 计算 letterbox 参数
    float scale = std::min(static_cast<float>(input_w_) / src_w,
                          static_cast<float>(input_h_) / src_h);
    int scaled_w = static_cast<int>(src_w * scale);
    int scaled_h = static_cast<int>(src_h * scale);
    int offset_x = (input_w_ - scaled_w) / 2;
    int offset_y = (input_h_ - scaled_h) / 2;
    
    float scale_x = static_cast<float>(src_w) / scaled_w;
    float scale_y = static_cast<float>(src_h) / scaled_h;
    
    // 调用 NV12 核函数
    dim3 block(16, 16);
    dim3 grid((input_w_ + block.x - 1) / block.x, (input_h_ + block.y - 1) / block.y);
    
    nv12_letterbox_normalize_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane,
        static_cast<float*>(buffer.input_tensor->get()),
        src_w, src_h,
        y_pitch, uv_pitch,
        input_w_, input_h_,
        offset_x, offset_y,
        scaled_w, scaled_h,
        scale_x, scale_y,
        preprocess_params_.scale,
        preprocess_params_.pad_value
    );
    
    buffer.preprocess_done->record(stream);
}

void YoloEngine::inference(InferenceBuffer& buffer, CudaStream& stream) {
    // 等待预处理完成
    stream.waitEvent(*buffer.preprocess_done);
    
    // 设置绑定
    void* bindings[2];
    bindings[input_index_] = buffer.input_tensor->get();
    bindings[output_index_] = buffer.output_tensor->get();
    
    // 设置输入维度
    context_->setBindingDimensions(input_index_, 
        nvinfer1::Dims4(1, 3, input_h_, input_w_));
    
    // 执行推理
    context_->enqueueV2(bindings, stream, nullptr);
    
    // 记录推理完成事件
    buffer.inference_done->record(stream);
}

void YoloEngine::postprocess(InferenceBuffer& buffer, CudaStream& stream) {
    // 等待推理完成
    stream.waitEvent(*buffer.inference_done);
    
    // 清空输出缓冲区
    buffer.decode_output->memset(0, stream);
    
    // 解码
    int block_size = GPU_BLOCK_THREADS;
    int grid_size = (num_boxes_ + block_size - 1) / block_size;
    
    decode_kernel<<<grid_size, block_size, 0, stream>>>(
        buffer.output_tensor->get(),
        buffer.decode_output->get(),
        num_boxes_, num_classes_,
        postprocess_params_.conf_threshold,
        buffer.affine_d2i[0], buffer.affine_d2i[1], buffer.affine_d2i[2],
        buffer.affine_d2i[3], buffer.affine_d2i[4], buffer.affine_d2i[5],
        postprocess_params_.max_detections
    );
    
    // NMS
    grid_size = (postprocess_params_.max_detections + block_size - 1) / block_size;
    nms_kernel<<<grid_size, block_size, 0, stream>>>(
        buffer.decode_output->get(),
        postprocess_params_.max_detections,
        postprocess_params_.nms_threshold
    );
    
    // 拷贝结果到零拷贝内存
    size_t result_size = 1 + postprocess_params_.max_detections * NUM_BOX_ELEMENT;
    CUDA_CHECK(cudaMemcpyAsync(
        buffer.detection_results->host(),
        buffer.decode_output->get(),
        result_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));
    
    // 记录后处理完成事件
    buffer.postprocess_done->record(stream);
}

void YoloEngine::drawResults(InferenceBuffer& buffer, const std::vector<std::string>& class_names) {
    // 确保后处理完成
    buffer.postprocess_done->synchronize();
    
    buffer.result_frame = buffer.src_frame.clone();
    
    // 使用整数指针读取保存在 float 缓冲区开头的计数值
    const float* results_ptr = buffer.detection_results->host();
    int count = std::min(*(reinterpret_cast<const int*>(results_ptr)), postprocess_params_.max_detections);
    
    buffer.detection_count = 0;
    buffer.detections.clear();
    
    for (int i = 0; i < count; ++i) {
        const float* det = results_ptr + 1 + i * NUM_BOX_ELEMENT;
        
        if (det[6] < 0.5f) continue;  // 被NMS抑制
        
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        int cls = static_cast<int>(det[5]);
        
        // 保存检测结果
        buffer.detections.push_back(x1);
        buffer.detections.push_back(y1);
        buffer.detections.push_back(x2);
        buffer.detections.push_back(y2);
        buffer.detections.push_back(conf);
        buffer.detections.push_back(static_cast<float>(cls));
        buffer.detection_count++;
        
        // 绘制边界框
        cv::Scalar color(0, 255, 0);  // 绿色
        cv::rectangle(buffer.result_frame, 
                      cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                      cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                      color, 2);
        
        // 绘制标签
        std::string label;
        if (cls >= 0 && cls < static_cast<int>(class_names.size())) {
            label = class_names[cls] + ": " + cv::format("%.2f", conf);
        } else {
            label = cv::format("class%d: %.2f", cls, conf);
        }
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::rectangle(buffer.result_frame,
                      cv::Point(static_cast<int>(x1), static_cast<int>(y1) - text_size.height - 5),
                      cv::Point(static_cast<int>(x1) + text_size.width, static_cast<int>(y1)),
                      color, -1);
        
        cv::putText(buffer.result_frame, label,
                    cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

} // namespace jetson
