# ============================================================================
# CMakeLists.txt - Jetson YOLO 高性能推理系统
# 针对 Jetson Orin NX 优化
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(YoloJetson LANGUAGES CXX CUDA)

# ============================================================================
# 编译选项
# ============================================================================

# C++ 标准 11.4
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 编译优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
endif()

# ============================================================================
# CUDA 配置
# ============================================================================

find_package(CUDA REQUIRED)

# Jetson Orin NX: Ampere 架构 (SM 8.7)
# Jetson Xavier: Volta 架构 (SM 7.2)
# 自动检测或手动设置
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 默认为 Orin NX
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# 查找依赖库
# ============================================================================

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# VPI (用于 VIC 加速预处理)
find_package(VPI REQUIRED)
message(STATUS "VPI found: ${VPI_INCLUDE_DIRS}")
message(STATUS "VPI libraries: ${VPI_LIBRARIES}")

# 如果 VPI_LIBRARIES 为空，手动查找
if(NOT VPI_LIBRARIES)
    find_library(VPI_LIBRARY NAMES nvvpi vpi
        PATHS /opt/nvidia/vpi2/lib64
              /opt/nvidia/vpi1/lib64
              /usr/lib/aarch64-linux-gnu
              /usr/lib
    )
    if(VPI_LIBRARY)
        set(VPI_LIBRARIES ${VPI_LIBRARY})
        message(STATUS "VPI library found manually: ${VPI_LIBRARIES}")
    else()
        message(FATAL_ERROR "VPI library not found!")
    endif()
endif()

# 如果 VPI_INCLUDE_DIRS 为空，手动设置
if(NOT VPI_INCLUDE_DIRS)
    set(VPI_INCLUDE_DIRS 
        /opt/nvidia/vpi2/include
        /opt/nvidia/vpi1/include
    )
    message(STATUS "VPI include dirs set manually: ${VPI_INCLUDE_DIRS}")
endif()

# GStreamer (可选，用于 NVDEC 硬件解码)
option(USE_GSTREAMER "Enable GStreamer + NVDEC support" ON)

if(USE_GSTREAMER)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GST gstreamer-1.0)
        pkg_check_modules(GST_APP gstreamer-app-1.0)
        pkg_check_modules(GST_VIDEO gstreamer-video-1.0)
        if(GST_FOUND AND GST_APP_FOUND)
            message(STATUS "GStreamer found: ${GST_VERSION}")
            # 合并所有 GStreamer 相关的 include 和库
            set(GSTREAMER_INCLUDE_DIRS 
                ${GST_INCLUDE_DIRS}
                ${GST_APP_INCLUDE_DIRS}
                ${GST_VIDEO_INCLUDE_DIRS}
            )
            set(GSTREAMER_LIBRARIES 
                ${GST_LIBRARIES}
                ${GST_APP_LIBRARIES}
                ${GST_VIDEO_LIBRARIES}
            )
                        # 查找 Jetson Multimedia API (用于 NVMM 缓冲区访问)
            find_path(NVBUF_INCLUDE_DIR nvbufsurface.h
                PATHS /usr/src/jetson_multimedia_api/include
                      /usr/include/aarch64-linux-gnu
            )
            find_library(NVBUF_LIBRARY nvbufsurface
                PATHS /usr/lib/aarch64-linux-gnu/tegra
                      /usr/lib/aarch64-linux-gnu
            )
            
            if(NVBUF_INCLUDE_DIR AND NVBUF_LIBRARY)
                message(STATUS "Jetson Multimedia API found: ${NVBUF_LIBRARY}")
                list(APPEND GSTREAMER_INCLUDE_DIRS ${NVBUF_INCLUDE_DIR})
                list(APPEND GSTREAMER_LIBRARIES ${NVBUF_LIBRARY})
                set(NVMM_AVAILABLE TRUE)
            else()
                message(STATUS "Jetson Multimedia API not found. NV12 GPU access disabled.")
                set(NVMM_AVAILABLE FALSE)
            endif()
            message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")
            message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
            set(GSTREAMER_AVAILABLE TRUE)
        else()
            message(WARNING "GStreamer not found. NVDEC support disabled.")
            set(GSTREAMER_AVAILABLE FALSE)
        endif()
    else()
        # 手动查找 GStreamer
        find_path(GSTREAMER_INCLUDE_DIRS gst/gst.h
            PATHS /usr/include/gstreamer-1.0
        )
        find_path(GLIB_INCLUDE_DIRS glib.h
            PATHS /usr/include/glib-2.0
        )
        find_path(GLIB_CONFIG_INCLUDE_DIRS glibconfig.h
            PATHS /usr/lib/aarch64-linux-gnu/glib-2.0/include
                  /usr/lib/x86_64-linux-gnu/glib-2.0/include
        )
        find_library(GSTREAMER_LIBRARIES gstreamer-1.0)
        find_library(GSTREAMER_APP_LIBRARIES gstapp-1.0)
        find_library(GSTREAMER_VIDEO_LIBRARIES gstvideo-1.0)
        find_library(GLIB_LIBRARIES glib-2.0)
        find_library(GOBJECT_LIBRARIES gobject-2.0)
        
        if(GSTREAMER_LIBRARIES AND GSTREAMER_APP_LIBRARIES)
            set(GSTREAMER_INCLUDE_DIRS 
                ${GSTREAMER_INCLUDE_DIRS}
                ${GLIB_INCLUDE_DIRS}
                ${GLIB_CONFIG_INCLUDE_DIRS}
            )
            set(GSTREAMER_LIBRARIES 
                ${GSTREAMER_LIBRARIES}
                ${GSTREAMER_APP_LIBRARIES}
                ${GSTREAMER_VIDEO_LIBRARIES}
                ${GLIB_LIBRARIES}
                ${GOBJECT_LIBRARIES}
            )
            
            set(GSTREAMER_AVAILABLE TRUE)
            message(STATUS "GStreamer found (manual): ${GSTREAMER_LIBRARIES}")
        else()
            message(WARNING "GStreamer not found. NVDEC support disabled.")
            set(GSTREAMER_AVAILABLE FALSE)
        endif()
    endif()
else()
    set(GSTREAMER_AVAILABLE FALSE)
    message(STATUS "GStreamer support disabled by user")
endif()
# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu
          /usr/include/aarch64-linux-gnu
          /usr/local/tensorrt/include
          ${CMAKE_SOURCE_DIR}/third_party/tensorrt/include
)

find_library(TENSORRT_LIB_NVINFER nvinfer
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
          /usr/local/tensorrt/lib
          ${CMAKE_SOURCE_DIR}/third_party/tensorrt/lib
)

find_library(TENSORRT_LIB_NVONNXPARSER nvonnxparser
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
          /usr/local/tensorrt/lib
          ${CMAKE_SOURCE_DIR}/third_party/tensorrt/lib
)

find_library(TENSORRT_LIB_NVINFER_PLUGIN nvinfer_plugin
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
          /usr/local/tensorrt/lib
          ${CMAKE_SOURCE_DIR}/third_party/tensorrt/lib
)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libraries: ${TENSORRT_LIB_NVINFER}")

# ============================================================================
# 包含目录
# ============================================================================

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${VPI_INCLUDE_DIRS}
)
# 添加 GStreamer 头文件目录
if(GSTREAMER_AVAILABLE)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
endif()

link_directories(
    ${CUDA_LIBRARY_DIRS}
)

# ============================================================================
# 源文件定义
# ============================================================================

set(SOURCES
    src/yolo_engine.cu
    src/pipeline.cpp
    src/config_loader.cpp
    src/gst_decoder.cpp
)

# ============================================================================
# 构建目标
# ============================================================================

# YOLO 高性能推理系统
add_executable(YoloJetson
    main.cpp
    ${SOURCES}
)

set_target_properties(YoloJetson PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    # CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(YoloJetson PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# 添加 GStreamer 编译定义
if(GSTREAMER_AVAILABLE)
    target_compile_definitions(YoloJetson PRIVATE USE_GSTREAMER)
    if(NVMM_AVAILABLE)
        target_compile_definitions(YoloJetson PRIVATE USE_NVMM)
    endif()
endif()

target_link_libraries(YoloJetson PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_LIB_NVINFER}
    ${TENSORRT_LIB_NVONNXPARSER}
    ${TENSORRT_LIB_NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${VPI_LIBRARIES}
    cudart
    yaml-cpp
    pthread
)

# 链接 GStreamer 库
if(GSTREAMER_AVAILABLE)
    target_link_libraries(YoloJetson PRIVATE ${GSTREAMER_LIBRARIES})
endif()

# ============================================================================
# 安装
# ============================================================================

install(TARGETS YoloJetson DESTINATION bin)
install(DIRECTORY config/ DESTINATION config)
install(DIRECTORY model/ DESTINATION model)

# ============================================================================
# 打印配置信息
# ============================================================================

message(STATUS "==========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV found: ${OpenCV_FOUND}")
message(STATUS "VPI found: TRUE")
if(GSTREAMER_AVAILABLE)
    message(STATUS "GStreamer: ENABLED (NVDEC hardware decoding)")
else()
    message(STATUS "GStreamer: DISABLED")
endif()
message(STATUS "==========================================")
message(STATUS "Build Targets:")
message(STATUS "  YoloJetson - High-performance pipeline")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./YoloJetson -c ../config/default.yaml")
message(STATUS "  ./YoloJetson -s  # Use simple synchronous pipeline")
message(STATUS "  ./YoloJetson -p  # Enable profiling")
message(STATUS "")
message(STATUS "Hardware Acceleration Options (in config):")
message(STATUS "  decoder: opencv / gstreamer_nvdec")
message(STATUS "  preprocess_backend: cuda / vpi_cuda / vpi_vic")
message(STATUS "==========================================")
