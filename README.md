# Easy Jetson YOLO Deployment

## Overview
This project is a high-performance YOLO deployment system for the **Jetson Orin NX(16GB)** platform.

## Core Components

### 1. CUDA Utils 

- **CudaStream**: CUDA stream wrapper, supporting RAII mode
- **CudaEvent**: CUDA event wrapper, used for stream synchronization
- **StreamPool**: CUDA stream pool, managing multiple parallel streams
- **ZeroCopyBuffer**: Zero copy memory buffer, utilizing Jetson UMA 
- **DeviceBuffer**: device memory buffer for device-side memory
- **PinnedBuffer**: pinned memory buffer for host-side memory

### 2. Inference Buffer 

- **InferenceBuffer**: Single inference buffer group, containing complete input/output data for a single frame

- **BufferPool**: Buffer pool, managing multiple inference buffers

### 3. YOLO Engine 

- **PreprocessKernel**: High-performance preprocessing kernel function (affine transform + normalization + channel conversion)
- **InferenceKernel**: TensorRT inference kernel function for YOLOv8 model

### 4. Pipeline 

- **Pipeline**: High-performance CUDA pipeline, supporting multiple threads, buffers, and CUDA streams
- **SimplePipeline**: Simple synchronous pipeline for debugging or low-latency scenarios

## Key Optimization Techniques

### 1. Triple Buffer Pipeline

```
Frame N:   [Preprocess] -> [Inference] -> [Postprocess] -> Output
Frame N+1:              [Preprocess] -> [Inference] -> [Postprocess]
Frame N+2:                           [Preprocess] -> [Inference]
```

Three stages in parallel on GPU, maximizing GPU utilization.

### 2. Multiple CUDA Streams

**Each buffer uses a separate CUDA stream, running operations on the GPU, allowing parallel execution.**

### 3. Zero Copy Memory

**Utilizes unified memory architecture of Jetson UMA, CPU and GPU share the same physical memory:**
- Avoids traditional Host->Device data copy
- Reduces memory bandwidth consumption
- Reduces processing latency

### 4. Async Data Transfer

**Uses CUDA asynchronous memory operations, data transfer overlaps with GPU computation.**

## Quick Start

## System Requirements

- **Jetson Orin NX (16GB)**
- **TensorRT = 8.5.2**
- **CUDA = 11.4**
- **OpenCV = 4.6.0**
- **Gstreamer = 1.2**
## pt->onnx->tran_onnx
```bash
cd tools
python onnx_trans.py yolov8n.onnx
```

### Compilation

```bash
cd Yolo_jetson_m && mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Running Demo

```bash
# Recommended: Use high-performance pipeline for best performance
./YoloJetson -c ../config/default.yaml

# Use simple synchronous pipeline (low latency)
./YoloJetson -s -c ../config/default.yaml

# Enable performance analysis
./YoloJetson -p -c ../config/default.yaml
```

## Performance Optimization

### Jetson System Settings

```bash
# Set highest performance mode
sudo nvpmodel -m 0

# Lock clock frequency
sudo jetson_clocks

# View current status
sudo tegrastats
```

## Performance 

Jetson Orin NX (15W mode):

| Model | Resolution | Throughput | Latency |
|------|--------|--------|------|
| YOLOv8n-FP16 | 640x640 | ~80 FPS | ~25ms |
| YOLOv8s-FP16 | 640x640 | ~50 FPS | ~40ms |
| YOLOv8m-FP16 | 640x640 | ~30 FPS | ~65ms |

*注：Actual performance depends on input video resolution, and number of detected objects.*
