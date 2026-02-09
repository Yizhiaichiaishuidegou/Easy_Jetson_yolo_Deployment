/**
 * @file cuda_utils.h
 * @brief CUDA 工具类 - 流管理器和零拷贝内存管理
 * @description 针对 Jetson Orin NX 优化的 CUDA 资源管理
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_CHECK_NOTHROW(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

namespace jetson {

/**
 * @class CudaStream
 * @brief CUDA 流封装类，支持 RAII
 */
class CudaStream {
public:
    CudaStream(unsigned int flags = cudaStreamDefault) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CudaStream() {
        if (stream_) {
            CUDA_CHECK_NOTHROW(cudaStreamDestroy(stream_));
        }
    }
    
    // 禁止拷贝
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // 支持移动
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                CUDA_CHECK_NOTHROW(cudaStreamDestroy(stream_));
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    bool query() {
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);
        return false;
    }
    
    void waitEvent(cudaEvent_t event) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
    }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @class CudaEvent
 * @brief CUDA 事件封装类，用于流同步
 */
class CudaEvent {
public:
    CudaEvent(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    
    ~CudaEvent() {
        if (event_) {
            CUDA_CHECK_NOTHROW(cudaEventDestroy(event_));
        }
    }
    
    // 禁止拷贝
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // 支持移动
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    bool query() {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);
        return false;
    }
    
    // 计算两个事件之间的时间(毫秒)
    static float elapsedTime(CudaEvent& start, CudaEvent& end) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), end.get()));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @class StreamPool
 * @brief CUDA 流池，管理多个流用于并行处理
 */
class StreamPool {
public:
    explicit StreamPool(size_t num_streams = 3) {
        streams_.reserve(num_streams);
        for (size_t i = 0; i < num_streams; ++i) {
            streams_.emplace_back(std::make_unique<CudaStream>(cudaStreamNonBlocking));
        }
        std::cout << "StreamPool created with " << num_streams << " streams" << std::endl;
    }
    
    CudaStream& get(size_t index) {
        return *streams_[index % streams_.size()];
    }
    
    size_t size() const { return streams_.size(); }
    
    void synchronizeAll() {
        for (auto& stream : streams_) {
            stream->synchronize();
        }
    }

private:
    std::vector<std::unique_ptr<CudaStream>> streams_;
};

/**
 * @class ZeroCopyBuffer
 * @brief 零拷贝内存缓冲区 - 利用 Jetson 统一内存架构
 * @description CPU 和 GPU 共享同一块物理内存，无需显式拷贝
 */
template<typename T>
class ZeroCopyBuffer {
public:
    ZeroCopyBuffer() = default;
    
    explicit ZeroCopyBuffer(size_t count) {
        allocate(count);
    }
    
    ~ZeroCopyBuffer() {
        deallocate();
    }
    
    // 禁止拷贝
    ZeroCopyBuffer(const ZeroCopyBuffer&) = delete;
    ZeroCopyBuffer& operator=(const ZeroCopyBuffer&) = delete;
    
    // 支持移动
    ZeroCopyBuffer(ZeroCopyBuffer&& other) noexcept 
        : host_ptr_(other.host_ptr_), device_ptr_(other.device_ptr_), 
          count_(other.count_), bytes_(other.bytes_) {
        other.host_ptr_ = nullptr;
        other.device_ptr_ = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }
    
    ZeroCopyBuffer& operator=(ZeroCopyBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            host_ptr_ = other.host_ptr_;
            device_ptr_ = other.device_ptr_;
            count_ = other.count_;
            bytes_ = other.bytes_;
            other.host_ptr_ = nullptr;
            other.device_ptr_ = nullptr;
            other.count_ = 0;
            other.bytes_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t count) {
        if (count == count_ && host_ptr_ != nullptr) return;
        
        deallocate();
        count_ = count;
        bytes_ = count * sizeof(T);
        
        // 使用 cudaHostAlloc 分配零拷贝内存
        // cudaHostAllocMapped: 内存映射到设备地址空间
        // cudaHostAllocWriteCombined: 写合并优化（仅写入时使用）
        CUDA_CHECK(cudaHostAlloc(&host_ptr_, bytes_, 
                                  cudaHostAllocMapped | cudaHostAllocWriteCombined));
        
        // 获取设备指针
        CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0));
    }
    
    void deallocate() {
        if (host_ptr_) {
            CUDA_CHECK_NOTHROW(cudaFreeHost(host_ptr_));
            host_ptr_ = nullptr;
            device_ptr_ = nullptr;
            count_ = 0;
            bytes_ = 0;
        }
    }
    
    T* host() { return host_ptr_; }
    const T* host() const { return host_ptr_; }
    
    T* device() { return device_ptr_; }
    const T* device() const { return device_ptr_; }
    
    size_t count() const { return count_; }
    size_t bytes() const { return bytes_; }
    
    bool empty() const { return host_ptr_ == nullptr; }
    
    // 数组访问运算符（主机端）
    T& operator[](size_t index) { return host_ptr_[index]; }
    const T& operator[](size_t index) const { return host_ptr_[index]; }

private:
    T* host_ptr_ = nullptr;
    T* device_ptr_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};

/**
 * @class DeviceBuffer
 * @brief 设备端内存缓冲区
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    
    explicit DeviceBuffer(size_t count) {
        allocate(count);
    }
    
    ~DeviceBuffer() {
        deallocate();
    }
    
    // 禁止拷贝
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // 支持移动
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            ptr_ = other.ptr_;
            count_ = other.count_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.bytes_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t count) {
        if (count == count_ && ptr_ != nullptr) return;
        
        deallocate();
        count_ = count;
        bytes_ = count * sizeof(T);
        CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
    }
    
    void deallocate() {
        if (ptr_) {
            CUDA_CHECK_NOTHROW(cudaFree(ptr_));
            ptr_ = nullptr;
            count_ = 0;
            bytes_ = 0;
        }
    }
    
    void copyFromHost(const T* src, size_t count, cudaStream_t stream = 0) {
        size_t copy_bytes = std::min(count * sizeof(T), bytes_);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, src, copy_bytes, cudaMemcpyHostToDevice, stream));
    }
    
    void copyToHost(T* dst, size_t count, cudaStream_t stream = 0) {
        size_t copy_bytes = std::min(count * sizeof(T), bytes_);
        CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, copy_bytes, cudaMemcpyDeviceToHost, stream));
    }
    
    void memset(int value, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemsetAsync(ptr_, value, bytes_, stream));
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    size_t count() const { return count_; }
    size_t bytes() const { return bytes_; }
    
    bool empty() const { return ptr_ == nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};

/**
 * @class PinnedBuffer
 * @brief 锁页内存缓冲区 - 用于高速 DMA 传输
 */
template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    
    explicit PinnedBuffer(size_t count) {
        allocate(count);
    }
    
    ~PinnedBuffer() {
        deallocate();
    }
    
    // 禁止拷贝
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    
    // 支持移动
    PinnedBuffer(PinnedBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }
    
    void allocate(size_t count) {
        if (count == count_ && ptr_ != nullptr) return;
        
        deallocate();
        count_ = count;
        bytes_ = count * sizeof(T);
        CUDA_CHECK(cudaHostAlloc(&ptr_, bytes_, cudaHostAllocDefault));
    }
    
    void deallocate() {
        if (ptr_) {
            CUDA_CHECK_NOTHROW(cudaFreeHost(ptr_));
            ptr_ = nullptr;
            count_ = 0;
            bytes_ = 0;
        }
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    size_t count() const { return count_; }
    size_t bytes() const { return bytes_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};

} // namespace jetson

#endif // CUDA_UTILS_H
