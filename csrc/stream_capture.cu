#include "stream_capture.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace vllm {
namespace {

thread_local int g_stream_capture_depth = 0;

static std::atomic<uint64_t> g_forward_epoch{0};

thread_local std::unordered_map<uintptr_t,
                                std::pair<uint64_t, cudaStreamCaptureStatus>>
    g_rocm_stream_capture_cache;

}  // namespace

void stream_capture_enter() { ++g_stream_capture_depth; }

void stream_capture_leave() {
  if (g_stream_capture_depth > 0) {
    --g_stream_capture_depth;
  }
}

int stream_capture_depth() { return g_stream_capture_depth; }

void stream_capture_bump_forward_epoch() {
  g_forward_epoch.fetch_add(1, std::memory_order_relaxed);
}

void stream_capture_get_stream_capture_status(
    cudaStream_t stream, cudaStreamCaptureStatus* status) {
  if (stream_capture_depth() > 0) {
    *status = cudaStreamCaptureStatusActive;
    return;
  }
#if defined(USE_ROCM)
  const uint64_t ep = g_forward_epoch.load(std::memory_order_relaxed);
  const uintptr_t key = reinterpret_cast<uintptr_t>(stream);
  const auto it = g_rocm_stream_capture_cache.find(key);
  if (it != g_rocm_stream_capture_cache.end() && it->second.first == ep) {
    *status = it->second.second;
    return;
  }
  CUDA_CHECK(cudaStreamIsCapturing(stream, status));
  g_rocm_stream_capture_cache[key] = {ep, *status};
#else
  CUDA_CHECK(cudaStreamIsCapturing(stream, status));
#endif
}

}  // namespace vllm
