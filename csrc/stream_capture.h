#pragma once

namespace vllm {

// Thread-local depth incremented while Python holds vLLM's `graph_capture`
// contexts (see `parallel_state.GroupCoordinator.graph_capture`). Custom
// kernels can use this to avoid per-call cudaStreamIsCapturing /
// hipStreamIsCapturing queries when vLLM is not in a known capture scope.

void stream_capture_enter();
void stream_capture_leave();
int stream_capture_depth();

void stream_capture_bump_forward_epoch();

}  // namespace vllm
