# SPDX-License-Identifier: Apache-2.0
"""CUDA/HIP graph capture helpers and ROCm `hipStreamIsCapturing` mitigation.

vLLM extension (`_C_cuda_utils`): `stream_capture_enter` /
`stream_capture_leave` / `stream_capture_bump_forward_epoch` so
`custom_all_reduce` avoids redundant driver queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from vllm.platforms import current_platform


def maybe_bump_rocm_stream_capture_forward_epoch() -> None:
    """Bump ROCm capture-cache epoch once per worker forward step."""
    if not current_platform.is_rocm():
        return
    try:
        torch.ops._C_cuda_utils.stream_capture_bump_forward_epoch()
    except AttributeError:
        pass


@contextmanager
def vllm_graph_capture_stream_mark() -> Iterator[None]:
    try:
        enter = torch.ops._C_cuda_utils.stream_capture_enter
        leave = torch.ops._C_cuda_utils.stream_capture_leave
    except AttributeError:
        yield
        return
    enter()
    try:
        yield
    finally:
        leave()
