"""Smoke test — just verifies the package imports and torch sees the GPU."""

from __future__ import annotations

import torch


def test_decode_lab_imports() -> None:
    import decode_lab

    assert decode_lab.__version__ == "0.1.0"


def test_torch_available() -> None:
    assert torch.__version__


def test_cuda_available() -> None:
    """If this fails on your 1650 Ti, your PyTorch install is wrong (CPU-only build)."""
    assert torch.cuda.is_available(), "CUDA not available — reinstall PyTorch with cu121 wheel"
