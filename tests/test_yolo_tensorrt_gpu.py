"""Explicitly skipped GPU integration-test placeholder for CI discovery."""

import pytest


@pytest.mark.skip(reason="requires an NVIDIA GPU, TensorRT engine, and cuda-python")
def test_tensorrt_engine_integration() -> None:
    """Document that real-engine validation is performed on the target GPU."""
