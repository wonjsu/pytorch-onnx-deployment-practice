"""TensorRT 11.1 runner using reusable PyTorch CUDA tensors."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_INPUT_SHAPE = (1, 3, 640, 640)


def trt_dtype_to_torch(dtype: Any, trt: Any, torch: Any) -> Any:
    """Map supported TensorRT scalar types explicitly to PyTorch types."""
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def validate_static_shape(shape: Any, name: str) -> tuple[int, ...]:
    """Return a concrete positive shape or reject dynamic/invalid metadata."""
    result = tuple(int(value) for value in shape)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"Tensor {name!r} has non-static/invalid shape {result}")
    return result


def validate_engine_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"TensorRT engine file is empty: {path}")
    return path


class TensorRTRunner:
    """Own a TensorRT 11.1 context and persistent PyTorch CUDA I/O buffers."""

    def __init__(self, engine_path: Path) -> None:
        validate_engine_path(engine_path)
        self.torch = importlib.import_module("torch")
        self.trt = importlib.import_module("tensorrt")
        if self.trt.__version__ != "11.1.0.106":
            raise RuntimeError(f"TensorRT 11.1.0.106 is required, found {self.trt.__version__}")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; TensorRT has no fallback backend")
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine deserialize failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation failed")

        self.metadata: dict[str, dict[str, Any]] = {}
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.device_buffers: dict[str, Any] = {}
        self.host_buffers: dict[str, Any] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = validate_static_shape(self.engine.get_tensor_shape(name), name)
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = trt_dtype_to_torch(trt_dtype, self.trt, self.torch)
            if trt_dtype != self.trt.float32:
                raise TypeError(f"FP32 engine expected, but tensor {name!r} is {trt_dtype}")
            self.metadata[name] = {"mode": mode, "shape": shape, "trt_dtype": trt_dtype, "torch_dtype": torch_dtype}
            (self.input_names if mode == self.trt.TensorIOMode.INPUT else self.output_names).append(name)
            self.device_buffers[name] = self.torch.empty(shape, dtype=torch_dtype, device="cuda")
            if mode == self.trt.TensorIOMode.OUTPUT:
                self.host_buffers[name] = self.torch.empty(shape, dtype=torch_dtype, pin_memory=True)
            if not self.context.set_tensor_address(name, self.device_buffers[name].data_ptr()):
                raise RuntimeError(f"Tensor address registration failed for {name!r}")
        if len(self.input_names) != 1:
            raise ValueError(f"Exactly one input is required, got {self.input_names}")
        self.input_name = self.input_names[0]
        if self.metadata[self.input_name]["shape"] != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"Expected static input {EXPECTED_INPUT_SHAPE}, got {self.metadata[self.input_name]['shape']}")
        if not self.output_names:
            raise ValueError("Engine has no output tensors")

    def infer_outputs_timed(self, input_array: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Execute H2D, compute and D2H on the current PyTorch stream with events."""
        expected = self.metadata[self.input_name]["shape"]
        if input_array.shape != expected:
            raise ValueError(f"Input shape mismatch: expected {expected}, got {input_array.shape}")
        if input_array.dtype != np.float32:
            raise TypeError(f"Input dtype mismatch: expected float32, got {input_array.dtype}")
        cpu_input = self.torch.from_numpy(np.ascontiguousarray(input_array))
        stream = self.torch.cuda.current_stream()
        events = [self.torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record(stream)
        self.device_buffers[self.input_name].copy_(cpu_input, non_blocking=True)
        events[1].record(stream)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        events[2].record(stream)
        for name in self.output_names:
            self.host_buffers[name].copy_(self.device_buffers[name], non_blocking=True)
        events[3].record(stream)
        events[3].synchronize()
        timings = {
            "h2d_ms": events[0].elapsed_time(events[1]),
            "gpu_compute_ms": events[1].elapsed_time(events[2]),
            "d2h_ms": events[2].elapsed_time(events[3]),
        }
        try:
            outputs = {name: self.host_buffers[name].numpy().copy() for name in self.output_names}
        except Exception as exc:
            raise RuntimeError("TensorRT output shape/D2H conversion failed") from exc
        return outputs, timings

    def infer_outputs(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        return self.infer_outputs_timed(input_array)[0]

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        outputs = self.infer_outputs(input_array)
        if len(outputs) != 1:
            raise RuntimeError(f"Single-output inference requested, engine returned {list(outputs)}")
        return next(iter(outputs.values()))

    def infer_timed(self, input_array: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        outputs, timings = self.infer_outputs_timed(input_array)
        if len(outputs) != 1:
            raise RuntimeError(f"Single-output inference requested, engine returned {list(outputs)}")
        return next(iter(outputs.values())), timings

    def warmup(self, input_array: np.ndarray, iterations: int = 3) -> float:
        import time
        started = time.perf_counter()
        for _ in range(iterations):
            self.infer_outputs(input_array)
        return time.perf_counter() - started

    def __enter__(self) -> "TensorRTRunner":
        return self

    def __exit__(self, *_: object) -> None:
        return None
