"""Execute a static YOLOv8n TensorRT engine with reusable CUDA buffers."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.yolo_onnx.postprocess_onnx import (
    EXPECTED_OUTPUT_SHAPE,
    get_coco_class_name,
    letterbox_preprocess_image,
    postprocess_output,
)

EXPECTED_INPUT_SHAPE = (1, 3, 640, 640)


def _cuda_check(result: tuple[Any, ...], operation: str) -> Any:
    """Validate a cuda-python runtime result and return its payload."""
    error, *payload = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA {operation} failed with error code {int(error)}")
    if not payload:
        return None
    return payload[0] if len(payload) == 1 else tuple(payload)


class EngineAdapter:
    """Normalize TensorRT 8 binding APIs and TensorRT 10 tensor APIs."""

    def __init__(self, engine: Any, context: Any, trt: Any) -> None:
        self.engine, self.context, self.trt = engine, context, trt
        self.modern = hasattr(engine, "num_io_tensors")
        count = engine.num_io_tensors if self.modern else engine.num_bindings
        self.names = [engine.get_tensor_name(i) if self.modern else engine.get_binding_name(i) for i in range(count)]
        self.input_names = [n for i, n in enumerate(self.names) if self._is_input(i, n)]
        self.output_names = [n for i, n in enumerate(self.names) if not self._is_input(i, n)]

    def _is_input(self, index: int, name: str) -> bool:
        if self.modern:
            return self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
        return bool(self.engine.binding_is_input(index))

    def shape(self, name: str) -> tuple[int, ...]:
        """Return a context-resolved tensor shape."""
        shape = self.context.get_tensor_shape(name) if self.modern else self.context.get_binding_shape(self.names.index(name))
        return tuple(int(value) for value in shape)

    def dtype(self, name: str) -> np.dtype[Any]:
        """Return the NumPy dtype of a tensor."""
        dtype = self.engine.get_tensor_dtype(name) if self.modern else self.engine.get_binding_dtype(self.names.index(name))
        return np.dtype(self.trt.nptype(dtype))

    def execute(self, stream: int, pointers: dict[str, int]) -> None:
        """Enqueue inference using the correct TensorRT generation API."""
        if self.modern:
            for name, pointer in pointers.items():
                self.context.set_tensor_address(name, pointer)
            ok = self.context.execute_async_v3(stream_handle=stream)
        else:
            bindings = [pointers[name] for name in self.names]
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT enqueue failed")


class TensorRTRunner:
    """Own an engine, execution context, stream, and persistent device buffers."""

    def __init__(self, engine_path: Path) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
        try:
            trt = importlib.import_module("tensorrt")
            try:
                cudart = importlib.import_module("cuda.bindings.runtime")
            except ImportError:
                cudart = importlib.import_module("cuda.cudart")
        except (ImportError, OSError) as exc:
            raise RuntimeError("TensorRT and NVIDIA cuda-python are required; install requirements-tensorrt.txt in an NVIDIA CUDA environment") from exc
        self.trt, self.cudart = trt, cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.runtime, self.engine, self.context = runtime, engine, context
        self.adapter = EngineAdapter(engine, context, trt)
        if len(self.adapter.input_names) != 1 or len(self.adapter.output_names) != 1:
            raise ValueError(f"Expected one input and one output, found inputs={self.adapter.input_names}, outputs={self.adapter.output_names}")
        self.input_name, self.output_name = self.adapter.input_names[0], self.adapter.output_names[0]
        if self.adapter.shape(self.input_name) != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"Expected static input shape {EXPECTED_INPUT_SHAPE}, got {self.adapter.shape(self.input_name)}")
        if self.adapter.shape(self.output_name) != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(f"Expected YOLOv8n output shape {EXPECTED_OUTPUT_SHAPE}, got {self.adapter.shape(self.output_name)}")
        self.stream = int(_cuda_check(cudart.cudaStreamCreate(), "stream creation"))
        self.host_output = np.empty(EXPECTED_OUTPUT_SHAPE, dtype=self.adapter.dtype(self.output_name))
        self.pointers: dict[str, int] = {}
        for name in self.adapter.names:
            size = int(np.prod(self.adapter.shape(name))) * self.adapter.dtype(name).itemsize
            self.pointers[name] = int(_cuda_check(cudart.cudaMalloc(size), f"allocation for {name}"))

    def infer(self, input_tensor: np.ndarray, synchronize: bool = True) -> np.ndarray:
        """Copy input, enqueue inference, copy output, and optionally synchronize."""
        if input_tensor.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {input_tensor.shape}")
        array = np.ascontiguousarray(input_tensor, dtype=self.adapter.dtype(self.input_name))
        kind = self.cudart.cudaMemcpyKind
        _cuda_check(self.cudart.cudaMemcpyAsync(self.pointers[self.input_name], array.ctypes.data, array.nbytes, kind.cudaMemcpyHostToDevice, self.stream), "H2D copy")
        self.adapter.execute(self.stream, self.pointers)
        _cuda_check(self.cudart.cudaMemcpyAsync(self.host_output.ctypes.data, self.pointers[self.output_name], self.host_output.nbytes, kind.cudaMemcpyDeviceToHost, self.stream), "D2H copy")
        if synchronize:
            _cuda_check(self.cudart.cudaStreamSynchronize(self.stream), "stream synchronization")
        return self.host_output.copy()

    def infer_timed(self, input_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Run inference and time H2D, enqueue/compute, and D2H with CUDA events."""
        if input_tensor.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {input_tensor.shape}")
        array = np.ascontiguousarray(input_tensor, dtype=self.adapter.dtype(self.input_name))
        events = [int(_cuda_check(self.cudart.cudaEventCreate(), "event creation")) for _ in range(4)]
        kind = self.cudart.cudaMemcpyKind
        try:
            _cuda_check(self.cudart.cudaEventRecord(events[0], self.stream), "event record")
            _cuda_check(self.cudart.cudaMemcpyAsync(self.pointers[self.input_name], array.ctypes.data, array.nbytes, kind.cudaMemcpyHostToDevice, self.stream), "H2D copy")
            _cuda_check(self.cudart.cudaEventRecord(events[1], self.stream), "event record")
            self.adapter.execute(self.stream, self.pointers)
            _cuda_check(self.cudart.cudaEventRecord(events[2], self.stream), "event record")
            _cuda_check(self.cudart.cudaMemcpyAsync(self.host_output.ctypes.data, self.pointers[self.output_name], self.host_output.nbytes, kind.cudaMemcpyDeviceToHost, self.stream), "D2H copy")
            _cuda_check(self.cudart.cudaEventRecord(events[3], self.stream), "event record")
            _cuda_check(self.cudart.cudaEventSynchronize(events[3]), "event synchronization")
            elapsed = lambda a, b: float(_cuda_check(self.cudart.cudaEventElapsedTime(a, b), "event elapsed time"))
            timings = {"h2d_ms": elapsed(events[0], events[1]), "gpu_compute_ms": elapsed(events[1], events[2]), "d2h_ms": elapsed(events[2], events[3])}
            return self.host_output.copy(), timings
        finally:
            for event in events:
                _cuda_check(self.cudart.cudaEventDestroy(event), "event destruction")

    def close(self) -> None:
        """Release CUDA allocations and stream."""
        for pointer in self.pointers.values():
            _cuda_check(self.cudart.cudaFree(pointer), "free")
        self.pointers.clear()
        if self.stream:
            _cuda_check(self.cudart.cudaStreamDestroy(self.stream), "stream destruction")
            self.stream = 0

    def __enter__(self) -> "TensorRTRunner": return self
    def __exit__(self, *_: object) -> None: self.close()


def main() -> None:
    """Run TensorRT on one image and print raw metadata and detections."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    args = parser.parse_args()
    tensor, original_size, ratio, pad_x, pad_y = letterbox_preprocess_image(args.image_path)
    with TensorRTRunner(args.engine_path) as runner:
        print(f"input: name={runner.input_name} shape={runner.adapter.shape(runner.input_name)}")
        print(f"output: name={runner.output_name} shape={runner.adapter.shape(runner.output_name)}")
        raw = runner.infer(tensor)
    print(f"raw output shape: {raw.shape}")
    classes, scores, boxes = postprocess_output(raw.astype(np.float32), original_size, ratio, pad_x, pad_y, args.conf_threshold, args.iou_threshold)
    print(f"detection count: {len(boxes)}")
    for cls, score, box in zip(classes, scores, boxes):
        print(f"class={get_coco_class_name(int(cls))} confidence={float(score):.4f} bbox={tuple(float(x) for x in box)}")


if __name__ == "__main__": main()
