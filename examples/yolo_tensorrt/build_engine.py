"""Build a static, FP32 YOLO engine with the TensorRT 11.1 Python API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def positive_workspace_gb(value: str) -> float:
    """Argparse converter for a finite, positive workspace size."""
    try:
        workspace = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workspace must be a number") from exc
    if not workspace > 0 or workspace == float("inf"):
        raise argparse.ArgumentTypeError("workspace must be a finite positive number")
    return workspace


def existing_onnx_path(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"ONNX file not found: {path}")
    return path


def writable_engine_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() not in {".engine", ".plan"}:
        raise argparse.ArgumentTypeError("engine path must end in .engine or .plan")
    if path.exists() and not path.is_file():
        raise argparse.ArgumentTypeError(f"engine path is not a file: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=existing_onnx_path, required=True)
    parser.add_argument("--engine-path", type=writable_engine_path, required=True)
    parser.add_argument("--workspace-gb", type=positive_workspace_gb, default=1.0)
    parser.add_argument("--tf32", choices=("off", "on"), default="off")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _shape(shape: Any, tensor_name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in shape)
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Tensor {tensor_name!r} has a non-static/invalid shape: {values}")
    return values


def _relative_or_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    args = parse_args()
    import tensorrt as trt
    import torch

    if trt.__version__ != "11.1.0.106":
        raise RuntimeError(f"TensorRT 11.1.0.106 is required, found {trt.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to build for an unknown GPU")

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = args.onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        print(f"ONNX path: {args.onnx_path}")
        print(f"parser.num_errors: {parser.num_errors}")
        for index in range(parser.num_errors):
            print(f"parser.get_error({index}): {parser.get_error(index)}")
        raise RuntimeError("TensorRT ONNX parsing failed; no engine was written")

    for index in range(network.num_inputs):
        _shape(network.get_input(index).shape, network.get_input(index).name)
    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * 2**30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    tf32_enabled = args.tf32 == "on"
    if tf32_enabled:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; no engine was written")
    engine_bytes = bytes(serialized)
    if not engine_bytes:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    args.engine_path.parent.mkdir(parents=True, exist_ok=True)
    args.engine_path.write_bytes(engine_bytes)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        args.engine_path.unlink(missing_ok=True)
        raise RuntimeError("New engine could not be deserialized")
    io_metadata = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        item = {
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "shape": list(_shape(engine.get_tensor_shape(name), name)),
            "dtype": str(engine.get_tensor_dtype(name)),
        }
        io_metadata.append(item)

    metadata = {
        "tensorrt_version": trt.__version__,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "onnx_path": _relative_or_name(args.onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "engine_path": _relative_or_name(args.engine_path),
        "tf32_enabled": tf32_enabled,
        "workspace_bytes": workspace_bytes,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_time_seconds": build_seconds,
        "io_tensors": io_metadata,
    }
    metadata_path = args.engine_path.with_suffix(args.engine_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for key in ("tensorrt_version", "pytorch_version", "cuda_version", "gpu_name", "onnx_path", "engine_path"):
        print(f"{key}: {metadata[key]}")
    print(f"engine_file_size: {len(engine_bytes)} bytes")
    print(f"TF32: {'enabled' if tf32_enabled else 'disabled'}")
    print(f"workspace: {workspace_bytes} bytes")
    print(f"engine_build_time: {build_seconds:.3f} s")
    for item in io_metadata:
        print(f"I/O: name={item['name']} mode={item['mode']} shape={tuple(item['shape'])} dtype={item['dtype']}")
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
