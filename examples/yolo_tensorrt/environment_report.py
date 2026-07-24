"""Print deployment environment details without requiring optional packages."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from typing import Any


def optional_import(name: str) -> Any | None:
    """Import an optional package, returning ``None`` on any import failure."""
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def command_output(command: list[str]) -> str:
    """Return the first line printed by a command or ``unavailable``."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else "unavailable"
    except (OSError, IndexError):
        return "unavailable"


def main() -> None:
    """Print OS, Python, framework, CUDA, GPU, and driver information."""
    torch = optional_import("torch")
    onnx = optional_import("onnx")
    ort = optional_import("onnxruntime")
    trt = optional_import("tensorrt")
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    vram = (
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB"
        if cuda_available else "unavailable"
    )
    capability = (
        ".".join(map(str, torch.cuda.get_device_capability(0)))
        if cuda_available else "unavailable"
    )
    values = {
        "OS": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": getattr(torch, "__version__", "unavailable"),
        "CUDA runtime version": getattr(getattr(torch, "version", None), "cuda", None) or "unavailable",
        "CUDA available": cuda_available,
        "GPU name": gpu_name,
        "GPU total VRAM": vram,
        "Compute capability": capability,
        "ONNX version": getattr(onnx, "__version__", "unavailable"),
        "ONNX Runtime version": getattr(ort, "__version__", "unavailable"),
        "ONNX Runtime providers": ort.get_available_providers() if ort else "unavailable",
        "TensorRT version": getattr(trt, "__version__", "unavailable"),
        "NVIDIA driver version": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
    }
    for key, value in values.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
