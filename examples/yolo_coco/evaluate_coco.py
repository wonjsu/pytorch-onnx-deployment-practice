"""Evaluate YOLOv8n PyTorch, ONNX Runtime, or TensorRT CUDA on COCO 2017.

Both backends execute the same raw model graph and deliberately share the
letterbox and postprocessing implementation in ``yolo_onnx.postprocess_onnx``.
This keeps preprocessing, confidence filtering, coordinate restoration, and
NMS identical instead of relying on backend-specific convenience APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# Support the documented ``python examples\yolo_coco\evaluate_coco.py`` form.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.yolo_onnx.postprocess_onnx import (  # noqa: E402
    letterbox_preprocess_image,
    postprocess_output,
)


DEFAULT_IMAGES_DIR = Path("input/coco/images/val2017")
DEFAULT_ANNOTATION_PATH = Path("input/coco/annotations/instances_val2017.json")
DEFAULT_MODEL_PATH = Path("yolov8n.pt")
DEFAULT_ONNX_PATH = Path("examples/yolo_onnx/artifacts/yolov8n.onnx")
DEFAULT_ENGINE_PATH = Path("examples/yolo_tensorrt/artifacts/yolov8n_fp32_strict.engine")
DEFAULT_OUTPUT_JSON = Path("benchmark-results/coco_predictions.json")
TENSORRT_WARMUP_ITERATIONS = 3

# Ultralytics' contiguous class indices correspond to these official COCO IDs.
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)


class Backend(Protocol):
    """Raw YOLO inference backend."""

    def warmup(self, input_tensor: np.ndarray) -> float: ...
    def infer(self, input_tensor: np.ndarray) -> np.ndarray: ...


@dataclass
class Timings:
    """Accumulated per-image benchmark durations in seconds."""

    preprocess: float = 0.0
    inference: float = 0.0
    h2d: float = 0.0
    tensorrt_compute: float = 0.0
    d2h: float = 0.0
    postprocess: float = 0.0
    tensorrt_total: float = 0.0

    @property
    def total(self) -> float:
        if self.tensorrt_total:
            return self.tensorrt_total
        return self.preprocess + self.inference + self.postprocess


def yolo_class_to_coco_category_id(class_index: int) -> int:
    """Map a contiguous YOLO class index to the sparse official COCO ID."""
    if not 0 <= class_index < len(COCO_CATEGORY_IDS):
        raise ValueError(f"YOLO class index must be in [0, 79], got {class_index}")
    return COCO_CATEGORY_IDS[class_index]


def clip_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return one xyxy box clipped to image boundaries."""
    clipped = np.asarray(box, dtype=np.float32).copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height)
    return clipped


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    """Convert one corner-form box to COCO's top-left width/height form."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def make_prediction(
    image_id: int, class_index: int, box: np.ndarray, score: float
) -> dict[str, int | float | list[float]]:
    """Build a JSON-serializable COCO detection record with stable types."""
    return {
        "image_id": int(image_id),
        "category_id": int(yolo_class_to_coco_category_id(class_index)),
        "bbox": [float(value) for value in xyxy_to_xywh(box)],
        "score": float(score),
    }


def select_image_ids(coco: Any, limit: int | None) -> list[int]:
    """Select deterministic COCO image IDs, applying limit to evaluation too."""
    image_ids = sorted(int(image_id) for image_id in coco.getImgIds())
    return image_ids if limit is None else image_ids[:limit]


def _extract_torch_output(output: Any) -> Any:
    """Extract the raw prediction tensor returned by Ultralytics' torch model."""
    while isinstance(output, (tuple, list)):
        output = output[0]
    if not hasattr(output, "detach"):
        raise RuntimeError("PyTorch model did not return a raw prediction tensor")
    return output


class PyTorchBackend:
    """Ultralytics YOLOv8 raw PyTorch model running strictly on CUDA."""

    def __init__(self, model_path: Path) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is required; refusing CPU fallback")
        if not model_path.is_file():
            raise FileNotFoundError(f"PyTorch model not found: {model_path}")
        self.torch = torch
        self.model = YOLO(str(model_path), task="detect").model.to("cuda").float()
        self.model.eval()

    def _run(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(input_tensor).to("cuda")
        self.torch.cuda.synchronize()
        with self.torch.inference_mode():
            output = _extract_torch_output(self.model(tensor))
        self.torch.cuda.synchronize()
        return output.detach().float().cpu().numpy()

    def warmup(self, input_tensor: np.ndarray) -> float:
        start = time.perf_counter()
        self._run(input_tensor)
        return time.perf_counter() - start

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        return self._run(input_tensor)


class OnnxRuntimeBackend:
    """YOLOv8 ONNX model running strictly with CUDAExecutionProvider."""

    def __init__(self, onnx_path: Path) -> None:
        # torch is imported first so its CUDA DLLs are loaded on Windows. Newer
        # ORT versions can additionally preload compatible DLLs when available.
        import torch  # noqa: F401
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("ONNX session rejected CUDA; refusing CPU fallback")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected one ONNX input, got {len(inputs)}")
        self.input_name = inputs[0].name

    def _run(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one ONNX output, got {len(outputs)}")
        return outputs[0]

    def warmup(self, input_tensor: np.ndarray) -> float:
        start = time.perf_counter()
        self._run(input_tensor)
        return time.perf_counter() - start

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        return self._run(input_tensor)


class TensorRTBackend:
    """TensorRT 11.1 FP32 backend without any fallback path."""

    def __init__(self, engine_path: Path) -> None:
        from examples.yolo_tensorrt.tensorrt_runner import TensorRTRunner

        self.runner = TensorRTRunner(engine_path)

    def warmup(self, input_tensor: np.ndarray) -> float:
        return self.runner.warmup(input_tensor, TENSORRT_WARMUP_ITERATIONS)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        return self.runner.infer(input_tensor)

    def infer_timed(self, input_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return self.runner.infer_timed(input_tensor)


def parse_args() -> argparse.Namespace:
    """Parse COCO benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--engine-path", type=Path, default=DEFAULT_ENGINE_PATH)
    parser.add_argument("--backend", choices=("pytorch", "onnxruntime", "tensorrt"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf-threshold must be between 0 and 1")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")
    return args


def print_timings(timings: Timings, count: int, initialization: float, warmup: float, tensorrt: bool = False) -> None:
    """Print initialization separately and benchmark timing totals."""
    print(f"Initialization time (excluded): {initialization:.3f} s")
    print(f"Warm-up time (excluded): {warmup:.3f} s")
    print(f"Total processing time: {timings.total:.3f} s")
    print(f"Average total time/image: {timings.total / count * 1000:.3f} ms")
    print(f"Preprocessing time: {timings.preprocess:.3f} s")
    if tensorrt:
        print(f"H2D time: {timings.h2d:.3f} s")
        print(f"TensorRT compute time: {timings.tensorrt_compute:.3f} s")
        print(f"D2H time: {timings.d2h:.3f} s")
    else:
        print(f"Inference time: {timings.inference:.3f} s")
    print(f"Postprocessing time: {timings.postprocess:.3f} s")


def main() -> None:
    """Run inference, save COCO predictions, and execute COCOeval."""
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {args.images_dir}")
    if not args.annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {args.annotation_path}")

    coco = COCO(str(args.annotation_path))
    image_ids = select_image_ids(coco, args.limit)
    if not image_ids:
        raise RuntimeError("No COCO images selected")

    initialization_start = time.perf_counter()
    backend: Backend
    if args.backend == "pytorch":
        backend = PyTorchBackend(args.model_path)
    elif args.backend == "onnxruntime":
        backend = OnnxRuntimeBackend(args.onnx_path)
    else:
        backend = TensorRTBackend(args.engine_path)
    initialization = time.perf_counter() - initialization_start

    first_info = coco.loadImgs([image_ids[0]])[0]
    first_tensor, *_ = letterbox_preprocess_image(args.images_dir / first_info["file_name"])
    warmup = backend.warmup(first_tensor)

    timings = Timings()
    predictions: list[dict[str, int | float | list[float]]] = []
    progress_interval = max(1, min(100, len(image_ids) // 10))
    for position, image_id in enumerate(image_ids, start=1):
        image_started = time.perf_counter()
        image_info = coco.loadImgs([image_id])[0]
        image_path = args.images_dir / image_info["file_name"]

        start = time.perf_counter()
        input_tensor, original_size, ratio, pad_x, pad_y = letterbox_preprocess_image(image_path)
        timings.preprocess += time.perf_counter() - start

        if isinstance(backend, TensorRTBackend):
            raw_output, gpu_timings = backend.infer_timed(input_tensor)
            timings.h2d += gpu_timings["h2d_ms"] / 1000.0
            timings.tensorrt_compute += gpu_timings["gpu_compute_ms"] / 1000.0
            timings.d2h += gpu_timings["d2h_ms"] / 1000.0
        else:
            start = time.perf_counter()
            raw_output = backend.infer(input_tensor)
            timings.inference += time.perf_counter() - start

        start = time.perf_counter()
        classes, scores, boxes = postprocess_output(
            raw_output, original_size, ratio, pad_x, pad_y,
            args.conf_threshold, args.iou_threshold,
        )
        for class_index, score, box in zip(classes, scores, boxes):
            clipped = clip_xyxy(box, original_size[0], original_size[1])
            predictions.append(make_prediction(image_id, int(class_index), clipped, float(score)))
        timings.postprocess += time.perf_counter() - start
        if isinstance(backend, TensorRTBackend):
            timings.tensorrt_total += time.perf_counter() - image_started

        if position % progress_interval == 0 or position == len(image_ids):
            print(f"Processed {position}/{len(image_ids)} images")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    print(f"Prediction count: {len(predictions)}")
    print(f"Saved predictions to {args.output_json}")
    print_timings(timings, len(image_ids), initialization, warmup, isinstance(backend, TensorRTBackend))

    # pycocotools.loadRes cannot consume an empty result list.
    if not predictions:
        raise RuntimeError("No predictions were produced; COCOeval cannot load an empty result file")
    result_coco = coco.loadRes(str(args.output_json))
    evaluator = COCOeval(coco, result_coco, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    print("\nRequested metrics")
    print(f"AP 0.50:0.95: {stats[0]:.4f}")
    print(f"AP 0.50: {stats[1]:.4f}")
    print(f"AP 0.75: {stats[2]:.4f}")
    print(f"AP small: {stats[3]:.4f}")
    print(f"AP medium: {stats[4]:.4f}")
    print(f"AP large: {stats[5]:.4f}")
    print(f"AR (maxDets=100): {stats[8]:.4f}")


if __name__ == "__main__":
    main()
