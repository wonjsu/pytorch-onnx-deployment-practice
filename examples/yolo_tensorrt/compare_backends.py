"""Compare raw and postprocessed YOLOv8n outputs across three GPU backends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess_image, postprocess_output
from examples.yolo_tensorrt.infer_tensorrt import TensorRTRunner


@dataclass(frozen=True)
class Detection:
    """A backend-independent final detection."""
    class_index: int
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Match:
    """Indices and IoU for a same-class detection match."""
    reference_index: int
    candidate_index: int
    iou: float


def bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Calculate IoU between two xyxy boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(reference: list[Detection], candidate: list[Detection], threshold: float) -> tuple[list[Match], list[int], list[int]]:
    """Greedily match same-class detections in descending IoU order."""
    pairs = sorted(
        (Match(i, j, bbox_iou(a.box, b.box)) for i, a in enumerate(reference) for j, b in enumerate(candidate) if a.class_index == b.class_index),
        key=lambda item: item.iou, reverse=True,
    )
    matches: list[Match] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for pair in pairs:
        if pair.iou >= threshold and pair.reference_index not in used_a and pair.candidate_index not in used_b:
            matches.append(pair); used_a.add(pair.reference_index); used_b.add(pair.candidate_index)
    return matches, [i for i in range(len(reference)) if i not in used_a], [i for i in range(len(candidate)) if i not in used_b]


def to_detections(raw: np.ndarray, metadata: tuple[tuple[int, int], float, int, int], conf: float, iou: float) -> list[Detection]:
    """Apply the shared postprocessor and construct detection objects."""
    classes, scores, boxes = postprocess_output(raw.astype(np.float32), *metadata, conf, iou)
    return [Detection(int(c), float(s), tuple(map(float, b))) for c, s, b in zip(classes, scores, boxes)]


def pytorch_raw(weights: Path, tensor: np.ndarray) -> np.ndarray:
    """Run only the Ultralytics model module on the already-letterboxed tensor."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required")
    model = YOLO(str(weights)).model.eval().cuda().float()
    with torch.inference_mode():
        output = model(torch.from_numpy(tensor).cuda())
    raw = output[0] if isinstance(output, (tuple, list)) else output
    return raw.detach().float().cpu().numpy()


def onnx_raw(model_path: Path, tensor: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime CUDA directly on the shared input tensor."""
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: tensor})[0]


def main() -> None:
    """Run all backends and print raw differences and detection matches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path); parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--onnx-path", type=Path, required=True); parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25); parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    tensor, size, ratio, pad_x, pad_y = letterbox_preprocess_image(args.image_path)
    metadata = (size, ratio, pad_x, pad_y)
    raw = {"PyTorch CUDA FP32": pytorch_raw(args.weights, tensor), "ONNX Runtime CUDA FP32": onnx_raw(args.onnx_path, tensor)}
    with TensorRTRunner(args.engine_path) as runner: raw["TensorRT"] = runner.infer(tensor)
    reference_name = "PyTorch CUDA FP32"
    detections = {name: to_detections(value, metadata, args.conf_threshold, args.iou_threshold) for name, value in raw.items()}
    for name, value in raw.items(): print(f"{name} raw output shape: {value.shape}")
    for name in ("ONNX Runtime CUDA FP32", "TensorRT"):
        difference = np.abs(raw[reference_name].astype(np.float32) - raw[name].astype(np.float32))
        print(f"\n{reference_name} vs {name}: raw MAE={difference.mean():.8f}, max_abs_diff={difference.max():.8f}")
        matches, unmatched_ref, unmatched_candidate = match_detections(detections[reference_name], detections[name], args.match_iou_threshold)
        for match in matches:
            a, b = detections[reference_name][match.reference_index], detections[name][match.candidate_index]
            print(f"matched class={a.class_index} confidence_abs_diff={abs(a.confidence-b.confidence):.6f} bbox_iou={match.iou:.6f}")
        print(f"unmatched {reference_name}: {unmatched_ref}; unmatched {name}: {unmatched_candidate}")
        print(f"detection count: {reference_name}={len(detections[reference_name])}, {name}={len(detections[name])}")


if __name__ == "__main__": main()
