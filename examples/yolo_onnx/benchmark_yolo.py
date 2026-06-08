"""Benchmark YOLOv8n PyTorch, Ultralytics ONNX, and direct ONNX Runtime latency."""

import argparse
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import onnxruntime as ort
from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = EXAMPLE_DIR / "artifacts" / "yolov8n.onnx"
INPUT_SIZE = (640, 640)
PADDING_VALUE = 114
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASUREMENT_RUNS = 100
EXPECTED_OUTPUT_SHAPE = (1, 84, 8400)

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark YOLOv8n PyTorch/Ultralytics and ONNX Runtime latency."
        )
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an input image for benchmarking YOLOv8n latency.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=f"Path to the exported YOLOv8n ONNX model. Default: {DEFAULT_ONNX_PATH}",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold. Default: {DEFAULT_CONF_THRESHOLD}",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for NMS. Default: {DEFAULT_IOU_THRESHOLD}",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"Number of warmup runs before timing. Default: {DEFAULT_WARMUP_RUNS}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_MEASUREMENT_RUNS,
        help=f"Number of timed measurement runs. Default: {DEFAULT_MEASUREMENT_RUNS}",
    )
    return parser.parse_args()


def letterbox_preprocess_image(
    image_path: Path,
) -> tuple[np.ndarray, tuple[int, int], float, int, int]:
    """Load an image and letterbox it to a YOLOv8n 640x640 NCHW tensor."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    original_width, original_height = original_size
    target_width, target_height = INPUT_SIZE

    scale_ratio = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_ratio))
    resized_height = int(round(original_height * scale_ratio))
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2

    resized_image = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    letterboxed_image = Image.new(
        "RGB", INPUT_SIZE, (PADDING_VALUE, PADDING_VALUE, PADDING_VALUE)
    )
    letterboxed_image.paste(resized_image, (pad_x, pad_y))

    image_array = np.asarray(letterboxed_image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    input_tensor = np.expand_dims(image_array, axis=0)
    return input_tensor, original_size, scale_ratio, pad_x, pad_y


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from center x/y, width, height to x1, y1, x2, y2."""
    half_width = boxes[:, 2] / 2.0
    half_height = boxes[:, 3] / 2.0

    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - half_width
    converted[:, 1] = boxes[:, 1] - half_height
    converted[:, 2] = boxes[:, 0] + half_width
    converted[:, 3] = boxes[:, 1] + half_height
    return converted


def restore_original_coordinates(
    boxes: np.ndarray,
    original_size: tuple[int, int],
    scale_ratio: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Undo letterbox padding/scale and clip boxes to original image bounds."""
    original_width, original_height = original_size
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    return boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and many boxes in x1, y1, x2, y2 format."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_width = np.maximum(0.0, x2 - x1)
    intersection_height = np.maximum(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Apply Non-Maximum Suppression to one set of boxes."""
    if len(boxes) == 0:
        return []

    selected_indices = []
    remaining_indices = np.argsort(scores)[::-1]

    while remaining_indices.size > 0:
        current_index = int(remaining_indices[0])
        selected_indices.append(current_index)

        if remaining_indices.size == 1:
            break

        remaining_without_current = remaining_indices[1:]
        ious = compute_iou(boxes[current_index], boxes[remaining_without_current])
        remaining_indices = remaining_without_current[ious <= iou_threshold]

    return selected_indices


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Apply NMS independently for each predicted class."""
    selected_indices = []

    for class_index in np.unique(class_indices):
        class_mask = class_indices == class_index
        original_indices = np.flatnonzero(class_mask)
        selected_for_class = nms(boxes[class_mask], scores[class_mask], iou_threshold)
        selected_indices.extend(original_indices[selected_for_class].tolist())

    selected_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    return selected_indices


def postprocess_output(
    raw_output: np.ndarray,
    original_size: tuple[int, int],
    scale_ratio: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert YOLOv8n raw output into final class IDs, confidences, and boxes."""
    if raw_output.shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"Expected YOLOv8n raw output shape {EXPECTED_OUTPUT_SHAPE}, "
            f"but got {raw_output.shape}"
        )

    candidates = raw_output[0].transpose(1, 0)
    boxes_xywh = candidates[:, :4]
    class_scores = candidates[:, 4:]

    confidences = np.max(class_scores, axis=1)
    class_indices = np.argmax(class_scores, axis=1)

    confidence_mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[confidence_mask]
    confidences = confidences[confidence_mask]
    class_indices = class_indices[confidence_mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    boxes_xyxy = restore_original_coordinates(
        boxes_xyxy, original_size, scale_ratio, pad_x, pad_y
    )

    selected_indices = class_aware_nms(
        boxes_xyxy, confidences, class_indices, iou_threshold
    )
    return (
        class_indices[selected_indices],
        confidences[selected_indices],
        boxes_xyxy[selected_indices],
    )


def benchmark_ms(
    function: Callable[[], T], warmup_runs: int, measurement_runs: int
) -> float:
    """Warm up a callable, then return average measured latency in milliseconds."""
    if warmup_runs < 0:
        raise ValueError("warmup must be greater than or equal to 0")
    if measurement_runs <= 0:
        raise ValueError("runs must be greater than 0")

    for _ in range(warmup_runs):
        function()

    started_at = time.perf_counter()
    for _ in range(measurement_runs):
        function()
    elapsed_seconds = time.perf_counter() - started_at
    return elapsed_seconds * 1000.0 / measurement_runs


def run_direct_onnx(
    session: ort.InferenceSession, input_name: str, input_tensor: np.ndarray
) -> np.ndarray:
    """Run direct ONNX Runtime inference and validate that one output is returned."""
    outputs = session.run(None, {input_name: input_tensor})
    if len(outputs) != 1:
        raise ValueError(f"Expected one YOLOv8n ONNX output, but got {len(outputs)}")
    return outputs[0]


def main() -> None:
    """Run YOLO latency benchmarks and print average latency in milliseconds."""
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {args.onnx_path}")

    from ultralytics import YOLO

    pytorch_model = YOLO("yolov8n.pt")
    ultralytics_onnx_model = YOLO(str(args.onnx_path))
    session = ort.InferenceSession(
        str(args.onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    input_tensor, original_size, scale_ratio, pad_x, pad_y = letterbox_preprocess_image(
        args.image_path
    )
    raw_output = run_direct_onnx(session, input_name, input_tensor)

    def run_ultralytics_pytorch() -> object:
        return pytorch_model(
            str(args.image_path),
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=False,
        )

    def run_ultralytics_onnx() -> object:
        return ultralytics_onnx_model(
            str(args.image_path),
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=False,
        )

    def run_direct_inference_only() -> np.ndarray:
        return run_direct_onnx(session, input_name, input_tensor)

    def run_direct_postprocess() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return postprocess_output(
            raw_output.copy(),
            original_size,
            scale_ratio,
            pad_x,
            pad_y,
            args.conf_threshold,
            args.iou_threshold,
        )

    def run_direct_end_to_end() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_input, current_size, current_scale, current_pad_x, current_pad_y = (
            letterbox_preprocess_image(args.image_path)
        )
        current_raw_output = run_direct_onnx(session, input_name, current_input)
        return postprocess_output(
            current_raw_output,
            current_size,
            current_scale,
            current_pad_x,
            current_pad_y,
            args.conf_threshold,
            args.iou_threshold,
        )

    pytorch_latency = benchmark_ms(
        run_ultralytics_pytorch, args.warmup, args.runs
    )
    ultralytics_onnx_latency = benchmark_ms(
        run_ultralytics_onnx, args.warmup, args.runs
    )
    direct_inference_latency = benchmark_ms(
        run_direct_inference_only, args.warmup, args.runs
    )
    direct_postprocess_latency = benchmark_ms(
        run_direct_postprocess, args.warmup, args.runs
    )
    direct_end_to_end_latency = benchmark_ms(
        run_direct_end_to_end, args.warmup, args.runs
    )

    print(f"Image path: {args.image_path}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Measurement runs: {args.runs}")
    print(f"Ultralytics PyTorch end-to-end latency(ms): {pytorch_latency:.3f}")
    print(f"Ultralytics ONNX end-to-end latency(ms): {ultralytics_onnx_latency:.3f}")
    print(
        "Direct ONNX Runtime inference-only latency(ms): "
        f"{direct_inference_latency:.3f}"
    )
    print(
        "Direct ONNX Runtime postprocess latency(ms): "
        f"{direct_postprocess_latency:.3f}"
    )
    print(
        "Direct ONNX Runtime end-to-end latency(ms): "
        f"{direct_end_to_end_latency:.3f}"
    )


if __name__ == "__main__":
    main()
