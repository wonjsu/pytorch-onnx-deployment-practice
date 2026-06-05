"""Run YOLOv8n ONNX inference and postprocess raw detections."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = EXAMPLE_DIR / "artifacts" / "yolov8n.onnx"
INPUT_SIZE = (640, 640)
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
EXPECTED_OUTPUT_SHAPE = (1, 84, 8400)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n ONNX inference and print postprocessed detections."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an input image for YOLOv8n ONNX postprocessing.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=f"Path to the YOLOv8n ONNX model. Default: {DEFAULT_ONNX_PATH}",
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
    return parser.parse_args()


def preprocess_image(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image and convert it to a YOLOv8n 640x640 NCHW tensor."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized_image = image.resize(INPUT_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0), original_size


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
) -> np.ndarray:
    """Scale 640x640 input-space boxes back to the original image coordinates."""
    original_width, original_height = original_size
    scale = np.array(
        [
            original_width / INPUT_SIZE[0],
            original_height / INPUT_SIZE[1],
            original_width / INPUT_SIZE[0],
            original_height / INPUT_SIZE[1],
        ],
        dtype=np.float32,
    )
    boxes = boxes * scale
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
    boxes_xyxy = restore_original_coordinates(boxes_xyxy, original_size)

    selected_indices = class_aware_nms(
        boxes_xyxy, confidences, class_indices, iou_threshold
    )
    return (
        class_indices[selected_indices],
        confidences[selected_indices],
        boxes_xyxy[selected_indices],
    )


def main() -> None:
    """Run ONNX Runtime inference and print final detection results."""
    args = parse_args()

    onnx_path = args.onnx_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    input_tensor, original_size = preprocess_image(args.image_path)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    if len(outputs) != 1:
        raise ValueError(f"Expected one YOLOv8n ONNX output, but got {len(outputs)}")

    class_indices, confidences, boxes = postprocess_output(
        outputs[0],
        original_size,
        args.conf_threshold,
        args.iou_threshold,
    )

    if len(boxes) == 0:
        print("No detections")
        return

    for class_index, confidence, box in zip(class_indices, confidences, boxes):
        x1, y1, x2, y2 = box
        print(
            "class_index="
            f"{int(class_index)} "
            f"confidence={float(confidence):.4f} "
            f"bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
        )


if __name__ == "__main__":
    main()
