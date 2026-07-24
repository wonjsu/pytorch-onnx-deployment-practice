"""Compare Ultralytics YOLOv8n PyTorch and exported ONNX detections."""

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ONNX_PATH = Path("examples/yolo_onnx/artifacts/yolov8n.onnx")
DEFAULT_CONF_THRESHOLD = 0.05
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MATCH_IOU_THRESHOLD = 0.5

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLOv8n PyTorch and exported ONNX detections using only "
            "the Ultralytics YOLO API."
        )
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an input image for comparing YOLOv8n detections.",
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
        "--match-iou-threshold",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help=(
            "Minimum bbox IoU for considering a PyTorch/ONNX detection pair "
            f"matched. Default: {DEFAULT_MATCH_IOU_THRESHOLD}"
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Detection:
    """One object detection in original image coordinates."""

    class_index: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionPair:
    """A PyTorch detection paired with an ONNX detection by bbox IoU."""

    pytorch_index: int
    onnx_index: int
    iou: float


def format_detection(
    class_index: int,
    class_name: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> str:
    """Format one detection for visual comparison."""
    x1, y1, x2, y2 = box
    return (
        f"class_index={class_index} "
        f"class_name={class_name} "
        f"confidence={confidence:.4f} "
        f"bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
    )


def get_class_name(names: dict[int, str] | list[str], class_index: int) -> str:
    """Return the COCO class name for a YOLO class index when available."""
    if 0 <= class_index < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_index]
    if isinstance(names, dict):
        return names.get(class_index, "unknown")
    if 0 <= class_index < len(names):
        return names[class_index]
    return "unknown"


def collect_ultralytics_detections(
    model_path: str | Path,
    image_path: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Run Ultralytics prediction and return detections."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False,
    )
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    xyxy_boxes = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_indices = boxes.cls.cpu().numpy().astype(int)

    detections = []
    for class_index, confidence, box in zip(class_indices, confidences, xyxy_boxes):
        class_index_int = int(class_index)
        detections.append(
            Detection(
                class_index=class_index_int,
                class_name=get_class_name(names, class_index_int),
                confidence=float(confidence),
                box=tuple(float(value) for value in box),
            )
        )
    return detections


def calculate_bbox_iou(
    first_box: tuple[float, float, float, float],
    second_box: tuple[float, float, float, float],
) -> float:
    """Calculate IoU for two bounding boxes in (x1, y1, x2, y2) format."""
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def match_detections_by_class_and_iou(
    pytorch_detections: list[Detection],
    onnx_detections: list[Detection],
    match_iou_threshold: float,
) -> tuple[list[DetectionPair], list[DetectionPair], list[int], list[int]]:
    """Greedily match same-class detections by descending bbox IoU."""
    candidate_pairs = []
    for pytorch_index, pytorch_detection in enumerate(pytorch_detections):
        for onnx_index, onnx_detection in enumerate(onnx_detections):
            if pytorch_detection.class_index != onnx_detection.class_index:
                continue
            candidate_pairs.append(
                DetectionPair(
                    pytorch_index=pytorch_index,
                    onnx_index=onnx_index,
                    iou=calculate_bbox_iou(pytorch_detection.box, onnx_detection.box),
                )
            )

    candidate_pairs.sort(key=lambda pair: pair.iou, reverse=True)

    matched_pairs = []
    matched_pytorch_indices = set()
    matched_onnx_indices = set()
    for pair in candidate_pairs:
        if pair.iou < match_iou_threshold:
            continue
        if (
            pair.pytorch_index in matched_pytorch_indices
            or pair.onnx_index in matched_onnx_indices
        ):
            continue
        matched_pairs.append(pair)
        matched_pytorch_indices.add(pair.pytorch_index)
        matched_onnx_indices.add(pair.onnx_index)

    reference_pairs = []
    referenced_pytorch_indices = set()
    referenced_onnx_indices = set()
    for pair in candidate_pairs:
        if pair.iou >= match_iou_threshold:
            continue
        if (
            pair.pytorch_index in matched_pytorch_indices
            or pair.onnx_index in matched_onnx_indices
            or pair.pytorch_index in referenced_pytorch_indices
            or pair.onnx_index in referenced_onnx_indices
        ):
            continue
        reference_pairs.append(pair)
        referenced_pytorch_indices.add(pair.pytorch_index)
        referenced_onnx_indices.add(pair.onnx_index)

    unmatched_pytorch_indices = [
        index
        for index in range(len(pytorch_detections))
        if index not in matched_pytorch_indices
    ]
    unmatched_onnx_indices = [
        index
        for index in range(len(onnx_detections))
        if index not in matched_onnx_indices
    ]
    return (
        matched_pairs,
        reference_pairs,
        unmatched_pytorch_indices,
        unmatched_onnx_indices,
    )


def format_pair(
    pair: DetectionPair,
    pytorch_detections: list[Detection],
    onnx_detections: list[Detection],
) -> str:
    """Format a matched or reference detection pair."""
    pytorch_detection = pytorch_detections[pair.pytorch_index]
    onnx_detection = onnx_detections[pair.onnx_index]
    confidence_difference = abs(
        pytorch_detection.confidence - onnx_detection.confidence
    )
    return (
        f"class_index={pytorch_detection.class_index} "
        f"class_name={pytorch_detection.class_name} "
        f"pytorch_confidence={pytorch_detection.confidence:.4f} "
        f"onnx_confidence={onnx_detection.confidence:.4f} "
        f"confidence_abs_diff={confidence_difference:.4f} "
        f"bbox_iou={pair.iou:.4f} "
        f"pytorch_bbox={format_box(pytorch_detection.box)} "
        f"onnx_bbox={format_box(onnx_detection.box)}"
    )


def format_box(box: tuple[float, float, float, float]) -> str:
    """Format a bbox tuple."""
    x1, y1, x2, y2 = box
    return f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"


def print_detections(title: str, detections: list[Detection]) -> None:
    """Print a named detection list."""
    print(f"{title}:")
    if not detections:
        print("  No detections")
        return

    for detection in detections:
        print(
            "  "
            + format_detection(
                detection.class_index,
                detection.class_name,
                detection.confidence,
                detection.box,
            )
        )


def print_comparison(
    pytorch_detections: list[Detection],
    onnx_detections: list[Detection],
    match_iou_threshold: float,
) -> None:
    """Print same-class PyTorch/ONNX matches and unmatched detections."""
    (
        matched_pairs,
        reference_pairs,
        unmatched_pytorch_indices,
        unmatched_onnx_indices,
    ) = match_detections_by_class_and_iou(
        pytorch_detections, onnx_detections, match_iou_threshold
    )

    print(
        "Matched PyTorch vs ONNX detections "
        f"(same class, bbox IoU >= {match_iou_threshold:.2f}):"
    )
    if not matched_pairs:
        print("  No matched detections")
    for pair in matched_pairs:
        print(f"  {format_pair(pair, pytorch_detections, onnx_detections)}")

    print(
        "Best below-threshold same-class pairs "
        "(reference only, not counted as matched):"
    )
    if not reference_pairs:
        print("  No below-threshold same-class pairs")
    for pair in reference_pairs:
        print(f"  {format_pair(pair, pytorch_detections, onnx_detections)}")

    print("Unmatched PyTorch detections:")
    if not unmatched_pytorch_indices:
        print("  No unmatched PyTorch detections")
    for index in unmatched_pytorch_indices:
        detection = pytorch_detections[index]
        print(
            "  "
            + format_detection(
                detection.class_index,
                detection.class_name,
                detection.confidence,
                detection.box,
            )
        )

    print("Unmatched ONNX detections:")
    if not unmatched_onnx_indices:
        print("  No unmatched ONNX detections")
    for index in unmatched_onnx_indices:
        detection = onnx_detections[index]
        print(
            "  "
            + format_detection(
                detection.class_index,
                detection.class_name,
                detection.confidence,
                detection.box,
            )
        )


def main() -> None:
    """Print PyTorch and ONNX detections for the same image."""
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {args.onnx_path}")

    pytorch_detections = collect_ultralytics_detections(
        "yolov8n.pt",
        args.image_path,
        args.conf_threshold,
        args.iou_threshold,
    )
    onnx_detections = collect_ultralytics_detections(
        args.onnx_path,
        args.image_path,
        args.conf_threshold,
        args.iou_threshold,
    )

    print(f"image path: {args.image_path}")
    print_detections("Ultralytics PyTorch detections", pytorch_detections)
    print_detections("Ultralytics ONNX detections", onnx_detections)
    print_comparison(
        pytorch_detections,
        onnx_detections,
        args.match_iou_threshold,
    )


if __name__ == "__main__":
    main()
