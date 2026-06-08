"""Compare Ultralytics YOLOv8n PyTorch and exported ONNX detections."""

import argparse
from pathlib import Path


DEFAULT_ONNX_PATH = Path("examples/yolo_onnx/artifacts/yolov8n.onnx")
DEFAULT_CONF_THRESHOLD = 0.05
DEFAULT_IOU_THRESHOLD = 0.45


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
    return parser.parse_args()


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
    """Return a class name from an Ultralytics result names mapping."""
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
) -> list[str]:
    """Run Ultralytics prediction and return formatted detections."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
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
            format_detection(
                class_index_int,
                get_class_name(names, class_index_int),
                float(confidence),
                tuple(float(value) for value in box),
            )
        )
    return detections


def print_detections(title: str, detections: list[str]) -> None:
    """Print a named detection list."""
    print(f"{title}:")
    if not detections:
        print("  No detections")
        return

    for detection in detections:
        print(f"  {detection}")


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


if __name__ == "__main__":
    main()
