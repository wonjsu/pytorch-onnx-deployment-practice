"""Compare Ultralytics YOLOv8n PyTorch results with ONNX Runtime postprocessing."""

import argparse
from pathlib import Path

import onnxruntime as ort

from postprocess_onnx import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_ONNX_PATH,
    get_coco_class_name,
    letterbox_preprocess_image,
    postprocess_output,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Ultralytics YOLOv8n PyTorch detections with ONNX Runtime "
            "detections from the same image."
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


def format_detection(
    class_index: int,
    class_name: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> str:
    """Format one detection for side-by-side visual comparison."""
    x1, y1, x2, y2 = box
    return (
        f"class_index={class_index} "
        f"class_name={class_name} "
        f"confidence={confidence:.4f} "
        f"bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
    )


def print_ultralytics_detections(
    image_path: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> None:
    """Run Ultralytics YOLOv8n PyTorch inference and print detections."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model(
        str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False,
    )
    result = results[0]
    boxes = result.boxes

    print("Ultralytics detections:")
    if boxes is None or len(boxes) == 0:
        print("  No detections")
        return

    xyxy_boxes = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_indices = boxes.cls.cpu().numpy().astype(int)
    for class_index, confidence, box in zip(class_indices, confidences, xyxy_boxes):
        class_name = get_coco_class_name(class_index)
        print(
            "  "
            + format_detection(
                class_index,
                class_name,
                float(confidence),
                tuple(float(value) for value in box),
            )
        )


def print_onnx_detections(
    image_path: Path,
    onnx_path: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> None:
    """Run ONNX Runtime inference with local postprocessing and print detections."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    input_tensor, original_size, scale_ratio, pad_x, pad_y = letterbox_preprocess_image(
        image_path
    )
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
        scale_ratio,
        pad_x,
        pad_y,
        conf_threshold,
        iou_threshold,
    )

    print("ONNX Runtime detections:")
    if len(boxes) == 0:
        print("  No detections")
        return

    for class_index, confidence, box in zip(class_indices, confidences, boxes):
        class_index_int = int(class_index)
        print(
            "  "
            + format_detection(
                class_index_int,
                get_coco_class_name(class_index_int),
                float(confidence),
                tuple(float(value) for value in box),
            )
        )


def main() -> None:
    """Print Ultralytics and ONNX Runtime detections for the same image."""
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    print(f"image path: {args.image_path}")
    print_ultralytics_detections(
        args.image_path,
        args.conf_threshold,
        args.iou_threshold,
    )
    print_onnx_detections(
        args.image_path,
        args.onnx_path,
        args.conf_threshold,
        args.iou_threshold,
    )


if __name__ == "__main__":
    main()
