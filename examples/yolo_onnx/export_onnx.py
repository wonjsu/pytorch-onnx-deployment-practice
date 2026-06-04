"""Export Ultralytics YOLOv8n to ONNX."""

import os
from pathlib import Path

from ultralytics import YOLO


EXAMPLE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXAMPLE_DIR / "artifacts"
ONNX_PATH = ARTIFACT_DIR / "yolov8n.onnx"
MODEL_NAME = "yolov8n.pt"


def main() -> None:
    """Export YOLOv8n to ONNX inside the example artifacts directory."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    original_cwd = Path.cwd()

    try:
        os.chdir(ARTIFACT_DIR)
        model.export(format="onnx", imgsz=640, opset=17)
    finally:
        os.chdir(original_cwd)

    target_path = ONNX_PATH.resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"ONNX export failed: {target_path} was not created")

    print(f"Exported ONNX model: {target_path}")


if __name__ == "__main__":
    main()
