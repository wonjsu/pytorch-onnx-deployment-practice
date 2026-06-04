"""Export Ultralytics YOLOv8n to ONNX."""

from pathlib import Path
from shutil import move

from ultralytics import YOLO


EXAMPLE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXAMPLE_DIR / "artifacts"
ONNX_PATH = ARTIFACT_DIR / "yolov8n.onnx"
MODEL_NAME = "yolov8n.pt"


def main() -> None:
    """Export YOLOv8n to ONNX and place the artifact in the example directory."""
    model = YOLO(MODEL_NAME)

    exported_path = Path(model.export(format="onnx")).resolve()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ONNX_PATH.resolve()

    if exported_path != target_path:
        if target_path.exists():
            target_path.unlink()
        move(str(exported_path), target_path)

    if not target_path.exists():
        raise FileNotFoundError(f"ONNX export failed: {target_path} was not created")

    print(f"Exported ONNX model: {target_path}")


if __name__ == "__main__":
    main()
