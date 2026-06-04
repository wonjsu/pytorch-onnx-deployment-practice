"""Export Ultralytics YOLOv8n to ONNX."""

from pathlib import Path
from shutil import move

from ultralytics import YOLO


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
ARTIFACT_DIR = EXAMPLE_DIR / "artifacts"
ONNX_PATH = ARTIFACT_DIR / "yolov8n.onnx"
MODEL_NAME = "yolov8n.pt"


def main() -> None:
    """Export YOLOv8n to ONNX and place the artifact in the example directory."""
    model = YOLO(MODEL_NAME)

    exported_path = Path(model.export(format="onnx", imgsz=640, opset=17)).resolve()
    if not exported_path.exists():
        exported_path = REPO_ROOT / "yolov8n.onnx"

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
