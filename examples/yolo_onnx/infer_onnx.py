"""Run YOLOv8n ONNX inference and print raw output shapes."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = EXAMPLE_DIR / "artifacts" / "yolov8n.onnx"
INPUT_SIZE = (640, 640)
PADDING_VALUE = 114


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n ONNX inference and print raw output shapes."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an input image for YOLOv8n ONNX inference.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=f"Path to the YOLOv8n ONNX model. Default: {DEFAULT_ONNX_PATH}",
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


def main() -> None:
    """Run ONNX Runtime inference and print input/output tensor shapes."""
    args = parse_args()

    onnx_path = args.onnx_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    input_tensor, _, _, _, _ = letterbox_preprocess_image(args.image_path)
    print(f"Input tensor shape: {input_tensor.shape}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    print(f"ONNX output count: {len(outputs)}")
    for index, output in enumerate(outputs):
        print(f"Output {index} shape: {output.shape}")


if __name__ == "__main__":
    main()
