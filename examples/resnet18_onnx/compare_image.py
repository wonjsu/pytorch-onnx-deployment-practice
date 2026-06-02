"""Compare PyTorch and ONNX Runtime ResNet18 logits for a real image input."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


ONNX_PATH = Path(__file__).resolve().parent / "artifacts" / "resnet18.onnx"


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Compare raw ResNet18 logits from PyTorch and ONNX Runtime after "
            "torchvision ImageNet preprocessing of a real image."
        )
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an image file to preprocess and pass through both models.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=ONNX_PATH,
        help=f"Path to the exported ONNX model. Default: {ONNX_PATH}",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for np.allclose. Default: 1e-3",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for np.allclose. Default: 1e-5",
    )
    return parser.parse_args()


def preprocess_image(image_path: Path, weights: ResNet18_Weights) -> torch.Tensor:
    """Load an image and apply the preprocessing tied to the ResNet18 weights."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    preprocess = weights.transforms()
    return preprocess(rgb_image).unsqueeze(0)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute cosine similarity between two output tensors."""
    first_flat = first.reshape(-1).astype(np.float64)
    second_flat = second.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(first_flat) * np.linalg.norm(second_flat)
    if denominator == 0:
        return float("nan")
    return float(np.dot(first_flat, second_flat) / denominator)


def top_k_indices_and_probabilities(logits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return top-k class indices and softmax probabilities for auxiliary inspection."""
    logits_1d = logits.reshape(-1)
    shifted_logits = logits_1d - logits_1d.max()
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probabilities)[-k:][::-1]
    return [(int(index), float(probabilities[index])) for index in top_indices]


def main() -> None:
    """Run a real image through PyTorch and ONNX Runtime and compare raw logits."""
    args = parse_args()
    if not args.onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX file not found: {args.onnx_path}\n"
            "Run `python examples/resnet18_onnx/export_onnx.py` first."
        )

    weights = ResNet18_Weights.DEFAULT
    input_tensor = preprocess_image(args.image_path, weights)

    model = resnet18(weights=weights)
    model.eval()

    with torch.no_grad():
        torch_logits = model(input_tensor).detach().cpu().numpy()

    session = ort.InferenceSession(
        str(args.onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: input_tensor.cpu().numpy()})[0]

    abs_diff = np.abs(torch_logits - onnx_logits)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    logits_cosine_similarity = cosine_similarity(torch_logits, onnx_logits)
    is_allclose = np.allclose(torch_logits, onnx_logits, rtol=args.rtol, atol=args.atol)

    print(f"Image path: {args.image_path}")
    print(f"PyTorch logits shape: {torch_logits.shape}")
    print(f"ONNX Runtime logits shape: {onnx_logits.shape}")
    print("\nPrimary raw logits consistency checks")
    print(f"Max abs diff: {max_abs_diff:.8f}")
    print(f"Mean abs diff: {mean_abs_diff:.8f}")
    print(f"Cosine similarity: {logits_cosine_similarity:.8f}")
    print(f"np.allclose(rtol={args.rtol:.0e}, atol={args.atol:.0e}): {is_allclose}")

    print("\nAuxiliary top-5 check (class index, softmax probability)")
    print(f"PyTorch top-5: {top_k_indices_and_probabilities(torch_logits)}")
    print(f"ONNX Runtime top-5: {top_k_indices_and_probabilities(onnx_logits)}")


if __name__ == "__main__":
    main()
