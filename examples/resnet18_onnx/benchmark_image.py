"""Benchmark PyTorch and ONNX Runtime ResNet18 latency for a real image input."""

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


ONNX_PATH = Path(__file__).resolve().parent / "artifacts" / "resnet18.onnx"
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 100
TOP_K = 5


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Benchmark PyTorch and ONNX Runtime ResNet18 latency with a real "
            "image input and torchvision ImageNet preprocessing."
        )
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an image file to benchmark.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=ONNX_PATH,
        help=f"Path to the exported ONNX model. Default: {ONNX_PATH}",
    )
    return parser.parse_args()


def preprocess_image(
    image_path: Path,
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """Load an image and apply the preprocessing tied to the ResNet18 weights."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    return preprocess(rgb_image).unsqueeze(0)


def torch_top_k(logits: torch.Tensor, k: int = TOP_K) -> list[tuple[int, float]]:
    """Return top-k class indices and softmax probabilities from PyTorch logits."""
    probabilities = torch.softmax(logits, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=1)
    return [
        (int(index), float(probability))
        for index, probability in zip(top_indices[0], top_probabilities[0])
    ]


def numpy_top_k(logits: np.ndarray, k: int = TOP_K) -> list[tuple[int, float]]:
    """Return top-k class indices and softmax probabilities from ONNX logits."""
    logits_1d = logits.reshape(-1)
    shifted_logits = logits_1d - logits_1d.max()
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probabilities)[-k:][::-1]
    return [(int(index), float(probabilities[index])) for index in top_indices]


def benchmark_pytorch_inference_only(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> float:
    """Return average PyTorch inference-only latency in milliseconds."""
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)

        start = perf_counter()
        for _ in range(MEASUREMENT_RUNS):
            model(input_tensor)
        end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def benchmark_onnxruntime_inference_only(
    session: ort.InferenceSession,
    input_array: np.ndarray,
) -> float:
    """Return average ONNX Runtime inference-only latency in milliseconds."""
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: input_array}

    for _ in range(WARMUP_RUNS):
        session.run(None, ort_inputs)

    start = perf_counter()
    for _ in range(MEASUREMENT_RUNS):
        session.run(None, ort_inputs)
    end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def benchmark_pytorch_end_to_end(
    model: torch.nn.Module,
    image_path: Path,
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> float:
    """Return average PyTorch load/preprocess/inference/postprocess latency in ms."""
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            input_tensor = preprocess_image(image_path, preprocess)
            logits = model(input_tensor)
            torch_top_k(logits)

        start = perf_counter()
        for _ in range(MEASUREMENT_RUNS):
            input_tensor = preprocess_image(image_path, preprocess)
            logits = model(input_tensor)
            torch_top_k(logits)
        end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def benchmark_onnxruntime_end_to_end(
    session: ort.InferenceSession,
    image_path: Path,
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> float:
    """Return average ONNX Runtime load/preprocess/inference/postprocess latency in ms."""
    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        input_tensor = preprocess_image(image_path, preprocess)
        logits = session.run(None, {input_name: input_tensor.cpu().numpy()})[0]
        numpy_top_k(logits)

    start = perf_counter()
    for _ in range(MEASUREMENT_RUNS):
        input_tensor = preprocess_image(image_path, preprocess)
        logits = session.run(None, {input_name: input_tensor.cpu().numpy()})[0]
        numpy_top_k(logits)
    end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def main() -> None:
    """Benchmark real-image ResNet18 latency for PyTorch and ONNX Runtime."""
    args = parse_args()
    if not args.onnx_path.exists():
        print(
            f"ONNX file not found: {args.onnx_path}\n"
            "Run `python examples/resnet18_onnx/export_onnx.py` first."
        )
        raise SystemExit(1)

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    input_tensor = preprocess_image(args.image_path, preprocess)
    input_array = input_tensor.cpu().numpy()

    model = resnet18(weights=weights)
    model.eval()

    session = ort.InferenceSession(
        str(args.onnx_path),
        providers=["CPUExecutionProvider"],
    )

    pytorch_inference_only_latency_ms = benchmark_pytorch_inference_only(
        model,
        input_tensor,
    )
    onnxruntime_inference_only_latency_ms = benchmark_onnxruntime_inference_only(
        session,
        input_array,
    )
    pytorch_end_to_end_latency_ms = benchmark_pytorch_end_to_end(
        model,
        args.image_path,
        preprocess,
    )
    onnxruntime_end_to_end_latency_ms = benchmark_onnxruntime_end_to_end(
        session,
        args.image_path,
        preprocess,
    )

    print(f"Image path: {args.image_path}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Measurement runs: {MEASUREMENT_RUNS}")
    print(
        "PyTorch inference-only latency(ms): "
        f"{pytorch_inference_only_latency_ms:.3f}"
    )
    print(
        "ONNX Runtime inference-only latency(ms): "
        f"{onnxruntime_inference_only_latency_ms:.3f}"
    )
    print(f"PyTorch end-to-end latency(ms): {pytorch_end_to_end_latency_ms:.3f}")
    print(
        "ONNX Runtime end-to-end latency(ms): "
        f"{onnxruntime_end_to_end_latency_ms:.3f}"
    )


if __name__ == "__main__":
    main()
