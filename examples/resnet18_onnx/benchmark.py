"""Benchmark PyTorch ResNet18 and ONNX Runtime ResNet18 inference latency."""

from pathlib import Path
from time import perf_counter

import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights, resnet18


ONNX_PATH = Path(__file__).resolve().parent / "artifacts" / "resnet18.onnx"
DUMMY_INPUT_SHAPE = (1, 3, 224, 224)
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 100


def benchmark_pytorch(model: torch.nn.Module, dummy_input: torch.Tensor) -> float:
    """Return average PyTorch inference latency in milliseconds."""
    # warmup은 초기 실행 비용을 제외하고 더 안정적인 시간을 재기 위해 먼저 몇 번 추론합니다.
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(dummy_input)

        start = perf_counter()
        for _ in range(MEASUREMENT_RUNS):
            model(dummy_input)
        end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def benchmark_onnxruntime(session: ort.InferenceSession, dummy_input: torch.Tensor) -> float:
    """Return average ONNX Runtime inference latency in milliseconds."""
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.cpu().numpy()}

    # PyTorch와 같은 dummy input을 사용해 ONNX Runtime CPU 추론 시간을 측정합니다.
    for _ in range(WARMUP_RUNS):
        session.run(None, ort_inputs)

    start = perf_counter()
    for _ in range(MEASUREMENT_RUNS):
        session.run(None, ort_inputs)
    end = perf_counter()

    return (end - start) / MEASUREMENT_RUNS * 1000


def main() -> None:
    """Compare average inference latency for PyTorch and ONNX Runtime."""
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX file not found: {ONNX_PATH}\n"
            "Run `python examples/resnet18_onnx/export_onnx.py` first."
        )

    torch.manual_seed(0)
    dummy_input = torch.randn(DUMMY_INPUT_SHAPE)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    pytorch_latency_ms = benchmark_pytorch(model, dummy_input)
    onnxruntime_latency_ms = benchmark_onnxruntime(session, dummy_input)

    print(f"PyTorch latency(ms): {pytorch_latency_ms:.3f}")
    print(f"ONNX Runtime latency(ms): {onnxruntime_latency_ms:.3f}")


if __name__ == "__main__":
    main()
