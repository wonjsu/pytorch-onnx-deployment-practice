"""Compare PyTorch ResNet18 output with ONNX Runtime output."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights, resnet18


ONNX_PATH = Path(__file__).resolve().parent / "artifacts" / "resnet18.onnx"
DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


def main() -> None:
    """Run the same dummy input through PyTorch and ONNX Runtime."""
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX file not found: {ONNX_PATH}\n"
            "Run `python examples/resnet18_onnx/export_onnx.py` first."
        )

    # PyTorch와 ONNX Runtime에 완전히 같은 값을 넣기 위해 seed를 고정합니다.
    torch.manual_seed(0)
    dummy_input = torch.randn(DUMMY_INPUT_SHAPE)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # PyTorch 기준 출력(logits)을 계산합니다.
    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    # CPUExecutionProvider를 명시해서 CPU 기반 ONNX Runtime 추론을 수행합니다.
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]

    abs_diff = np.abs(torch_output - onnx_output)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    is_allclose = np.allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-05)

    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"ONNX Runtime output shape: {onnx_output.shape}")
    print(f"Max abs diff: {max_abs_diff:.8f}")
    print(f"Mean abs diff: {mean_abs_diff:.8f}")
    print(f"np.allclose(rtol=1e-03, atol=1e-05): {is_allclose}")


if __name__ == "__main__":
    main()
