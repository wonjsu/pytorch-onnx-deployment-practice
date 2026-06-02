"""Export torchvision ResNet18 to ONNX and validate the exported graph."""

from pathlib import Path

import onnx
import torch
from torchvision.models import ResNet18_Weights, resnet18


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ONNX_PATH = ARTIFACT_DIR / "resnet18.onnx"
DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


def main() -> None:
    """Export ResNet18 with a fixed dummy input shape."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # torchvision에서 제공하는 ImageNet 사전학습 ResNet18 가중치를 사용합니다.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # eval 모드는 BatchNorm/Dropout처럼 학습과 추론 동작이 다른 레이어를 추론용으로 고정합니다.
    model.eval()

    # ONNX export는 예시 입력(dummy input)을 따라 모델 그래프와 입출력 shape을 기록합니다.
    dummy_input = torch.randn(DUMMY_INPUT_SHAPE)

    # no_grad는 추론/내보내기 과정에서 불필요한 gradient 계산을 막아 메모리를 절약합니다.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
        )

    # 저장된 ONNX 파일을 다시 읽어 ONNX 표준을 만족하는 그래프인지 검사합니다.
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    print(f"Exported ONNX model: {ONNX_PATH}")
    print("ONNX checker validation: passed")


if __name__ == "__main__":
    main()
