# ResNet18 ONNX Export 예제

이 예제는 `torchvision`의 ImageNet 사전학습 ResNet18 모델을 ONNX 파일로 내보내고, 같은 dummy input을 PyTorch와 ONNX Runtime에 넣어 출력이 비슷한지 확인합니다.

## 파일 구성

- `export_onnx.py`: PyTorch ResNet18 모델을 `artifacts/resnet18.onnx`로 export하고 `onnx.checker.check_model`로 검증합니다.
- `compare_outputs.py`: PyTorch 출력과 ONNX Runtime CPU 출력의 shape 및 오차를 비교합니다.
- `benchmark.py`: PyTorch ResNet18과 ONNX Runtime ResNet18의 CPU 평균 추론 시간을 비교합니다.
- `artifacts/`: export된 ONNX 파일이 저장되는 디렉터리입니다.

## 준비

저장소 루트에서 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

> `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`를 사용하므로, 처음 실행할 때 ResNet18 사전학습 가중치를 다운로드할 수 있습니다.

## 1. ONNX export

저장소 루트에서 다음 명령을 실행합니다.

```bash
python examples/resnet18_onnx/export_onnx.py
```

스크립트가 수행하는 핵심 작업은 다음과 같습니다.

1. `ResNet18_Weights.DEFAULT` 가중치로 ResNet18 모델을 생성합니다.
2. `model.eval()`로 추론 모드로 전환합니다.
3. `[1, 3, 224, 224]` shape의 dummy input을 만듭니다.
4. `torch.onnx.export(..., opset_version=17)`로 ONNX 파일을 저장합니다.
5. `onnx.checker.check_model`로 export 결과를 검증합니다.

생성되는 파일 경로:

```text
examples/resnet18_onnx/artifacts/resnet18.onnx
```

## 2. PyTorch vs ONNX Runtime 출력 비교

ONNX 파일을 만든 뒤 다음 명령을 실행합니다.

```bash
python examples/resnet18_onnx/compare_outputs.py
```

출력 예시는 다음과 같습니다.

```text
PyTorch output shape: (1, 1000)
ONNX Runtime output shape: (1, 1000)
Max abs diff: 0.0000xxxx
Mean abs diff: 0.0000xxxx
np.allclose(rtol=1e-03, atol=1e-05): True
```

`max abs diff`와 `mean abs diff`는 두 출력 사이의 절대 오차를 나타냅니다. ONNX 변환 과정에서 부동소수점 연산 순서가 조금 달라질 수 있으므로 아주 작은 차이는 정상입니다.

## 3. PyTorch vs ONNX Runtime 추론 시간 benchmark

ONNX 파일을 만든 뒤 다음 명령을 실행합니다.

```bash
python examples/resnet18_onnx/benchmark.py
```

`benchmark.py`는 `[1, 3, 224, 224]` shape의 같은 dummy input으로 warmup 10회, measurement 100회를 실행한 뒤 평균 latency를 ms 단위로 출력합니다. ONNX Runtime은 `CPUExecutionProvider` 기준으로 측정합니다.

출력 예시는 다음과 같습니다.

```text
PyTorch latency(ms): 12.345
ONNX Runtime latency(ms): 8.901
```

실행 환경에 따라 latency는 달라질 수 있으므로, 측정한 결과를 아래 표에 기록해 비교해 볼 수 있습니다.

| Date | CPU / Machine | PyTorch latency(ms) | ONNX Runtime latency(ms) | Notes |
| --- | --- | ---: | ---: | --- |
| | | | | |
