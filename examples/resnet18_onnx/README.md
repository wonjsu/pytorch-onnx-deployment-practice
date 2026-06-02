# ResNet18 ONNX Export 예제

이 예제는 `torchvision`의 ImageNet 사전학습 ResNet18 모델을 ONNX 파일로 내보내고, 같은 dummy input을 PyTorch와 ONNX Runtime에 넣어 출력이 비슷한지 확인합니다.

## 파일 구성

- `export_onnx.py`: PyTorch ResNet18 모델을 `artifacts/resnet18.onnx`로 export하고 `onnx.checker.check_model`로 검증합니다.
- `compare_outputs.py`: PyTorch 출력과 ONNX Runtime CPU 출력의 shape 및 오차를 비교합니다.
- `compare_image.py`: 실제 이미지 전처리 후 PyTorch와 ONNX Runtime의 raw output(logits)이 수치적으로 일관되는지 비교합니다. top-5 class index/probability는 보조 확인용으로만 출력합니다.
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


## 3. 실제 이미지 입력에서 PyTorch vs ONNX Runtime logits 일관성 확인

ONNX 파일을 만든 뒤 실제 이미지 파일 경로를 인자로 전달합니다.

```bash
python examples/resnet18_onnx/compare_image.py path/to/image.jpg
```

이 예제의 목적은 ImageNet 분류 정확도 평가가 아닙니다. class label 이름을 붙여 예측이 맞는지 평가하는 대신, `torchvision`의 ImageNet 전처리를 실제 이미지에 적용한 뒤 PyTorch와 ONNX Runtime이 내는 raw output(logits)이 수치적으로 일관되는지 확인합니다. 따라서 주요 출력은 `max abs diff`, `mean abs diff`, `cosine similarity`, `np.allclose`입니다.

출력 예시는 다음과 같습니다.

```text
Image path: path/to/image.jpg
PyTorch logits shape: (1, 1000)
ONNX Runtime logits shape: (1, 1000)

Primary raw logits consistency checks
Max abs diff: 0.0000xxxx
Mean abs diff: 0.0000xxxx
Cosine similarity: 1.0000xxxx
np.allclose(rtol=1e-03, atol=1e-05): True

Auxiliary top-5 check (class index, softmax probability)
PyTorch top-5: [(..., 0.xxxx), ...]
ONNX Runtime top-5: [(..., 0.xxxx), ...]
```

top-5 class index/probability는 두 런타임의 결과가 직관적으로 비슷한지 보는 보조 확인용입니다. 이 스크립트는 class label 이름을 추가하지 않습니다.

### 실제 이미지 입력 비교 결과

아래 결과는 ImageNet 분류 정확도 평가가 아니라, ONNX 변환 후 실제 이미지 전처리까지 포함했을 때 PyTorch ResNet18과 ONNX Runtime ResNet18의 raw output(logits)이 수치적으로 일관되는지 확인하기 위한 실험입니다. top-5 class index/probability는 주요 평가 지표가 아니며, 두 런타임 출력이 직관적으로 같은 방향인지 살펴보는 보조 확인용입니다.

저장소 루트에서 다음 명령으로 각 이미지를 비교했습니다.

```bash
python examples/resnet18_onnx/compare_image.py assets/test_mouse.jpg
python examples/resnet18_onnx/compare_image.py assets/test_keyboard.jpg
python examples/resnet18_onnx/compare_image.py assets/test_headphone.jpg
```

| Image | Max Abs Diff | Mean Abs Diff | Cosine Similarity | allclose |
|---|---:|---:|---:|---|
| test_mouse.jpg | 0.00001335 | 0.00000243 | 1.00000000 | True |
| test_keyboard.jpg | 0.00000811 | 0.00000175 | 1.00000000 | True |
| test_headphone.jpg | 0.00000906 | 0.00000202 | 1.00000000 | True |

모든 이미지에서 output shape은 PyTorch와 ONNX Runtime 모두 `(1, 1000)`으로 동일했습니다. 세 이미지 모두 `np.allclose(rtol=1e-03, atol=1e-05)` 기준에서 `True`였으며, `max abs diff`는 약 `1e-5` 수준, `mean abs diff`는 약 `1e-6` 수준으로 측정되었습니다. 따라서 실제 이미지 입력에서도 ONNX Runtime 출력이 PyTorch 출력과 설정한 허용 오차 범위 내에서 수치적으로 일관됨을 확인했습니다. 또한 top-5 class index도 세 이미지에서 PyTorch와 ONNX Runtime이 동일한 순서로 나와, raw logits 비교 결과를 보조적으로 뒷받침했습니다.

## 4. PyTorch vs ONNX Runtime 추론 시간 benchmark

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
