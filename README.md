# pytorch-onnx-deployment-practice

PyTorch vision model을 ONNX로 export하고, ONNX Runtime/TensorRT에서 **inference consistency**와 **latency**를 비교하는 deployment workflow practice repository입니다.

이 프로젝트는 단순히 ONNX 파일을 생성하는 데서 끝나지 않고, 모델 종류에 따라 변환 후 확인해야 하는 지점이 어떻게 달라지는지 정리합니다. Classification 모델인 **ResNet18**과 detection 모델인 **YOLOv8n**을 함께 다루며, output consistency, preprocessing/postprocessing, end-to-end latency, TensorRT engine benchmark까지 단계적으로 확인합니다.

## 프로젝트 목적

- PyTorch vision model을 ONNX로 export하고, ONNX Runtime/TensorRT에서 inference consistency와 latency를 비교합니다.
- Classification 모델 ResNet18과 detection 모델 YOLOv8n의 변환 및 검증 흐름 차이를 비교합니다.
- Inference-only latency와 end-to-end latency를 분리해서 측정하고, 모델 추론 외의 image loading/preprocessing/postprocessing 비용이 결과에 미치는 영향을 확인합니다.
- 변환 자체보다 검증과 benchmark 과정을 중심으로, 실제 deployment workflow에서 확인해야 할 항목을 정리합니다.

## Repository 구성

| 경로 | 내용 |
|---|---|
| `examples/resnet18_onnx` | ResNet18 PyTorch → ONNX export, PyTorch vs ONNX Runtime output consistency 확인, inference-only/end-to-end latency benchmark |
| `examples/yolo_onnx` | YOLOv8n PyTorch → ONNX export, raw output shape 확인, letterbox preprocessing, bbox postprocessing, NMS, PyTorch vs ONNX detection comparison, latency benchmark, TensorRT FP32/FP16 benchmark |
| `requirements.txt` | 예제 실행에 필요한 Python dependency 목록 |

## 핵심 결과 요약

### ResNet18 classification

- PyTorch와 ONNX Runtime의 output shape이 모두 `(1, 1000)`으로 동일했습니다.
- Dummy input 및 실제 이미지 입력에서 `np.allclose(rtol=1e-03, atol=1e-05)` 기준 `True`로 확인되어, ONNX 변환 후 output consistency가 유지되었습니다.
- ONNX Runtime CPU는 inference-only 기준 PyTorch보다 빠르게 측정되었습니다.
- End-to-end latency에서는 이미지 로딩, preprocessing, softmax/top-k 후처리 비용이 포함되므로 inference-only 대비 개선 폭이 줄어들었습니다.

### YOLOv8n detection

- YOLOv8n ONNX raw output `(1, 84, 8400)`을 확인하고, letterbox preprocessing과 bbox decode/NMS 후처리를 직접 구성했습니다.
- 같은 이미지에 대해 PyTorch/Ultralytics 결과와 ONNX Runtime 결과를 class, confidence, bbox IoU 기준으로 비교했습니다.
- `test_mouse.jpg` 기준 bbox IoU는 `0.9959`로 측정되어, ONNX 변환 후 detection behavior가 잘 유지된 것으로 확인했습니다.
- CPU benchmark에서는 ONNX 변환만으로 Ultralytics end-to-end latency 개선이 뚜렷하지 않았습니다. Direct ONNX Runtime breakdown에서도 image loading과 letterbox preprocessing 비용이 큰 비중을 차지했습니다.

### TensorRT benchmark

- YOLOv8n ONNX 모델에서 TensorRT FP32/FP16 engine build와 benchmark를 수행했습니다.
- TensorRT GPU inference는 이전 ONNX Runtime CPU inference-only 측정값보다 훨씬 빠르게 측정되었지만, 이는 CPU backend와 GPU backend 비교라는 점을 함께 고려해야 합니다.
- NVIDIA GeForce GTX 1080 Ti에서는 FP16이 FP32보다 빨라지지 않았습니다.
- 이 결과는 GTX 1080 Ti가 Tensor Core가 없는 Pascal GPU라는 특성과 연결해 해석할 수 있으며, FP16 사용이 항상 latency 개선으로 이어지지는 않음을 보여줍니다.

## 실행 흐름

### 1. Dependency 설치

```bash
pip install -r requirements.txt
```

### 2. ResNet18 ONNX export 및 검증

```bash
python examples/resnet18_onnx/export_onnx.py
python examples/resnet18_onnx/compare_outputs.py
python examples/resnet18_onnx/compare_image.py assets/test_mouse.jpg
python examples/resnet18_onnx/benchmark_image.py assets/test_mouse.jpg
```

자세한 설명과 benchmark 결과는 `examples/resnet18_onnx/README.md`에 정리되어 있습니다.

### 3. YOLOv8n ONNX export, 후처리, 비교

```bash
python examples/yolo_onnx/export_onnx.py
python examples/yolo_onnx/infer_onnx.py assets/test_mouse.jpg
python examples/yolo_onnx/postprocess_onnx.py assets/test_mouse.jpg
python examples/yolo_onnx/compare_ultralytics_onnx.py assets/test_mouse.jpg
python examples/yolo_onnx/benchmark_yolo.py assets/test_mouse.jpg
```

YOLOv8n의 raw output 해석, letterbox preprocessing, NMS, PyTorch vs ONNX comparison, TensorRT benchmark는 `examples/yolo_onnx/README.md`에 정리되어 있습니다.

## Artifacts 및 local files

다음 파일들은 실행 과정에서 생성되거나 로컬 환경에 의존하는 파일이므로 GitHub repository에 포함하지 않습니다.

- `.onnx` 모델 파일
- TensorRT `.engine` 파일
- `yolov8n.pt` weight 파일
- 테스트 이미지 등 `assets/` local test files
- `examples/*/artifacts/` 아래 생성물

특히 TensorRT engine 파일은 GPU architecture, TensorRT version, driver/CUDA 환경에 영향을 받는 hardware/version-dependent artifact입니다. 따라서 engine binary를 공유하기보다 build command와 benchmark 결과를 문서화합니다.

## 해석 시 주의점

- ONNX export는 모델을 다른 runtime에서 실행할 수 있게 하는 중간 포맷 변환이며, 그 자체가 latency 개선을 보장하지는 않습니다.
- Inference-only benchmark와 end-to-end benchmark는 측정 범위가 다릅니다. Deployment workflow에서는 모델 forward 시간뿐 아니라 image loading, preprocessing, postprocessing, NMS 비용도 함께 확인해야 합니다.
- CPU backend와 GPU backend 결과는 같은 기준의 숫자로 단순 비교하기 어렵습니다. TensorRT 결과는 ONNX Runtime CPU 결과보다 빠르지만, backend와 hardware가 다르다는 점을 명시적으로 구분해야 합니다.
- INT8 TensorRT 실험은 calibration dataset과 별도 accuracy validation이 필요하므로 이번 practice 범위에서는 제외했습니다.
