# pytorch-onnx-deployment-practice

PyTorch 기반 vision model을 ONNX로 변환한 뒤, ONNX Runtime과 TensorRT에서 출력 일관성과 추론 시간을 비교하는 실습 repository입니다.

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
| `examples/yolo_tensorrt` | TensorRT Python API 실제 이미지 추론, PyTorch/ORT/TensorRT output 검증, 구간별 benchmark |
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

예제 명령의 `assets/test_mouse.jpg`는 로컬 테스트 이미지 예시이며, 사용자는 자신의 이미지 경로로 바꿔 실행할 수 있습니다.

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

### 4. YOLOv8n TensorRT 실제 추론 및 검증

TensorRT는 target NVIDIA 환경에 맞춰 별도로 설치해야 하며 버전을 임의로 고정하지 않습니다.

```bash
pip install -r requirements-tensorrt.txt
python examples/yolo_tensorrt/environment_report.py
python examples/yolo_tensorrt/infer_tensorrt.py assets/test_mouse.jpg --engine-path examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine
python examples/yolo_tensorrt/compare_backends.py assets/test_mouse.jpg --onnx-path examples/yolo_onnx/artifacts/yolov8n.onnx --engine-path examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine
python examples/yolo_tensorrt/benchmark_tensorrt.py assets/test_mouse.jpg --engine-path examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine --warmup 10 --runs 100 --json benchmark-results/result.json --csv benchmark-results/result.csv
```

세 backend 비교는 Ultralytics의 고수준 `predict()`를 호출하지 않습니다. 한 번 직접 만든 letterbox NCHW tensor를 PyTorch raw model, ONNX Runtime CUDA provider, TensorRT에 그대로 입력하고, 세 raw output 모두 기존 `postprocess_output`으로 처리합니다.

#### Engine build와 layer profiling

```bash
# FP32 (TensorRT 기본값은 TF32를 허용할 수 있음)
trtexec --onnx=examples/yolo_onnx/artifacts/yolov8n.onnx --saveEngine=examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine
# FP16
trtexec --onnx=examples/yolo_onnx/artifacts/yolov8n.onnx --fp16 --saveEngine=examples/yolo_tensorrt/artifacts/yolov8n-fp16.engine
# 상세 layer 정보 JSON (TensorRT 8.x/10.x)
trtexec --loadEngine=examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=benchmark-results/layers.json
# per-layer 실행 profile 및 JSON
trtexec --loadEngine=examples/yolo_tensorrt/artifacts/yolov8n-fp32.engine --profilingVerbosity=detailed --dumpProfile --exportProfile=benchmark-results/profile.json
```

`--profilingVerbosity=detailed`, `--dumpLayerInfo`, `--exportLayerInfo`, `--dumpProfile`, `--exportProfile`의 지원 여부와 철자는 설치된 `trtexec --help`에서 확인해야 합니다. 구형 TensorRT 8.x 배포판 일부에는 JSON export 옵션이 없으므로 `--dumpLayerInfo`/`--dumpProfile` 표준 출력만 저장하십시오. TensorRT 10.x에서는 위 옵션을 사용할 수 있습니다.

TensorRT의 “FP32” build는 NVIDIA에서 허용한 TF32 tactic을 선택할 수 있으므로 strict IEEE FP32 baseline과 같지 않을 수 있습니다. **먼저 `environment_report.py`로 실제 TensorRT 버전을 확인**하십시오. TensorRT 8.x의 `trtexec`에서는 `--noTF32`로 strict baseline을 만들 수 있습니다. TensorRT 10.x에서는 배포판별 CLI 변경 가능성이 있으므로 `trtexec --help | grep -i tf32`로 지원 옵션을 확인하고, 옵션이 없으면 Python builder의 `BuilderFlag.TF32`를 clear한 별도 build가 필요합니다. 이 저장소의 현재 CPU 환경에는 TensorRT가 없어 특정 target 버전용 명령을 단정하지 않습니다.

#### 결과 기록 template (측정값을 직접 입력)

| GPU / TRT | Precision | TF32 | Engine-only mean/p95 (ms) | E2E mean/p95 (ms) | Raw MAE/max | Matched/unmatched |
|---|---|---|---|---|---|---|
| RTX 3060 Laptop / ___ | FP32 | on/off | ___ / ___ | ___ / ___ | ___ / ___ | ___ / ___ |
| RTX 3060 Laptop / ___ | FP16 | n/a | ___ / ___ | ___ / ___ | ___ / ___ | ___ / ___ |

`benchmark_tensorrt.py`는 CUDA event로 H2D, enqueue/GPU compute, D2H를 재고 `perf_counter`로 CPU 구간과 end-to-end를 잽니다. `loaded_image_reuse`와 매 iteration 파일을 여는 `reopen_each_iteration` 결과를 분리합니다. 이는 backend consistency 검증이며 ground-truth dataset mAP를 대신하지 않습니다.

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
