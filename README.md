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

### 4. YOLOv8n TensorRT 11.1 실제 추론 및 검증

이 repository의 현재 TensorRT 구현은 아래의 **TensorRT 11.1 FP32 전용 workflow**를 사용합니다. 이전 TensorRT 세대의 binding API나 FP16/INT8 engine은 이 작업 범위에 포함하지 않습니다.

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

## TensorRT 11.1 FP32 전용 workflow

이 구현은 **TensorRT 11.1.0.106만** 대상으로 하며, FP32 ONNX의 정적 입력 `(1, 3, 640, 640)`만 지원합니다. TensorRT 11의 기본 strongly typed network를 그대로 생성하고 FP16, INT8, calibration, Q/DQ 및 Model Optimizer는 사용하지 않습니다. `strict FP32` build는 `BuilderFlag.TF32`를 명시적으로 clear하고, `FP32 with TF32 allowed` build는 FP32 tensor dtype을 유지한 채 해당 flag를 set하여 TF32 tactic 선택만 허용합니다.

TensorRT engine은 GPU, TensorRT, CUDA/driver 환경에 의존하는 binary입니다. 아래 engine은 **실행 대상 RTX 3060 Laptop GPU에서 직접 생성**해야 하고 Git에 commit하지 않습니다. 다른 GPU 또는 TensorRT 버전에서는 재빌드가 필요할 수 있습니다. 각 build는 ONNX SHA-256, 상대 artifact 경로, 환경, TF32 설정 및 I/O metadata를 `<engine>.json`에 함께 기록합니다.

Windows CMD에서 repository root와 활성화된 Python 3.11 virtual environment를 기준으로 실행합니다.

```bat
REM TensorRT 11.1.0.106 target dependencies
pip install -r requirements-tensorrt.txt

REM strict FP32: TF32 explicitly disabled
python examples\yolo_tensorrt\build_engine.py --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine --workspace-gb 1 --tf32 off

REM FP32 tensors with TF32 tactics allowed
python examples\yolo_tensorrt\build_engine.py --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_tf32.engine --workspace-gb 1 --tf32 on

REM one-image ONNX Runtime CUDA versus strict TensorRT comparison
python examples\yolo_tensorrt\compare_onnx_tensorrt.py --image-path assets\test_mouse.jpg --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine

REM strict FP32 COCO smoke evaluation (10 images)
python examples\yolo_coco\evaluate_coco.py --backend tensorrt --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine --limit 10 --output-json benchmark-results\coco_strict_10_predictions.json

REM strict FP32 full COCO val2017 evaluation (5,000 images)
python examples\yolo_coco\evaluate_coco.py --backend tensorrt --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine --limit 5000 --output-json benchmark-results\coco_strict_5000_predictions.json

REM TF32-allowed FP32 full COCO val2017 evaluation (5,000 images)
python examples\yolo_coco\evaluate_coco.py --backend tensorrt --engine-path examples\yolo_tensorrt\artifacts\yolov8n_fp32_tf32.engine --limit 5000 --output-json benchmark-results\coco_tf32_5000_predictions.json
```

TensorRT 평가의 초기화(엔진 deserialize/context 생성)와 3회 warm-up은 평균에서 제외됩니다. CPU preprocessing, H2D, TensorRT compute, D2H, CPU postprocessing은 별도로 출력되며 GPU 세 구간은 PyTorch CUDA event로 측정합니다. H2D는 공유 NumPy FP32 전처리 결과를 재사용 CUDA input tensor로 복사하는 구간이고, D2H는 재사용 CUDA output tensor를 pinned CPU tensor로 복사하는 구간입니다.

### 실제 결과 기록 template

측정 전에는 숫자를 채우지 않습니다.

| Engine | GPU / TensorRT | TF32 | Engine size | AP 0.50:0.95 | AP 0.50 | Avg total/image | H2D | Compute | D2H |
|---|---|---|---|---|---|---|---|---|---|
| `yolov8n_fp32_strict.engine` | ___ | disabled | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| `yolov8n_fp32_tf32.engine` | ___ | allowed | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

## TensorRT precision latency benchmark

정확도와 latency는 분리해 평가합니다. 정확도는 기존
`examples\yolo_coco\evaluate_coco.py`로 precision별 COCO 5,000장 full evaluation을
한 번 수행하며(accuracy threshold `0.001/0.7`), latency 비교는 COCOeval/AP/예측 JSON을
만들지 않는 `examples\yolo_benchmark\benchmark_precision.py`를 사용합니다. Latency의
confidence/NMS threshold 기본값은 `0.25/0.45`입니다.

Benchmark는 기본 4라운드 중 첫 1라운드를 warm-up 및 파일 캐시 형성 라운드로 raw
결과에는 남기되 aggregate에서 제외합니다. 나머지 3라운드의 모든 sample을 합친
mean/median/P95/표준편차와 라운드별 평균의 표준편차를 보고합니다. 보통 총 라운드는
3~5 범위를 사용하십시오. 여러 engine은 매 라운드 `fp32 → fp16 → int8`, `fp16 →
int8 → fp32`, `int8 → fp32 → fp16` 순으로 회전합니다.

`engine` mode는 실제 COCO 이미지 한 장을 한 번 letterbox한 고정 입력으로 persistent
buffer, non-default stream, 재사용 CUDA event의 H2D/compute/D2H를 측정합니다.
`pipeline` mode는 매 이미지의 load/decode, CPU letterbox, H2D, compute, D2H, 실제
confidence filtering/좌표 복원/class-aware NMS, 기타 Python overhead를 측정합니다.
주 비교 지표는 파일 I/O를 제외한 pipeline이며 full end-to-end는 파일 캐시 영향을 받는
보조 지표입니다.

전체 5,000장 측정 전에 아래처럼 `--limit 100 --pipeline-rounds 2
--discard-rounds 1`로 기능만 확인하십시오. 이 단축 실행 결과는 최종 성능 결과로
사용하지 않습니다. 실제 latency 수치는 RTX 3060 Laptop GPU에서 사용자가 실행한 뒤
기록합니다.

```bat
python examples\yolo_benchmark\benchmark_precision.py --mode pipeline --engine fp32=examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine --limit 100 --pipeline-rounds 2 --discard-rounds 1 --output-json benchmark-results\fp32_smoke.json --output-csv benchmark-results\fp32_smoke.csv
```

## TensorRT 11.1 mixed-FP16 workflow (Windows CMD)

The verified strict-FP32 baseline and empty FP16/INT8 comparison columns are recorded in [`docs/precision_benchmark_results.md`](docs/precision_benchmark_results.md). TensorRT 11.1 strongly typed networks do **not** use `BuilderFlag.FP16`: ModelOpt AutoCast first writes precision into a mixed FP32/FP16 ONNX graph, with FP32 external I/O retained. The builder's `--model-precision` is a validation/metadata label, not a precision flag. INT8 is not implemented.

Keep ModelOpt isolated because installing it can alter ONNX/ORT packages in the working TensorRT environment. AutoCast defaults to CPU provider; this workflow makes no claim that the RTX 3060 is within ModelOpt for Windows' official GPU support. A failed import/conversion produces an explicit error and never silently falls back to blanket casting or renaming.

```bat
REM 1. Separate conversion environment
py -3.11 -m venv .venv-modelopt
.venv-modelopt\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-modelopt.txt

REM 2. Deterministic real-image AutoCast reference batches (not INT8 calibration)
python examples\yolo_fp16\generate_autocast_data.py --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --images-dir input\coco\images\val2017 --annotation-path input\coco\annotations\instances_val2017.json --output-dir examples\yolo_fp16\artifacts\autocast_data --count 32 --seed 0

REM 3. ModelOpt AutoCast conversion (actual installed ModelOpt version is recorded)
python examples\yolo_fp16\convert_fp16_modelopt.py --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --output-path examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.onnx --calibration-data examples\yolo_fp16\artifacts\autocast_data --providers cpu

REM 4. Checker, FP32 I/O, static shape, FP16 initializer/Cast inspection
python examples\yolo_fp16\inspect_mixed_precision_onnx.py --onnx-path examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.onnx --output-json examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.inspection.json

REM 5. Conversion error (FP32 ONNX -> mixed-FP16 ONNX), ORT CUDA only
python examples\yolo_fp16\compare_fp32_fp16_onnx.py --image-path assets\test_mouse.jpg --fp32-onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx --fp16-onnx-path examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.onnx

REM 6. Return to the main TensorRT/ORT-GPU/CUDA-PyTorch environment
call .venv-modelopt\Scripts\deactivate.bat
.venv\Scripts\activate

REM 7. Build strongly typed graph; TF32 stays off for remaining FP32 operations
python examples\yolo_tensorrt\build_engine.py --onnx-path examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.onnx --engine-path examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --model-precision mixed-fp16 --tf32 off --workspace-gb 1

REM 8. Preserve/read raw TRT 11.1 inspector output (do not infer precision from filename)
type examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine.inspector.json

REM 9. Build/runtime error (mixed-FP16 ONNX -> TensorRT), separate from conversion error
python examples\yolo_tensorrt\compare_onnx_tensorrt.py --image-path assets\test_mouse.jpg --onnx-path examples\yolo_fp16\artifacts\yolov8n_mixed_fp16.onnx --engine-path examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine

REM 10-11. Accuracy smoke then identical full COCO evaluation
python examples\yolo_coco\evaluate_coco.py --backend tensorrt --engine-path examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --conf-threshold 0.001 --iou-threshold 0.7 --limit 10 --output-json benchmark-results\fp16_coco_10.json
python examples\yolo_coco\evaluate_coco.py --backend tensorrt --engine-path examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --conf-threshold 0.001 --iou-threshold 0.7 --limit 5000 --output-json benchmark-results\fp16_coco_5000.json

REM 12. Pipeline functionality smoke (not a performance result)
python examples\yolo_benchmark\benchmark_precision.py --mode pipeline --engine fp16=examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --limit 100 --pipeline-rounds 2 --discard-rounds 1 --conf-threshold 0.25 --iou-threshold 0.45

REM 13. FP16 engine-only benchmark
python examples\yolo_benchmark\benchmark_precision.py --mode engine --engine fp16=examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --engine-warmup 50 --engine-iterations 500 --engine-rounds 4 --discard-rounds 1 --output-json benchmark-results\fp16_engine.json --output-csv benchmark-results\fp16_engine.csv

REM 14. FP16 full pipeline benchmark
python examples\yolo_benchmark\benchmark_precision.py --mode pipeline --engine fp16=examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --limit 5000 --pipeline-rounds 4 --discard-rounds 1 --conf-threshold 0.25 --iou-threshold 0.45 --output-json benchmark-results\fp16_pipeline.json --output-csv benchmark-results\fp16_pipeline.csv

REM 15. Final rotating-order FP32/FP16 comparison
python examples\yolo_benchmark\benchmark_precision.py --mode both --engine fp32=examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine --engine fp16=examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine --limit 5000 --engine-warmup 50 --engine-iterations 500 --engine-rounds 4 --pipeline-rounds 4 --discard-rounds 1 --conf-threshold 0.25 --iou-threshold 0.45 --output-json benchmark-results\fp32_fp16.json --output-csv benchmark-results\fp32_fp16.csv
```

The structure inspector proves only that internal FP16 representation exists (initializer and/or Cast); it does not claim every layer executes in FP16. The TensorRT inspector raw JSON is retained, while its schema is not guessed. Actual FP16 AP and latency remain `not measured` until these GPU commands are run locally.
