# YOLOv8n ONNX / TensorRT Deployment Optimization

PyTorch YOLOv8n을 ONNX로 변환하고, ONNX Runtime과 TensorRT에서 출력 일관성·정확도·지연시간을 검증하는 배포 최적화 프로젝트입니다.

이 저장소의 중심은 단순한 포맷 변환이 아니라 다음 문제를 재현 가능한 방식으로 확인하는 것입니다.

- 변환된 모델의 출력이 원본과 얼마나 일치하는가
- TensorRT FP32와 mixed-FP16이 실제 엔진 내부에서 어떤 precision으로 실행되는가
- COCO 정확도를 유지하면서 TensorRT compute latency가 얼마나 줄어드는가
- 모델 연산이 빨라져도 전체 pipeline이 충분히 빨라지지 않는 이유는 무엇인가
- INT8 explicit Q/DQ 실험을 동일한 평가 기준으로 어떻게 확장할 것인가

## 핵심 결과

실행 환경: NVIDIA GeForce RTX 3060 Laptop GPU, TensorRT 11.1.0.106, CUDA 12.6, batch 1, `1x3x640x640`.

### FP32 strict vs mixed-FP16 TensorRT

| Metric | FP32 strict | mixed FP16 | 변화 |
|---|---:|---:|---:|
| COCO AP 0.50:0.95 | 0.3672 | 0.3674 | +0.0002 |
| Engine compute median | 2.625 ms | 1.337 ms | **49.1% 감소** |
| GPU total median | 3.590 ms | 2.332 ms | **35.0% 감소** |
| Host latency median | 6.010 ms | 4.689 ms | **22.0% 감소** |
| Engine throughput median | 166.40 FPS | 213.26 FPS | **28.2% 증가** |
| Pipeline excluding file I/O median | 17.295 ms | 16.616 ms | **3.9% 감소** |
| Full E2E median | 20.342 ms | 19.665 ms | **3.3% 감소** |

FP16은 TensorRT compute를 거의 절반으로 줄였지만 전체 pipeline 개선은 3~4%에 그쳤습니다. CPU letterbox preprocessing, H2D/D2H, postprocessing/NMS, Python overhead가 남아 있기 때문입니다. 또한 이번 측정에서는 FP16의 pipeline H2D/D2H와 P95 지연시간이 일부 증가했으므로, FP16을 단순히 모든 지표에서 빠르다고 해석하지 않습니다.

상세 결과와 측정 조건은 [`docs/precision_benchmark_results.md`](docs/precision_benchmark_results.md)에 기록했습니다.

## 검증 근거

### 1. FP32 ONNX → mixed-FP16 ONNX

ModelOpt AutoCast로 생성된 ONNX는 외부 입력과 출력을 FP32로 유지하고 내부 표현을 FP16 중심으로 변환합니다.

- FP16 initializer: 130
- FP32 initializer: 3
- Cast to FP16: 1
- Cast to FP32: 2
- Input: `images`, FP32, `(1, 3, 640, 640)`
- Output: `output0`, FP32, `(1, 84, 8400)`

단일 이미지 ORT CUDA 비교에서는 raw tensor `allclose=False`였지만, 최종 검출은 동일 class로 매칭되었고 confidence 차이는 `0.003905`, bbox IoU는 `0.998908`이었습니다. 최종 정확도 판단은 단일 tensor tolerance가 아니라 COCO val2017 5,000장 평가로 수행했습니다.

### 2. mixed-FP16 ONNX → TensorRT engine

TensorRT 11.1 strongly typed workflow를 사용합니다.

- `BuilderFlag.FP16`을 사용하지 않음
- FP16 precision은 ModelOpt가 생성한 mixed-FP16 ONNX 그래프에 기록
- TF32는 비활성화
- TensorRT Engine Inspector에서 내부 `Half` datatype 확인
- 외부 engine I/O는 `Float` 유지

엔진 파일명이나 CLI label만으로 precision을 주장하지 않고, ONNX 구조 검사와 Engine Inspector 결과를 metadata에 함께 저장합니다.

## Benchmark 설계

정확도와 latency를 분리합니다.

### Accuracy

- COCO val2017 5,000장
- confidence threshold `0.001`
- NMS IoU threshold `0.7`
- prediction JSON과 COCOeval 결과 저장

### Latency

- confidence threshold `0.25`
- NMS IoU threshold `0.45`
- 4 rounds, first round discarded
- engine order rotated between rounds
- mean / median / P95 / round-mean standard deviation 기록

`engine` mode는 고정된 preprocessed tensor와 재사용 CUDA buffer를 사용해 H2D, TensorRT compute, D2H를 측정합니다.

`pipeline` mode는 이미지 load/decode, CPU preprocessing, H2D, compute, D2H, confidence filtering, bbox 복원, class-aware NMS, 기타 overhead를 포함합니다.

## Repository 구성

| 경로 | 내용 |
|---|---|
| `examples/yolo_onnx` | YOLOv8n ONNX export, letterbox preprocessing, bbox 복원, NMS, PyTorch/ORT 출력 비교 |
| `examples/yolo_coco` | PyTorch, ONNX Runtime, TensorRT COCO 정확도 평가 |
| `examples/yolo_tensorrt` | TensorRT 11.1 engine builder, named tensor API runner, Engine Inspector, ONNX/TRT 비교 |
| `examples/yolo_fp16` | ModelOpt AutoCast 데이터 생성, mixed-FP16 변환, 구조 검사, FP32/FP16 ORT 비교 |
| `examples/yolo_int8` | ModelOpt PTQ calibration, explicit Q/DQ ONNX 생성 및 검사 |
| `examples/yolo_benchmark` | FP32/FP16/INT8 rotating-order engine 및 pipeline benchmark |
| `tools/run_precision_experiments.py` | `.venv`와 `.venv-modelopt`를 직접 호출하는 smoke/full 통합 실행기 |
| `docs/precision_benchmark_results.md` | 측정된 FP32/FP16 정확도와 latency 결과 |
| `examples/resnet18_onnx` | 초기 ONNX Runtime 변환 검증 예제. 현재 프로젝트의 핵심 결과는 아님 |

## 실행

Windows CMD, repository root 기준입니다.

### 환경

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-tensorrt.txt
```

ModelOpt는 dependency 충돌을 피하기 위해 별도 환경에 설치합니다.

```bat
py -3.11 -m venv .venv-modelopt
.venv-modelopt\Scripts\activate
python -m pip install -r requirements-modelopt.txt
```

### FP16 전체 실험

통합 실행기는 활성화된 환경에 의존하지 않고 각 virtual environment의 Python을 직접 호출합니다.

```bat
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope smoke
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope full --resume
```

`smoke`는 engine build, 단일 이미지 출력 비교, COCO 10장 accuracy, 100장 단축 benchmark가 끝까지 동작하는지 확인합니다. 최종 성능 수치로 사용하지 않습니다.

`full`은 COCO 5,000장 정확도와 FP32/FP16 rotating-order engine/pipeline benchmark를 수행합니다.

### INT8 explicit Q/DQ

TensorRT 11.1에서 legacy calibrator나 `BuilderFlag.INT8`을 사용하지 않습니다. ModelOpt가 FP32 ONNX에 calibration 결과와 explicit Q/DQ를 삽입하고, TensorRT는 해당 strongly typed graph를 build합니다.

```bat
py -3.11 tools\run_precision_experiments.py --stage int8 --scope smoke ^
  --calibration-images-dir input\coco\images\train2017 ^
  --calibration-annotation-path input\coco\annotations\instances_train2017.json ^
  --calibration-count 256 ^
  --calibration-method entropy
```

INT8 workflow는 구현되어 있지만, 정확도와 latency 수치는 실제 full protocol이 완료되기 전까지 결과 표에 기록하지 않습니다.

## 구현상의 주요 선택

- ONNX Runtime CUDA와 TensorRT에서 CPU fallback을 허용하지 않음
- TensorRT 11.1 named tensor API와 `execute_async_v3()` 사용
- non-default CUDA stream 사용
- reusable CUDA input/output buffers 및 pinned host output buffer 사용
- CUDA event로 H2D / compute / D2H 분리 측정
- engine artifact에 source ONNX SHA-256, TensorRT/CUDA/GPU 정보, I/O metadata 저장
- engine binary와 ONNX/calibration/result artifact는 Git에서 제외
- accuracy threshold와 latency threshold를 분리
- cold-cache 영향을 줄이기 위해 첫 round 제외 및 engine order rotation

## 현재 해석과 다음 단계

mixed-FP16은 정확도 저하 없이 TensorRT compute를 크게 줄였습니다. 그러나 full pipeline의 개선 폭은 작았고 tail latency는 일관되게 개선되지 않았습니다.

따라서 다음 최적화 우선순위는 다음과 같습니다.

1. CPU letterbox preprocessing의 CUDA 이전
2. 불필요한 host-device transfer 제거
3. postprocessing/NMS의 GPU 실행 또는 TensorRT graph 통합 검토
4. 같은 protocol로 FP32/FP16 재측정
5. INT8 calibration과 sensitivity 분석을 통해 최소 FP16 fallback 탐색

## Legacy ONNX example

`examples/resnet18_onnx`는 PyTorch와 ONNX Runtime의 기본 output consistency 및 inference-only/end-to-end 측정을 익히기 위해 만든 초기 예제입니다. 저장소에는 남겨 두지만, 취업용 프로젝트의 핵심은 YOLOv8n TensorRT precision 최적화와 pipeline bottleneck 분석입니다.

## Artifact 정책

다음 파일은 환경 의존적이거나 크기가 크므로 commit하지 않습니다.

- `.onnx`, `.engine`, `.plan`
- calibration `.npz`
- COCO dataset과 prediction JSON
- Engine Inspector raw JSON
- local benchmark raw JSON/CSV

TensorRT engine은 GPU architecture, TensorRT version, CUDA/driver에 의존하므로 binary 대신 build workflow, metadata schema, benchmark protocol, 검증 결과를 문서화합니다.
