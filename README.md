# YOLOv8n ONNX / TensorRT Deployment Optimization

PyTorch YOLOv8n을 ONNX로 변환하고, ONNX Runtime과 TensorRT에서 출력 일관성·정확도·지연시간을 검증하는 배포 최적화 프로젝트입니다.

이 저장소의 중심은 단순한 포맷 변환이 아니라 다음 문제를 재현 가능한 방식으로 확인하는 것입니다.

- 변환된 모델의 출력이 원본과 얼마나 일치하는가
- TensorRT FP32, mixed-FP16, explicit-Q/DQ INT8이 실제 엔진 내부에서 어떤 precision으로 실행되는가
- COCO 정확도 변화와 TensorRT compute latency 감소 사이의 trade-off는 무엇인가
- 모델 연산이 빨라져도 전체 pipeline이 충분히 빨라지지 않는 이유는 무엇인가
- INT8 calibration과 mixed-precision fallback을 어떻게 개선할 것인가

## 핵심 결과

실행 환경: NVIDIA GeForce RTX 3060 Laptop GPU, TensorRT 11.1.0.106, CUDA 12.6, batch 1, `1x3x640x640`.

정확도 평가는 COCO val2017 5,000장으로 수행했습니다. INT8 baseline은 val2017에서 seed 0으로 무작위 선택한 256장을 entropy calibration에 사용했으므로 calibration과 evaluation이 일부 겹치는 tutorial-style baseline입니다.

### 정확도

| Metric | FP32 strict | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|---:|
| COCO AP 0.50:0.95 | 0.3672 | 0.3674 | 0.3573 |
| AP 0.50 | 0.5165 | 0.5170 | 0.5089 |
| AP 0.75 | 0.3990 | 0.4000 | 0.3899 |
| AP small | 0.1774 | 0.1776 | 0.1668 |
| AP medium | 0.4048 | 0.4052 | 0.3902 |
| AP large | 0.5188 | 0.5186 | 0.5107 |
| AR maxDets=100 | 0.5547 | 0.5549 | 0.5469 |

FP16은 정확도를 유지했습니다. 현재 INT8 baseline은 FP32 대비 AP 0.50:0.95가 `0.0099` 감소했습니다.

### 동일 실행 FP32 / FP16 / INT8 latency

세 엔진을 같은 프로세스에서 rotating order로 측정했습니다. 4 rounds 중 첫 round는 제외했습니다.

| Metric | FP32 strict | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|---:|
| Engine compute median | 3.108 ms | 1.490 ms | 1.649 ms |
| GPU total median | 4.269 ms | 2.842 ms | 2.986 ms |
| Host latency median | 7.224 ms | 5.832 ms | 6.073 ms |
| Engine throughput median | 138.43 FPS | 171.45 FPS | 164.68 FPS |
| Pipeline compute median | 6.314 ms | 3.014 ms | 3.028 ms |
| Pipeline excluding file I/O median | 18.150 ms | 15.033 ms | 15.024 ms |
| Full E2E median | 21.385 ms | 18.182 ms | 18.178 ms |
| Pipeline throughput median | 55.10 FPS | 66.52 FPS | 66.56 FPS |

같은 실행 안에서는 FP16과 INT8의 full-pipeline latency가 사실상 동일했습니다. INT8 engine compute는 FP16보다 약 `10.7%` 느렸지만, 전체 E2E 차이는 `0.004 ms` 수준이었습니다. 현재 INT8 엔진은 순수 INT8이 아니라 INT8·FP16·FP32 mixed precision이며, Inspector 기준으로 INT8 구간과 FP16 fallback이 함께 존재합니다.

상세 결과와 측정 조건은 [`docs/precision_benchmark_results.md`](docs/precision_benchmark_results.md)에 기록했습니다.

## 검증 근거

### FP16

ModelOpt AutoCast로 생성된 ONNX는 외부 입력과 출력을 FP32로 유지하고 내부 표현을 FP16 중심으로 변환합니다.

- FP16 initializer: 130
- FP32 initializer: 3
- Cast to FP16: 1
- Cast to FP32: 2
- Input / output: FP32
- TensorRT Engine Inspector에서 내부 `Half` datatype 확인

단일 이미지 ORT CUDA 비교에서는 raw tensor `allclose=False`였지만, 최종 검출은 동일 class로 매칭되었고 confidence 차이는 `0.003905`, bbox IoU는 `0.998908`이었습니다. 최종 정확도는 COCO val2017 5,000장으로 판단했습니다.

### INT8 explicit Q/DQ

TensorRT 11.1에서 legacy calibrator나 `BuilderFlag.INT8`을 사용하지 않습니다. ModelOpt가 FP32 ONNX에 calibration 결과와 explicit Q/DQ를 삽입하고 TensorRT가 strongly typed graph를 build합니다.

- `QuantizeLinear`: 134
- `DequantizeLinear`: 134
- INT8 initializer: 268
- FP16 initializer: 399
- Input / output: FP32
- Engine Inspector에서 `Int8`, `Half`, `Float` datatype 확인

Q/DQ 노드 수만으로 모든 연산이 INT8이라고 주장하지 않습니다. 현재 엔진은 INT8 실행 구간과 FP16 fallback을 함께 사용합니다.

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
| `docs/precision_benchmark_results.md` | 측정된 FP32/FP16/INT8 정확도와 latency 결과 |
| `examples/resnet18_onnx` | 초기 ONNX Runtime 변환 검증 예제. 현재 프로젝트의 핵심 결과는 아님 |

## 실행

Windows CMD, repository root 기준입니다.

### FP16 전체 실험

```bat
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope smoke
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope full --resume
```

### INT8 baseline

```bat
py -3.11 tools\run_precision_experiments.py --stage int8 --scope smoke ^
  --calibration-images-dir input\coco\images\val2017 ^
  --calibration-annotation-path input\coco\annotations\instances_val2017.json ^
  --calibration-count 256 ^
  --calibration-seed 0 ^
  --calibration-method entropy ^
  --output-dir precision-experiment-results\int8_val256_entropy
```

```bat
py -3.11 tools\run_precision_experiments.py --stage int8 --scope full ^
  --calibration-images-dir input\coco\images\val2017 ^
  --calibration-annotation-path input\coco\annotations\instances_val2017.json ^
  --calibration-count 256 ^
  --calibration-seed 0 ^
  --calibration-method entropy ^
  --output-dir precision-experiment-results\int8_val256_entropy ^
  --resume
```

### 동일 실행 latency 비교

```bat
python -m examples.yolo_benchmark.benchmark_precision ^
  --mode both ^
  --engine fp32=examples\yolo_tensorrt\artifacts\yolov8n_fp32_strict.engine ^
  --engine fp16=examples\yolo_tensorrt\artifacts\yolov8n_mixed_fp16.engine ^
  --engine int8=examples\yolo_tensorrt\artifacts\yolov8n_int8.engine ^
  --images-dir input\coco\images\val2017 ^
  --annotation-path input\coco\annotations\instances_val2017.json ^
  --limit 5000 ^
  --engine-warmup 50 ^
  --engine-iterations 500 ^
  --engine-rounds 4 ^
  --pipeline-rounds 4 ^
  --discard-rounds 1 ^
  --conf-threshold 0.25 ^
  --iou-threshold 0.45 ^
  --output-json precision-experiment-results\timing_fp32_fp16_int8\benchmark_full.json ^
  --output-csv precision-experiment-results\timing_fp32_fp16_int8\benchmark_full.csv
```

## 현재 해석과 다음 단계

FP16은 정확도 저하 없이 현재 INT8 baseline과 같거나 더 나은 isolated-engine latency를 보였습니다. INT8은 FP32 대비 E2E를 줄였지만 FP16보다 추가 속도 이득이 없고 AP가 `0.0099` 감소했습니다.

따라서 다음 단계는 단순히 INT8을 다시 실행하는 것이 아니라 다음을 분석하는 것입니다.

1. calibration method, sample count, seed, sampling strategy 비교
2. layer/stage별 quantization sensitivity 측정
3. 민감한 구간의 FP16 fallback
4. AP와 latency의 Pareto 비교
5. 이후 CUDA preprocessing 및 GPU postprocessing 검토

## TensorRT builder sweep

현재 INT8 Q/DQ ONNX를 빠르게 검증하는 smoke sweep은 Windows CMD에서 다음과 같이 실행합니다.

```bat
py -3.11 -m examples.yolo_tensorrt.run_builder_sweep ^
  --onnx-path examples\yolo_int8\artifacts\yolov8n_int8_qdq.onnx ^
  --model-precision int8 ^
  --output-dir precision-experiment-results\builder_sweep_int8 ^
  --profile smoke
```

최종 비교에는 전체 rotating benchmark를 실행합니다.

```bat
py -3.11 -m examples.yolo_tensorrt.run_builder_sweep ^
  --onnx-path examples\yolo_int8\artifacts\yolov8n_int8_qdq.onnx ^
  --model-precision int8 ^
  --output-dir precision-experiment-results\builder_sweep_int8 ^
  --profile full ^
  --force
```

Builder optimization level은 tactic 검색 공간을 바꾸고, `max-num-tactics`는 timing할 후보 tactic 수를 제한합니다. `avg-timing-iterations`는 각 후보의 timing을 반복하여 선택 노이즈를 줄입니다. 이 설정들은 engine construction만 변경하며 calibration scale이나 Q/DQ 배치를 변경하지 않습니다. Smoke 결과는 실행 가능성을 확인하기 위한 validation일 뿐입니다. 최종 결론에는 engine 순서를 round마다 회전하는 full benchmark 결과를 사용해야 합니다.

## Legacy ONNX example

`examples/resnet18_onnx`는 PyTorch와 ONNX Runtime의 기본 output consistency 및 inference-only/end-to-end 측정을 익히기 위해 만든 초기 예제입니다. 저장소에는 남겨 두지만, 현재 프로젝트의 중심은 YOLOv8n TensorRT precision 최적화와 pipeline bottleneck 분석입니다.

## Artifact 정책

다음 파일은 환경 의존적이거나 크기가 크므로 commit하지 않습니다.

- `.onnx`, `.engine`, `.plan`
- calibration `.npz`
- COCO dataset과 prediction JSON
- Engine Inspector raw JSON
- local benchmark raw JSON/CSV

TensorRT engine은 GPU architecture, TensorRT version, CUDA/driver에 의존하므로 binary 대신 build workflow, metadata schema, benchmark protocol, 검증 결과를 문서화합니다.
