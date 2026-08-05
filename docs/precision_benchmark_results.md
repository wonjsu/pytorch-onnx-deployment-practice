# YOLOv8n TensorRT precision benchmark

이 문서는 target GPU에서 실제로 완료된 FP32, mixed-FP16, explicit-Q/DQ INT8 결과만 기록합니다.

## 1. 환경과 protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| TensorRT | 11.1.0.106 |
| PyTorch / CUDA | 2.13.0+cu126 / CUDA 12.6 |
| Input | batch 1, `1x3x640x640` |
| External engine I/O | FP32 |
| FP32 configuration | strict FP32, TF32 disabled |
| FP16 configuration | ModelOpt mixed-FP16 ONNX, TF32 disabled |
| INT8 configuration | ModelOpt explicit Q/DQ ONNX, entropy calibration |

정확도와 latency를 분리했습니다.

- Accuracy: COCO val2017 5,000 images, confidence `0.001`, NMS IoU `0.7`.
- Latency: confidence `0.25`, NMS IoU `0.45`.
- Engine benchmark: warm-up 50, 500 iterations per round, 4 rounds, round 1 discarded.
- Pipeline benchmark: 5,000 images per round, 4 rounds, round 1 discarded.
- FP32, FP16, INT8 engine order rotated between rounds.

INT8 baseline calibration은 val2017에서 seed 0으로 무작위 선택한 256장을 사용했습니다. 따라서 calibration 이미지와 evaluation 이미지가 일부 겹치는 tutorial-style baseline이며, 독립 calibration 결과로 표현하지 않습니다.

## 2. Precision validation

### 2.1 mixed-FP16

| ONNX inspection | Value |
|---|---:|
| FP16 initializers | 130 |
| FP32 initializers | 3 |
| Cast to FP16 | 1 |
| Cast to FP32 | 2 |
| Input / output dtype | FP32 / FP32 |

TensorRT Engine Inspector에서 internal `Half` datatype이 확인됐고 external engine I/O는 `Float`입니다.

단일 이미지 ORT CUDA 비교에서는 raw tensor `allclose=False`였지만 최종 detection은 동일 class로 매칭됐습니다.

| Metric | Value |
|---|---:|
| Max absolute error | 4.3576 |
| Mean absolute error | 0.006319 |
| RMSE | 0.04521 |
| Matched detections | 1 |
| Unmatched detections | 0 / 0 |
| Confidence | 0.662108 → 0.658203 |
| Confidence absolute difference | 0.003905 |
| Bounding-box IoU | 0.998908 |

### 2.2 explicit-Q/DQ INT8

| ONNX inspection | Value |
|---|---:|
| QuantizeLinear | 134 |
| DequantizeLinear | 134 |
| INT8 initializers | 268 |
| FP16 initializers | 399 |
| FP32 initializers | 2 |
| Input / output dtype | FP32 / FP32 |

TensorRT Engine Inspector에서 `Int8`, `Half`, `Float` datatype이 함께 확인됐습니다. Q/DQ node count는 explicit quantization graph를 증명하지만 모든 operation이 INT8로 실행된다는 뜻은 아닙니다. 현재 엔진은 INT8 구간과 FP16 fallback을 함께 사용하는 mixed-precision engine입니다.

## 3. COCO 정확도

| Metric | FP32 strict | mixed FP16 | INT8 Q/DQ | INT8 - FP32 |
|---|---:|---:|---:|---:|
| Prediction count | 857,979 | 860,789 | 893,396 | +35,417 |
| AP 0.50:0.95 | 0.3672 | 0.3674 | 0.3573 | -0.0099 |
| AP 0.50 | 0.5165 | 0.5170 | 0.5089 | -0.0076 |
| AP 0.75 | 0.3990 | 0.4000 | 0.3899 | -0.0091 |
| AP small | 0.1774 | 0.1776 | 0.1668 | -0.0106 |
| AP medium | 0.4048 | 0.4052 | 0.3902 | -0.0146 |
| AP large | 0.5188 | 0.5186 | 0.5107 | -0.0081 |
| AR maxDets=100 | 0.5547 | 0.5549 | 0.5469 | -0.0078 |

FP16에서는 의미 있는 정확도 저하가 없었습니다. 현재 INT8 baseline은 FP32 대비 AP 0.50:0.95가 `0.0099` 감소했습니다. 절대 감소는 medium object에서 가장 컸고, 상대 감소율은 small object에서 가장 컸습니다.

## 4. 동일 실행 engine-only benchmark

세 엔진을 같은 process에서 rotating order로 실행했습니다. 아래 값은 first round를 제외한 3 rounds aggregate입니다.

| Metric | FP32 strict | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|---:|
| H2D mean / median / P95 (ms) | 0.880 / 0.786 / 1.375 | 0.886 / 0.795 / 1.344 | 0.988 / 0.868 / 1.541 |
| Compute mean / median / P95 (ms) | 3.102 / 3.108 / 3.904 | 1.707 / 1.490 / 2.460 | 1.758 / 1.649 / 2.554 |
| D2H mean / median / P95 (ms) | 0.303 / 0.272 / 0.456 | 0.291 / 0.259 / 0.434 | 0.298 / 0.264 / 0.455 |
| GPU total mean / median / P95 (ms) | 4.285 / 4.269 / 5.311 | 2.884 / 2.842 / 3.778 | 3.044 / 2.986 / 4.090 |
| Host latency mean / median / P95 (ms) | 7.522 / 7.224 / 9.083 | 6.060 / 5.832 / 7.688 | 6.474 / 6.073 / 8.933 |
| Throughput mean / median (FPS) | 134.20 / 138.43 | 167.34 / 171.45 | 159.71 / 164.68 |

FP32 대비 median 변화:

| Metric | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|
| Compute latency | 52.1% 감소 | 46.9% 감소 |
| GPU total | 33.4% 감소 | 30.1% 감소 |
| Host latency | 19.3% 감소 | 15.9% 감소 |
| Throughput | 23.9% 증가 | 19.0% 증가 |

동일 실행에서는 INT8 compute median이 FP16보다 약 `10.7%` 느렸습니다. 따라서 현재 INT8 baseline이 FP16보다 더 낮은 precision을 사용한다는 사실만으로 더 빠른 engine이라고 결론 내릴 수 없습니다. INT8/FP16 precision transition, fallback layer, tactic selection, memory/launch overhead를 함께 고려해야 합니다.

## 5. 동일 실행 full pipeline benchmark

Pipeline mode는 image loading, CPU letterbox preprocessing, H2D, TensorRT execution, D2H, confidence filtering, coordinate restoration, class-aware NMS, Python overhead를 포함합니다.

| Metric | FP32 strict | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|---:|
| Image load mean / median / P95 (ms) | 3.101 / 3.096 / 4.370 | 3.104 / 3.090 / 4.375 | 3.113 / 3.110 / 4.369 |
| Preprocess mean / median / P95 (ms) | 5.188 / 4.773 / 7.662 | 5.231 / 4.818 / 7.566 | 5.240 / 4.829 / 7.614 |
| H2D mean / median / P95 (ms) | 1.088 / 0.969 / 1.638 | 1.132 / 1.004 / 1.694 | 1.193 / 1.077 / 1.755 |
| Compute mean / median / P95 (ms) | 6.386 / 6.314 / 8.235 | 3.681 / 3.014 / 7.330 | 3.578 / 3.028 / 7.544 |
| D2H mean / median / P95 (ms) | 0.333 / 0.314 / 0.477 | 0.336 / 0.306 / 0.544 | 0.352 / 0.310 / 0.580 |
| Postprocess mean / median / P95 (ms) | 2.105 / 2.018 / 2.834 | 2.177 / 2.103 / 2.984 | 2.152 / 2.068 / 2.941 |
| Other overhead mean / median / P95 (ms) | 3.315 / 3.171 / 4.911 | 3.261 / 3.066 / 4.759 | 3.253 / 3.038 / 4.791 |
| Pipeline excluding file I/O mean / median / P95 (ms) | 18.414 / 18.150 / 21.595 | 15.819 / 15.033 / 20.806 | 15.767 / 15.024 / 21.174 |
| Full E2E mean / median / P95 (ms) | 21.516 / 21.385 / 24.550 | 18.923 / 18.182 / 24.022 | 18.880 / 18.178 / 24.241 |
| Throughput mean / median (FPS) | 54.82 / 55.10 | 65.29 / 66.52 | 65.17 / 66.56 |

FP32 대비 median 변화:

| Metric | mixed FP16 | INT8 Q/DQ |
|---|---:|---:|
| Pipeline compute | 52.3% 감소 | 52.0% 감소 |
| Pipeline excluding file I/O | 17.2% 감소 | 17.2% 감소 |
| Full E2E | 15.0% 감소 | 15.0% 감소 |
| Pipeline throughput | 20.7% 증가 | 20.8% 증가 |

FP16과 INT8의 full E2E median 차이는 `0.004 ms`로 사실상 동일합니다. 모델 compute가 빨라진 뒤 CPU preprocessing, transfer, postprocessing, runtime overhead가 상대적으로 지배적이기 때문입니다.

전처리 median도 FP32 `4.773 ms`, FP16 `4.818 ms`, INT8 `4.829 ms`로 거의 같습니다. 세 engine이 동일한 CPU preprocessing 함수를 사용하므로 이 차이는 일반적인 실행 변동으로 해석합니다.

후처리 median은 FP32 `2.018 ms`, FP16 `2.103 ms`, INT8 `2.068 ms`입니다. 후처리는 raw score와 threshold 통과 candidate 수에 따라 작업량이 달라질 수 있지만, 현재 차이는 작고 candidate count를 benchmark에 별도로 저장하지 않았으므로 precision 영향으로 단정하지 않습니다.

## 6. 해석

현재 결과에서 FP16이 가장 실용적인 baseline입니다.

- FP16: accuracy 유지, isolated engine latency 최소, E2E 약 15% 감소.
- INT8: FP32 대비 E2E 약 15% 감소, 그러나 FP16 대비 추가 speedup 없음, AP 0.0099 감소.

따라서 다음 단계는 단순한 재측정이 아니라 INT8 구성 자체를 개선하는 것입니다.

1. entropy / max calibration 비교
2. sample count와 seed 안정성 비교
3. random / class-aware / size-aware calibration subset 비교
4. stage별 quantization sensitivity 측정
5. 민감한 구간의 FP16 fallback
6. AP-latency Pareto frontier 구성

## 7. 재현

### INT8 accuracy baseline

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

### 동일 실행 latency benchmark

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

## YOLOv8n INT8 calibration matrix

`examples/yolo_int8/run_calibration_matrix.py` runs an explicit-Q/DQ post-training INT8 matrix for YOLOv8n. It generates one deterministic master calibration set for the largest requested count and reuses prefixes for smaller configurations, so with seed `0` the 128-image set is the first 128 images of the 256 set, the 256-image set is the first 256 images of the 512 set, and the 512-image set is the first 512 images of the 1024 set. The `entropy_256` row is retained as the historical comparison baseline, not necessarily the winner.

Windows CMD example:

```cmd
python -m examples.yolo_int8.run_calibration_matrix ^
  --scope full ^
  --methods entropy max ^
  --counts 128 256 512 1024 ^
  --seed 0 ^
  --onnx-path examples\yolo_onnx\artifacts\yolov8n.onnx ^
  --calibration-images-dir input\coco\images\train2017 ^
  --calibration-annotation-path input\coco\annotations\instances_train2017.json ^
  --eval-images-dir input\coco\images\val2017 ^
  --eval-annotation-path input\coco\annotations\instances_val2017.json ^
  --output-dir precision-experiment-results\calibration_matrix_int8 ^
  --resume
```

The calibration image and annotation paths are required intentionally; the runner does not silently assume `train2017` exists. Use `--force` instead of `--resume` to delete and recreate only the selected output directory.
