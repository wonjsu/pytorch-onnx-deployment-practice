# Selective INT8 Quantization 결과

이 문서는 YOLOv8n TensorRT INT8 PTQ에서 calibration, quantization sensitivity, selective quantization exclusion, Engine Inspector 분석, 최종 latency 비교까지 수행한 결과를 정리합니다.

## 실험 환경

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- TensorRT: 11.1.0.106
- CUDA: 12.6
- PyTorch: 2.13.0+cu126
- Batch size: 1
- Input: `1x3x640x640`
- Accuracy: COCO val2017 5,000 images
- Accuracy thresholds: confidence `0.001`, NMS IoU `0.7`
- Latency thresholds: confidence `0.25`, NMS IoU `0.45`

INT8 calibration은 train2017에서 evaluation과 겹치지 않도록 seed 0으로 고정한 calibration subset을 사용했습니다. 최종 calibration baseline은 entropy / 128 images입니다.

## 1. Calibration matrix

Calibration method와 sample count를 비교한 결과 entropy 128이 가장 높은 AP를 보였습니다.

| Calibration | AP50:95 |
|---|---:|
| **entropy 128** | **0.358854** |
| entropy 256 | 0.354489 |
| entropy 512 | 0.355761 |
| entropy 1024 | 0.355072 |
| max 128 | 0.351699 |
| max 256 | 0.350052 |
| max 512 | 0.345063 |
| max 1024 | 0.346833 |

Strict FP32 AP50:95 `0.3672`와 비교하면 entropy128 baseline은 약 `0.00835` 낮았습니다. Calibration sample 수가 증가한다고 정확도가 단조롭게 증가하지 않았고, max calibration은 전체적으로 entropy보다 낮았습니다.

## 2. Selective quantization sensitivity

ModelOpt의 `nodes_to_exclude`를 이용해 특정 ONNX node를 Q/DQ quantization 대상에서 제외했습니다. 여기서 **quantization exclusion은 TensorRT kernel이 반드시 FP16으로 실행된다는 의미가 아닙니다.** 실제 engine precision과 fusion은 TensorRT Engine Inspector로 별도 확인했습니다.

### Stage 1: model.22 block-level exclusion

`model.22` 전체를 quantization에서 제외했을 때 가장 큰 정확도 회복이 나타났습니다.

| Candidate | AP50:95 | vs all-INT8 |
|---|---:|---:|
| all-INT8 entropy128 | 0.358854 | - |
| **block22 excluded** | **0.361656** | **+0.2803 AP point** |
| blocks00_04 | 0.361095 | +0.2242 AP point |

이 결과를 바탕으로 이후 실험은 `model.22` 내부를 세분화했습니다.

### Stage 2: model.22 subgroup 단독 exclusion

`model.22`를 `cv2`, `cv3`, `dfl`, `other`로 나누어 각각 단독으로 제외했습니다.

| Candidate | AP50:95 |
|---|---:|
| dfl only | 0.358423 |
| cv3 only | 0.357787 |
| other only | 0.355088 |
| cv2 only | 0.354881 |

어떤 subgroup도 단독으로는 all-INT8 baseline을 넘지 못했습니다. 즉 Stage 1의 정확도 회복은 특정 한 subgroup만의 효과가 아니라 subgroup 간 interaction에 의존했습니다.

### Stage 3: complement ablation

Stage 1 parent에서 하나의 subgroup만 다시 quantization에 포함시키는 complement ablation을 수행했습니다.

| Candidate | Quantized subgroup | Excluded node count | AP50:95 |
|---|---|---:|---:|
| leave_cv2_int8 | cv2 | 45 | 0.360372 |
| leave_cv3_int8 | cv3 | 45 | 0.359164 |
| leave_dfl_int8 | dfl | 61 | 0.357857 |
| **leave_other_int8** | **other** | **47** | **0.361695** |

Stage 3 winner `leave_other_int8`은 Stage 1의 66-node exclusion보다 exclusion 수를 `47`로 줄이면서 AP를 사실상 유지했습니다.

- Stage 1: `0.361656`
- Stage 3: `0.361695`
- exclusion count: `66 -> 47` (`-28.8%`)

특히 DFL은 단독 exclusion에서는 도움이 되지 않았지만 complement ablation에서 DFL을 quantized 상태로 되돌리면 정확도가 다시 감소했습니다. 이는 quantization sensitivity가 additive하지 않고 interaction-dependent임을 보여줍니다.

### Stage 4: pair exclusion

Stage 3 winner에서 exclusion을 더 줄이기 위해 `cv2`, `cv3`, `dfl` 중 두 그룹만 제외했습니다.

| Candidate | Excluded groups | Excluded node count | AP50:95 |
|---|---|---:|---:|
| **fp16_cv3_dfl** | **cv3 + dfl** | **26** | **0.360597** |
| fp16_cv2_dfl | cv2 + dfl | 26 | 0.359143 |
| fp16_cv2_cv3 | cv2 + cv3 | 42 | 0.357718 |

Stage 4 winner는 Stage 3보다 exclusion 수를 `47 -> 26`으로 줄였지만 AP50:95가 약 `0.00110` 감소했습니다.

### Stage 5: cv2 scale-branch ablation

Stage 3 winner를 parent로 두고 `cv2.0`, `cv2.1`, `cv2.2` 중 하나씩 Q/DQ quantization 대상으로 되돌렸습니다. 각 branch는 exact ONNX node 7개이며, exclusion 수는 `47 -> 40`으로 감소합니다.

| Candidate | Returned to Q/DQ | Excluded node count | AP50:95 | vs Stage 3 |
|---|---|---:|---:|---:|
| **quantize_cv2_1** | **cv2.1** | **40** | **0.361018** | **-0.0677 AP point** |
| quantize_cv2_0 | cv2.0 | 40 | 0.361013 | -0.0682 AP point |
| quantize_cv2_2 | cv2.2 | 40 | 0.360926 | -0.0769 AP point |

세 후보의 full-COCO accuracy는 매우 가까웠고, `quantize_cv2_1`을 latency finalist로 선택했습니다.

## 3. Engine Inspector에서 확인한 점

ModelOpt의 exclusion list와 실제 TensorRT execution precision은 동일하지 않았습니다.

- all-INT8 entropy128 ONNX도 TensorRT가 일부 `model.22` convolution을 Half로 실행했습니다.
- Stage 3에서는 `model.22/cv2`와 `cv3` 일부가 Half convolution으로 결합/fusion된 형태가 확인되었습니다.
- Stage 4에서 cv2를 다시 quantization 대상으로 돌리면 일부 cv2 convolution은 Int8 input/output/weight로 실행되었지만, cv3와 DFL은 여전히 Half로 실행되었습니다.

따라서 정확도 회복을 단순히 "INT8 layer를 FP16 layer로 바꿨다"고 해석하지 않습니다. 더 정확한 설명은 다음과 같습니다.

> 민감한 node를 Q/DQ quantization 대상에서 제외하여 quantization error를 줄였고, 그 결과 TensorRT의 precision assignment, fusion, reformat 경계까지 함께 변했다.

Q/DQ node count나 Inspector datatype endpoint count는 FLOPs 비율과 동일하지 않으므로 연산량 비율로 해석하지 않습니다.

## 4. Final balanced latency benchmark

최종 후보는 matched-builder FP16, Stage 3 winner, Stage 5 winner 세 엔진입니다.

기존 cyclic rotation은 각 엔진의 실행 위치는 균형화하지만 직전 predecessor가 고정되는 문제가 있었습니다. 최종 비교에서는 한 연속 session에서 첫 round를 버리고 다음 6개 permutation을 모두 한 번씩 실행했습니다.

1. FP16 -> Stage3 -> Stage5
2. Stage3 -> Stage5 -> FP16
3. Stage5 -> FP16 -> Stage3
4. FP16 -> Stage5 -> Stage3
5. Stage5 -> Stage3 -> FP16
6. Stage3 -> FP16 -> Stage5

이 protocol은 각 엔진이 1/2/3번째 실행 위치를 각각 두 번씩 차지하고, 모든 directed predecessor pair도 균형화합니다.

### Protocol

- one continuous benchmark session
- engine warmup: 50 iterations
- engine measurement: 500 iterations per round
- total rounds: 7
- discarded rounds: 1
- retained rounds: 6
- pipeline: COCO val2017 5,000 images per round
- reported values below: median of retained round means

### Accuracy + latency

| Model | AP50:95 | Engine compute | GPU total | Engine host | Pipeline compute | Pipeline no-I/O | Pipeline E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| **matched FP16** | **0.3674** | **1.628 ms** | **2.660 ms** | **5.854 ms** | **5.897 ms** | **15.556 ms** | **18.651 ms** |
| Stage 5 `quantize_cv2_1` | 0.361018 | 1.685 ms | 2.784 ms | 6.089 ms | 6.034 ms | 15.770 ms | 18.861 ms |
| Stage 3 `leave_other_int8` | 0.361695 | 1.734 ms | 2.856 ms | 6.128 ms | 6.223 ms | 15.864 ms | 18.963 ms |

Stage 5는 Stage 3보다 약간 빨랐습니다.

- engine compute: `-2.8%`
- pipeline compute: `-3.0%`
- pipeline no-I/O: `-0.60%`
- pipeline E2E: `-0.54%`

하지만 Stage 5는 Stage 3보다 AP50:95가 약 `0.000677` 낮습니다.

더 중요한 것은 FP16이 두 selective INT8 후보보다 정확도와 latency 모두 우수했다는 점입니다.

FP16 vs Stage 3:

- engine compute: 약 `6.1%` faster
- pipeline compute: 약 `5.2%` faster
- pipeline no-I/O: 약 `1.95%` faster
- pipeline E2E: 약 `1.64%` faster
- AP50:95: `0.3674` vs `0.361695`

FP16 vs Stage 5:

- engine compute: 약 `3.4%` faster
- pipeline compute: 약 `2.3%` faster
- pipeline no-I/O: 약 `1.36%` faster
- pipeline E2E: 약 `1.11%` faster
- AP50:95: `0.3674` vs `0.361018`

## 5. 최종 결론

Selective quantization exclusion은 all-INT8 entropy128 baseline의 정확도 손실을 일부 회복하는 데 성공했습니다.

- all-INT8 entropy128: `0.358854`
- Stage 3 winner: `0.361695`
- Stage 5 winner: `0.361018`
- matched FP16: `0.3674`

그러나 RTX 3060 Laptop GPU, TensorRT 11.1, batch 1 환경에서는 selective INT8이 FP16보다 latency 이점을 만들지 못했습니다. 최종 balanced benchmark에서는 matched-builder FP16이 accuracy와 engine/pipeline latency 모두 가장 우수했습니다.

따라서 현재 deployment candidate는 **mixed FP16**입니다. Selective INT8 실험은 여기서 종료하며, 추가적인 Stage 6 세분화는 진행하지 않습니다.

이 결과는 INT8이 항상 FP16보다 빠르다고 가정할 수 없다는 점을 보여줍니다. 실제 deployment 성능은 Q/DQ overhead, reformat, precision boundary, TensorRT tactic/fusion 선택, GPU architecture와 workload에 의해 결정됩니다.

## 재현성 메모

대용량 artifact는 repository에 commit하지 않습니다.

- `.onnx`, `.engine`, `.plan`
- calibration `.npz`
- COCO prediction JSON
- raw benchmark JSON/CSV
- Engine Inspector raw JSON

대신 실험 runner, metadata, benchmark protocol과 최종 요약 결과를 repository에 남깁니다.
