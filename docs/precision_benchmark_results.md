# YOLOv8n TensorRT precision benchmark

This document records only measurements completed on the target GPU. FP32 and mixed-FP16 are measured. INT8 remains unmeasured.

## 1. Environment and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| TensorRT | 11.1.0.106 |
| PyTorch / CUDA | 2.13.0+cu126 / CUDA 12.6 |
| Input | batch 1, `1x3x640x640` |
| External engine I/O | FP32 |
| FP32 configuration | strict FP32, TF32 disabled |
| FP16 configuration | ModelOpt mixed-FP16 ONNX, TF32 disabled |

Accuracy and latency were evaluated separately.

- Accuracy: COCO val2017 5,000 images, confidence `0.001`, NMS IoU `0.7`.
- Latency: confidence `0.25`, NMS IoU `0.45`.
- Engine benchmark: warm-up 50, 500 iterations per round, 4 rounds, round 1 discarded.
- Pipeline benchmark: 5,000 images per round, 4 rounds, round 1 discarded.
- FP32 and FP16 engine order was rotated between rounds.

## 2. Precision validation

The mixed-FP16 ONNX keeps FP32 external I/O while storing most internal initializers as FP16.

| ONNX inspection | Value |
|---|---:|
| FP16 initializers | 130 |
| FP32 initializers | 3 |
| Cast to FP16 | 1 |
| Cast to FP32 | 2 |
| Input / output dtype | FP32 / FP32 |

TensorRT Engine Inspector reported `Half` datatypes across internal layers while engine input and output remained `Float`. Precision is therefore supported by graph and inspector evidence, not inferred from the filename.

### FP32 ONNX vs mixed-FP16 ONNX sanity check

Single-image ORT CUDA comparison:

| Metric | Value |
|---|---:|
| Raw output shape | `(1, 84, 8400)` |
| Max absolute error | 4.3576 |
| Mean absolute error | 0.006319 |
| RMSE | 0.04521 |
| `allclose` | False |
| Matched detections | 1 |
| Unmatched detections | 0 / 0 |
| Confidence | 0.662108 → 0.658203 |
| Confidence absolute difference | 0.003905 |
| Bounding-box IoU | 0.998908 |

The strict elementwise tolerance did not pass, but the final detection remained matched with high box overlap. Final accuracy was therefore judged with the full COCO evaluation rather than `allclose` alone.

## 3. COCO accuracy

| Metric | FP32 strict | mixed FP16 | Absolute delta |
|---|---:|---:|---:|
| Prediction count | 857,979 | 860,789 | +2,810 |
| AP 0.50:0.95 | 0.3672 | 0.3674 | +0.0002 |
| AP 0.50 | 0.5165 | 0.5170 | +0.0005 |
| AP 0.75 | 0.3990 | 0.4000 | +0.0010 |
| AP small | 0.1774 | 0.1776 | +0.0002 |
| AP medium | 0.4048 | 0.4052 | +0.0004 |
| AP large | 0.5188 | 0.5186 | -0.0002 |
| AR maxDets=100 | 0.5547 | 0.5549 | +0.0002 |

No meaningful accuracy degradation was observed at COCO scale.

## 4. Engine-only benchmark

The engine-only benchmark isolates a fixed preprocessed tensor and reuses device buffers, pinned host buffers, a non-default CUDA stream, and CUDA events.

| Metric | FP32 strict | mixed FP16 | Change |
|---|---:|---:|---:|
| H2D mean / median / P95 (ms) | 0.721 / 0.700 / 0.868 | 0.733 / 0.712 / 0.913 | transfer unchanged |
| TensorRT compute mean / median / P95 (ms) | 2.625 / 2.625 / 2.770 | 1.395 / 1.337 / 1.870 | median **49.1% lower** |
| D2H mean / median / P95 (ms) | 0.271 / 0.261 / 0.323 | 0.275 / 0.259 / 0.340 | transfer unchanged |
| GPU total mean / median / P95 (ms) | 3.617 / 3.590 / 3.895 | 2.403 / 2.332 / 2.936 | median **35.0% lower** |
| Host latency mean / median / P95 (ms) | 6.077 / 6.010 / 6.614 | 4.806 / 4.689 / 5.601 | median **22.0% lower** |
| Throughput mean / median (FPS) | 164.89 / 166.40 | 209.30 / 213.26 | median **28.2% higher** |

This demonstrates a clear FP16 benefit inside TensorRT compute.

## 5. Full pipeline benchmark

Pipeline mode includes image loading, CPU letterbox preprocessing, H2D, TensorRT execution, D2H, confidence filtering, coordinate restoration, class-aware NMS, and Python overhead.

| Metric | FP32 strict | mixed FP16 | Change |
|---|---:|---:|---:|
| Image load mean / median / P95 (ms) | 2.847 / 2.845 / 3.948 | 2.837 / 2.834 / 3.910 | similar |
| Preprocess mean / median / P95 (ms) | 4.496 / 4.068 / 6.537 | 4.458 / 4.051 / 6.435 | similar |
| H2D mean / median / P95 (ms) | 0.959 / 0.942 / 1.173 | 1.302 / 1.256 / 1.716 | higher in this run |
| TensorRT compute mean / median / P95 (ms) | 7.465 / 7.630 / 7.794 | 6.388 / 6.176 / 7.972 | median **19.1% lower**; P95 slightly worse |
| D2H mean / median / P95 (ms) | 0.348 / 0.345 / 0.426 | 0.611 / 0.580 / 0.841 | higher in this run |
| Postprocess mean / median / P95 (ms) | 1.861 / 1.804 / 2.302 | 1.848 / 1.798 / 2.245 | similar |
| Other overhead mean / median / P95 (ms) | 2.505 / 2.413 / 3.051 | 2.570 / 2.508 / 3.061 | similar |
| Pipeline excluding file I/O mean / median / P95 (ms) | 17.634 / 17.295 / 20.144 | 17.176 / 16.616 / 20.617 | median **3.9% lower**; P95 2.3% higher |
| Full E2E mean / median / P95 (ms) | 20.481 / 20.342 / 22.853 | 20.013 / 19.665 / 23.675 | median **3.3% lower**; P95 3.6% higher |
| Pipeline throughput mean / median (FPS) | 57.08 / 57.82 | 58.70 / 60.18 | median **4.1% higher** |

## 6. Interpretation

FP16 reduced isolated TensorRT compute latency by approximately half, but full-pipeline median latency improved by only 3–4%. The dominant remaining costs are CPU preprocessing, transfers, postprocessing, and Python/runtime overhead. FP16 also showed higher H2D/D2H timings and slightly worse P95 pipeline latency in this measurement, so those transfer and tail-latency effects should be investigated before claiming a uniformly faster end-to-end system.

The next optimization target is therefore not another model-format conversion. It is reducing non-compute overhead through GPU preprocessing, avoiding unnecessary host transfers, moving NMS/postprocessing closer to the device, and then re-running the same rotating benchmark protocol.

## 7. Reproduction

```bat
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope smoke
py -3.11 tools\run_precision_experiments.py --stage fp16 --scope full --resume
```

INT8 explicit-Q/DQ tooling is implemented, but no INT8 values are recorded here until calibration, engine inspection, COCO accuracy, and the same latency protocol complete successfully on the target GPU.
