# YOLOv8n precision benchmark results

This page separates measured results from planned measurements. FP16 and INT8 remain **not measured**; no values are inferred. INT8 is comparison-table context only and is not implemented by this workflow.

## Measured environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop GPU |
| TensorRT | 11.1.0.106 |
| Python | 3.11 |
| Batch / input | 1 / `1x3x640x640` |
| Engine | strict FP32 |
| TF32 | disabled |

## COCO accuracy

COCO val2017 (5,000 images), confidence `0.001`, NMS IoU `0.7`.

| Metric | FP32 strict | mixed FP16 | INT8 |
|---|---:|---:|---:|
| Prediction count | 857,979 | not measured | not measured |
| AP 0.50:0.95 | 0.3672 | not measured | not measured |
| AP 0.50 | 0.5165 | not measured | not measured |
| AP 0.75 | 0.3990 | not measured | not measured |
| AP small | 0.1774 | not measured | not measured |
| AP medium | 0.4048 | not measured | not measured |
| AP large | 0.5188 | not measured | not measured |
| AR maxDets=100 | 0.5547 | not measured | not measured |
| Absolute difference from FP32 | baseline | not measured | not measured |

## Engine-only benchmark

Warm-up 50; 500 iterations/round; 4 rounds; round 1 discarded; aggregate rounds 2–4.

| Metric | FP32 strict | mixed FP16 | INT8 |
|---|---:|---:|---:|
| H2D mean / median / P95 / round-mean SD (ms) | 0.838 / 0.762 / 1.196 / 0.009 | not measured | not measured |
| TensorRT compute mean / median / P95 / SD (ms) | 2.858 / 2.759 / 3.717 / 0.029 | not measured | not measured |
| D2H mean / median / P95 / SD (ms) | 0.283 / 0.267 / 0.362 / 0.001 | not measured | not measured |
| GPU total mean / median / P95 / SD (ms) | 3.979 / 3.851 / 4.835 / 0.033 | not measured | not measured |
| Host latency mean / median / P95 / SD (ms) | 7.120 / 6.935 / 8.507 / 0.134 | not measured | not measured |
| Throughput mean / median (FPS) | 141.808 / 144.204 | not measured | not measured |

## Pipeline benchmark

COCO val2017 (5,000 images), confidence `0.25`, NMS IoU `0.45`; 4 rounds; round 1 discarded; aggregate rounds 2–4.

| Metric (ms unless FPS) | FP32 strict | mixed FP16 | INT8 |
|---|---:|---:|---:|
| Image load mean / median / P95 | 2.983 / 2.966 / 4.203 | not measured | not measured |
| Preprocessing mean / median / P95 | 4.918 / 4.520 / 7.258 | not measured | not measured |
| H2D mean / median / P95 | 1.061 / 0.958 / 1.594 | not measured | not measured |
| TensorRT compute mean / median / P95 | 6.372 / 6.288 / 8.977 | not measured | not measured |
| D2H mean / median / P95 | 0.322 / 0.313 / 0.427 | not measured | not measured |
| Postprocessing mean / median / P95 | 2.037 / 1.943 / 2.695 | not measured | not measured |
| Other overhead mean / median / P95 | 3.096 / 2.973 / 4.311 | not measured | not measured |
| Pipeline excluding I/O mean / median / P95 / SD | 17.806 / 17.468 / 21.203 / 0.148 | not measured | not measured |
| Full end-to-end mean / median / P95 / SD | 20.790 / 20.538 / 24.209 / 0.146 | not measured | not measured |
| Pipeline throughput mean / median (FPS) | 56.771 / 57.248 | not measured | not measured |

FP16 cells must be filled only after running the same 5,000-image accuracy and benchmark protocols. Preserve prediction count and report each absolute FP32 delta.
