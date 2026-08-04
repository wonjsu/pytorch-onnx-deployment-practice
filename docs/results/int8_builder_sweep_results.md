# INT8 TensorRT builder sweep results

## Scope

This experiment changed only TensorRT builder parameters for the same explicit-Q/DQ YOLOv8n INT8 ONNX model. It did not change calibration data, quantization scales, graph exclusions, input shape, or batch size.

Environment:

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- TensorRT: 11.1.0.106
- CUDA: 12.6
- Driver: 561.09
- Input: FP32 `[1, 3, 640, 640]`
- Batch size: 1
- Source ONNX SHA-256: `0131ba64009833601fdf277dfe357d766311cde9ac99ec3212aff84445a8800a`

The sweep had four stages:

1. Baseline suite: 11 configurations.
2. Extended suite: 13 additional configurations and interactions.
3. Same-process screening: all 24 engines.
4. Balanced final: six selected engines, seven rounds, first round discarded. The six retained rounds rotate every engine through every execution position once.

## Reference configuration

```text
builder_optimization_level = 5
avg_timing_iterations       = 8
max_num_tactics             = auto (-1)
workspace                   = 2 GiB
max_aux_streams             = 0
```

## Balanced final result

Each engine used 50 warm-up iterations and 500 measured iterations per round. Six retained rounds produced 3,000 samples per engine.

| Rank | Configuration | Compute median (ms) | Compute P95 (ms) | GPU total median (ms) | Host median (ms) | Host P95 (ms) | Median FPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `aux1` | 1.3624 | 2.0246 | 2.4463 | 4.8297 | 5.9501 | 207.05 |
| 2 | `tactics32` | 1.4113 | 2.1418 | 2.4878 | 4.8869 | 6.2022 | 204.63 |
| 3 | `opt2` | 1.4177 | **1.9735** | 2.4781 | 4.8819 | 6.5607 | 204.84 |
| 4 | `tactics8` | 1.4233 | 2.0934 | 2.4856 | 4.8535 | 5.9845 | 206.03 |
| 5 | `opt1` | 1.4367 | 2.0555 | 2.4844 | 4.9037 | 6.5220 | 203.93 |
| 6 | `reference` | 1.5435 | 2.2751 | 2.6103 | 5.0762 | 6.6060 | 197.00 |

## Selected setting

`aux1` changes only `max_aux_streams` from `0` to `1` relative to the reference.

Relative to the reference, its balanced-final medians changed by:

- TensorRT compute latency: **-11.74%**
- GPU total latency: **-6.28%**
- Host latency: **-4.86%**
- Throughput: **+5.10%**

For this hardware and software stack, `aux1` is the selected final-build configuration because it led on compute median, GPU-total median, host median, and median throughput. `opt2` had the lowest compute P95 but a worse host P95 than `aux1`.

## Build-time trade-off

Builder optimization levels 1 and 2 built in about 43 seconds in the extended sweep, while the level-5 reference and `aux1` builds took about 355 seconds. Their final runtime was slightly slower than `aux1`, so they are useful as development-time build presets rather than the selected final engine configuration.

## Interpretation limits

- These results are specific to this RTX 3060 Laptop GPU, TensorRT 11.1.0.106, CUDA 12.6, driver 561.09, model graph, and input shape.
- This is engine-only batch-1 latency. Image loading, preprocessing, postprocessing, and COCO evaluation are excluded.
- Builder parameters affect tactic search and execution planning; they do not change the trained weights or the Q/DQ calibration scales in the source ONNX model.
- Rebuilding can select different tactics. A rebuilt engine must be identified by its SHA-256 and benchmarked again.
- The 24-engine run was used as screening. The balanced six-engine run is the final comparison because it removes the fixed execution-position imbalance present when 24 engines were measured for only four rounds.

## Machine-readable artifact

`docs/results/int8_builder_sweep_results.json` contains compact metadata and aggregate metrics for the balanced final run and the protocols used by the preceding screening stages. Raw per-iteration benchmark files remain local because they are large and contain redundant samples.
