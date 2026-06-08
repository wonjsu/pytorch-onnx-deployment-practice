# pytorch-onnx-deployment-practice

Practice repository for exporting PyTorch vision models to ONNX and validating inference consistency.

## YOLO ONNX export

Run the YOLO export script from the repository root:

```bash
python examples/yolo_onnx/export_onnx.py
```

The script creates `examples/yolo_onnx/artifacts/` before exporting, temporarily switches the working directory to that folder for the Ultralytics export call, and writes the final model to `examples/yolo_onnx/artifacts/yolov8n.onnx`.

## YOLO ONNX latency benchmark breakdown

YOLO end-to-end latency can be measured with the benchmark script:

```bash
python examples/yolo_onnx/benchmark_yolo.py assets/test_mouse.jpg
```

The output keeps the existing PyTorch/Ultralytics ONNX/direct ONNX Runtime latency metrics and also prints a `Direct ONNX breakdown` section. If ONNX end-to-end latency is slow, check this breakdown before concluding that the model inference itself is slow: the extra time may come from image loading or letterbox preprocessing overhead rather than `session.run()` inference.
