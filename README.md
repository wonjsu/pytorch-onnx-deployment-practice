# pytorch-onnx-deployment-practice

Practice repository for exporting PyTorch vision models to ONNX and validating inference consistency.

## YOLO ONNX export

Run the YOLO export script from the repository root:

```bash
python examples/yolo_onnx/export_onnx.py
```

The script creates `examples/yolo_onnx/artifacts/` before exporting, temporarily switches the working directory to that folder for the Ultralytics export call, and writes the final model to `examples/yolo_onnx/artifacts/yolov8n.onnx`.
