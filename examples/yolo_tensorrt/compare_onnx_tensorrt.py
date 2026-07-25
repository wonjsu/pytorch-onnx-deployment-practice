"""Compare ONNX Runtime CUDA and TensorRT 11.1 on one shared image tensor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.yolo_onnx.postprocess_onnx import (  # noqa: E402
    get_coco_class_name,
    letterbox_preprocess_image,
    postprocess_output,
)
from examples.yolo_tensorrt.compare_backends import (  # noqa: E402
    Detection,
    match_detections,
)
from examples.yolo_tensorrt.tensorrt_runner import TensorRTRunner  # noqa: E402


def _detections(raw: np.ndarray, metadata: tuple[tuple[int, int], float, int, int], conf: float, iou: float) -> list[Detection]:
    classes, scores, boxes = postprocess_output(raw, *metadata, conf, iou)
    return [Detection(int(c), float(s), tuple(float(v) for v in b)) for c, s, b in zip(classes, scores, boxes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    import torch  # Load CUDA DLLs before ORT on Windows.
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no backend fallback is allowed")
    if not args.onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    tensor, size, ratio, pad_x, pad_y = letterbox_preprocess_image(args.image_path)
    session = ort.InferenceSession(str(args.onnx_path), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime rejected CUDA; refusing CPU fallback")
    if len(session.get_inputs()) != 1:
        raise RuntimeError(f"Expected one ONNX input, got {len(session.get_inputs())}")
    ort_values = session.run(None, {session.get_inputs()[0].name: tensor})
    ort_outputs = {item.name: value for item, value in zip(session.get_outputs(), ort_values)}
    with TensorRTRunner(args.engine_path) as runner:
        trt_outputs = runner.infer_outputs(tensor)

    if list(ort_outputs) != list(trt_outputs):
        raise RuntimeError(
            "Output tensor names/order differ; refusing an implicit reorder or reshape: "
            f"ONNX={list(ort_outputs)}, TensorRT={list(trt_outputs)}"
        )
    for name in ort_outputs:
        first, second = ort_outputs[name], trt_outputs[name]
        if first.shape != second.shape:
            raise RuntimeError(f"Output {name!r} shape differs: ONNX={first.shape}, TensorRT={second.shape}; refusing reshape")
        if first.dtype != second.dtype:
            raise RuntimeError(f"Output {name!r} dtype differs: ONNX={first.dtype}, TensorRT={second.dtype}")
        difference = np.abs(first - second)
        print(f"tensor name: {name}")
        print(f"  shape: {first.shape}")
        print(f"  dtype: {first.dtype}")
        print(f"  max absolute error: {float(difference.max()):.9g}")
        print(f"  mean absolute error: {float(difference.mean()):.9g}")
        print(f"  RMSE: {float(np.sqrt(np.mean(np.square(difference)))):.9g}")
        print(f"  np.allclose: {np.allclose(first, second, atol=args.atol, rtol=args.rtol)}")
        print(f"  atol: {args.atol}; rtol: {args.rtol}")

        metadata = (size, ratio, pad_x, pad_y)
        onnx_detections = _detections(first, metadata, args.conf_threshold, args.iou_threshold)
        trt_detections = _detections(second, metadata, args.conf_threshold, args.iou_threshold)
        matches, unmatched_onnx, unmatched_trt = match_detections(onnx_detections, trt_detections, 0.5)
        print("postprocessed detections:")
        for match in matches:
            a = onnx_detections[match.reference_index]
            b = trt_detections[match.candidate_index]
            print(f"  matched detection: class_index={a.class_index} class_name={get_coco_class_name(a.class_index)} confidence={a.confidence:.6f} confidence_absolute_difference={abs(a.confidence-b.confidence):.6g} bbox={a.box} bbox_IoU={match.iou:.6f}")
        for index in unmatched_onnx:
            item = onnx_detections[index]
            print(f"  unmatched ONNX detection: class_index={item.class_index} class_name={get_coco_class_name(item.class_index)} confidence={item.confidence:.6f} bbox={item.box}")
        for index in unmatched_trt:
            item = trt_detections[index]
            print(f"  unmatched TensorRT detection: class_index={item.class_index} class_name={get_coco_class_name(item.class_index)} confidence={item.confidence:.6f} bbox={item.box}")


if __name__ == "__main__":
    main()
