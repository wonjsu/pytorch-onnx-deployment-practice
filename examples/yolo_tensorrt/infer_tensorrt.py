"""Run one image through the TensorRT 11.1 PyTorch-buffer runner."""

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
from examples.yolo_tensorrt.tensorrt_runner import TensorRTRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    args = parser.parse_args()
    tensor, original_size, ratio, pad_x, pad_y = letterbox_preprocess_image(args.image_path)
    with TensorRTRunner(args.engine_path) as runner:
        print(f"input: name={runner.input_name} shape={runner.metadata[runner.input_name]['shape']}")
        print(f"outputs: {[(name, runner.metadata[name]['shape']) for name in runner.output_names]}")
        raw = runner.infer(tensor)
    classes, scores, boxes = postprocess_output(
        raw.astype(np.float32), original_size, ratio, pad_x, pad_y,
        args.conf_threshold, args.iou_threshold,
    )
    print(f"detection count: {len(boxes)}")
    for class_index, score, box in zip(classes, scores, boxes):
        print(f"class={get_coco_class_name(int(class_index))} confidence={float(score):.4f} bbox={tuple(float(value) for value in box)}")


if __name__ == "__main__":
    main()
