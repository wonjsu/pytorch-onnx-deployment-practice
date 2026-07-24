"""CPU-only tests for shared YOLO and TensorRT validation utilities."""

import numpy as np
from PIL import Image

from examples.yolo_onnx.postprocess_onnx import class_aware_nms, letterbox_preprocess, restore_original_coordinates
from examples.yolo_tensorrt.benchmark_tensorrt import percentile
from examples.yolo_tensorrt.compare_backends import Detection, match_detections


def test_letterbox_shape() -> None:
    tensor, size, ratio, pad_x, pad_y = letterbox_preprocess(Image.new("RGB", (320, 160)))
    assert tensor.shape == (1, 3, 640, 640)
    assert (size, ratio, pad_x, pad_y) == ((320, 160), 2.0, 0, 160)


def test_restore_original_coordinates() -> None:
    boxes = np.array([[0, 160, 640, 480]], dtype=np.float32)
    np.testing.assert_allclose(restore_original_coordinates(boxes, (320, 160), 2.0, 0, 160), [[0, 0, 320, 160]])


def test_class_aware_nms() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]], dtype=np.float32)
    assert class_aware_nms(boxes, np.array([.9, .8, .7]), np.array([0, 0, 1]), .5) == [0, 2]


def test_detection_matching() -> None:
    ref = [Detection(1, .9, (0, 0, 10, 10))]
    candidate = [Detection(1, .8, (1, 1, 11, 11)), Detection(2, .7, (0, 0, 10, 10))]
    matches, unmatched_ref, unmatched_candidate = match_detections(ref, candidate, .5)
    assert len(matches) == 1 and not unmatched_ref and unmatched_candidate == [1]


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 95) == 4.8
