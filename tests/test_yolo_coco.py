"""CPU-only tests for COCO evaluation data helpers."""

import json

import numpy as np

from examples.yolo_coco.evaluate_coco import (
    clip_xyxy,
    make_prediction,
    select_image_ids,
    xyxy_to_xywh,
    yolo_class_to_coco_category_id,
)


class FakeCOCO:
    def getImgIds(self) -> list[int]:
        return [42, 7, 100, 9]


def test_yolo_class_to_coco_category_mapping() -> None:
    assert yolo_class_to_coco_category_id(0) == 1
    assert yolo_class_to_coco_category_id(11) == 13
    assert yolo_class_to_coco_category_id(79) == 90


def test_xyxy_to_xywh() -> None:
    assert xyxy_to_xywh(np.array([10, 20, 35, 55])) == [10.0, 20.0, 25.0, 35.0]


def test_bbox_clipping() -> None:
    np.testing.assert_array_equal(
        clip_xyxy(np.array([-5, 8, 120, 90]), width=100, height=80),
        [0, 8, 100, 80],
    )


def test_prediction_json_fields_and_types() -> None:
    prediction = make_prediction(7, 11, np.array([1, 2, 11, 22]), np.float32(0.5))
    assert set(prediction) == {"image_id", "category_id", "bbox", "score"}
    assert isinstance(prediction["image_id"], int)
    assert isinstance(prediction["category_id"], int)
    assert all(isinstance(value, float) for value in prediction["bbox"])
    assert isinstance(prediction["score"], float)
    json.dumps([prediction])


def test_limit_selects_only_requested_image_ids() -> None:
    assert select_image_ids(FakeCOCO(), 2) == [7, 9]
    assert len(select_image_ids(FakeCOCO(), 2)) == 2
