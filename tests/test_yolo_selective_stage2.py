"""CPU-only tests for Stage-2 model.22 selective-FP16 grouping."""
from __future__ import annotations

import pytest

from examples.yolo_int8.run_selective_fp16_stage2 import (
    STAGE2_GROUP_ORDER,
    default_stage2_groups,
    parse_node_group,
    resolve_custom_groups,
)


def sample_block22_names() -> list[str]:
    return [
        "/model.22/Add_1",
        "/model.22/cv2.0/cv2.0.0/conv/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.0/cv3.0.0/conv/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
        "/model.22/dfl/Reshape",
        "/model.22/dfl/conv/Conv",
    ]


def test_default_groups_partition_model22_exactly_once():
    names = sample_block22_names()
    groups = default_stage2_groups(names)
    assert tuple(groups) == STAGE2_GROUP_ORDER
    assert groups["block22_cv2"] == [
        "/model.22/cv2.0/cv2.0.0/conv/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
    ]
    assert groups["block22_cv3"] == [
        "/model.22/cv3.0/cv3.0.0/conv/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
    ]
    assert groups["block22_dfl"] == [
        "/model.22/dfl/Reshape",
        "/model.22/dfl/conv/Conv",
    ]
    assert groups["block22_other"] == ["/model.22/Add_1"]
    flattened = [node for label in STAGE2_GROUP_ORDER for node in groups[label]]
    assert sorted(flattened) == sorted(names)
    assert len(flattened) == len(set(flattened))


def test_default_groups_reject_empty_partition_members():
    with pytest.raises(ValueError, match="resolved empty"):
        default_stage2_groups(["/model.22/cv2.0/Conv", "/model.22/cv3.0/Conv", "/model.22/Add"])


def test_parse_node_group_and_custom_resolution():
    assert parse_node_group(r"cv2=^/model\.22/cv2\.") == ("cv2", r"^/model\.22/cv2\.")
    groups = resolve_custom_groups(sample_block22_names(), [("only_dfl", r"^/model\.22/dfl/")])
    assert groups["only_dfl"] == ["/model.22/dfl/Reshape", "/model.22/dfl/conv/Conv"]


def test_custom_group_zero_match_is_rejected():
    with pytest.raises(ValueError, match="matched zero"):
        resolve_custom_groups(sample_block22_names(), [("missing", r"does-not-exist")])
