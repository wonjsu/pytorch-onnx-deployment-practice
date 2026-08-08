"""CPU-only tests for Stage-3 complement selective-FP16 grouping."""
from __future__ import annotations

from examples.yolo_int8.run_selective_fp16_stage3 import GROUP_ORDER, complement_variants, partition_block22


def block22_names() -> list[str]:
    return [
        "/model.22/cv2.0/cv2.0.0/conv/Conv",
        "/model.22/cv3.0/cv3.0.0/conv/Conv",
        "/model.22/dfl/conv/Conv",
        "/model.22/Add_1",
    ]


def test_partition_block22_assigns_every_node_once():
    groups = partition_block22(block22_names())
    assert tuple(groups) == GROUP_ORDER
    assert groups["cv2"] == ["/model.22/cv2.0/cv2.0.0/conv/Conv"]
    assert groups["cv3"] == ["/model.22/cv3.0/cv3.0.0/conv/Conv"]
    assert groups["dfl"] == ["/model.22/dfl/conv/Conv"]
    assert groups["other"] == ["/model.22/Add_1"]
    flattened = [node for group in GROUP_ORDER for node in groups[group]]
    assert sorted(flattened) == sorted(block22_names())


def test_complement_variants_leave_exactly_one_group_int8():
    groups = partition_block22(block22_names())
    variants = complement_variants(groups)
    assert set(variants) == {
        "leave_cv2_int8", "leave_cv3_int8", "leave_dfl_int8", "leave_other_int8"
    }
    all_nodes = set(block22_names())
    for group in GROUP_ORDER:
        fp16 = set(variants[f"leave_{group}_int8"])
        assert fp16 == all_nodes - set(groups[group])
        assert not (fp16 & set(groups[group]))


def test_realistic_counts_produce_expected_complement_sizes():
    groups = {
        "cv2": [f"cv2_{i}" for i in range(21)],
        "cv3": [f"cv3_{i}" for i in range(21)],
        "dfl": [f"dfl_{i}" for i in range(5)],
        "other": [f"other_{i}" for i in range(19)],
    }
    variants = complement_variants(groups)
    assert len(variants["leave_cv2_int8"]) == 45
    assert len(variants["leave_cv3_int8"]) == 45
    assert len(variants["leave_dfl_int8"]) == 61
    assert len(variants["leave_other_int8"]) == 47
