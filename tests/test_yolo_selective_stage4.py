from __future__ import annotations

import pytest

from examples.yolo_int8.run_selective_fp16_stage4 import PAIR_VARIANTS, pair_variants


def test_pair_variants_use_expected_groups_and_sizes() -> None:
    groups = {
        "cv2": [f"cv2_{i}" for i in range(21)],
        "cv3": [f"cv3_{i}" for i in range(21)],
        "dfl": [f"dfl_{i}" for i in range(5)],
        "other": [f"other_{i}" for i in range(19)],
    }

    variants = pair_variants(groups)

    assert set(variants) == set(PAIR_VARIANTS)
    assert len(variants["fp16_cv3_dfl"]) == 26
    assert len(variants["fp16_cv2_dfl"]) == 26
    assert len(variants["fp16_cv2_cv3"]) == 42

    for label, pair in PAIR_VARIANTS.items():
        expected = sorted(node for group in pair for node in groups[group])
        assert variants[label] == expected
        assert not any(name.startswith("other_") for name in variants[label])


def test_pair_variants_reject_missing_group() -> None:
    with pytest.raises(ValueError, match="requires groups"):
        pair_variants({"cv2": ["a"], "cv3": ["b"], "dfl": ["c"]})


def test_pair_variants_reject_extra_group() -> None:
    with pytest.raises(ValueError, match="requires groups"):
        pair_variants(
            {
                "cv2": ["a"],
                "cv3": ["b"],
                "dfl": ["c"],
                "other": ["d"],
                "extra": ["e"],
            }
        )
