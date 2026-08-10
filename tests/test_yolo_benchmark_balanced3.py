from __future__ import annotations

from collections import Counter

import pytest

from examples.yolo_benchmark.benchmark_balanced3 import (
    PERMUTATION_INDICES,
    balanced3_order,
    schedule_diagnostics,
    validate_protocol,
)


def test_balanced3_uses_all_six_permutations_once() -> None:
    labels = ["A", "B", "C"]
    retained = [tuple(balanced3_order(labels, round_index)) for round_index in range(1, 7)]
    assert len(set(retained)) == 6
    assert retained == [tuple(labels[index] for index in permutation) for permutation in PERMUTATION_INDICES]


def test_balanced3_balances_positions_and_directed_predecessors() -> None:
    labels = ["A", "B", "C"]
    diagnostics = schedule_diagnostics(labels)
    expected_positions = Counter({(label, position): 2 for label in labels for position in (1, 2, 3)})
    expected_predecessors = Counter({(a, b): 2 for a in labels for b in labels if a != b})
    assert diagnostics["positions"] == expected_positions
    assert diagnostics["predecessors"] == expected_predecessors


def test_protocol_requires_three_engines_one_discard_and_seven_rounds() -> None:
    valid = [
        "--mode", "both",
        "--engine", "fp16=a.engine",
        "--engine", "stage3=b.engine",
        "--engine", "stage5=c.engine",
        "--engine-rounds", "7",
        "--pipeline-rounds", "7",
        "--discard-rounds", "1",
    ]
    validate_protocol(valid)

    with pytest.raises(ValueError, match="exactly 3"):
        validate_protocol(valid[:-8])
    with pytest.raises(ValueError, match="engine-rounds 7"):
        validate_protocol(["6" if token == "7" and index == valid.index("--engine-rounds") + 1 else token for index, token in enumerate(valid)])
    with pytest.raises(ValueError, match="discard-rounds 1"):
        validate_protocol(["0" if token == "1" else token for token in valid])
