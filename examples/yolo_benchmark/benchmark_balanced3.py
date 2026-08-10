"""Run a three-engine benchmark with fully balanced six-permutation order.

This wrapper reuses benchmark_precision.py but replaces its cyclic rotation with a
three-engine schedule that balances both execution position and immediate
predecessor across the retained rounds.

Protocol enforced here:
- exactly 3 engines
- exactly 1 discarded round
- exactly 7 total rounds for every active mode
- retained rounds use all 6 permutations exactly once:
    A B C
    B C A
    C A B
    A C B
    C B A
    B A C

The discarded first round uses the supplied engine order and provides the normal
engine warm-up path used by benchmark_precision.py.
"""
from __future__ import annotations

import sys
from collections import Counter
from typing import Sequence

from examples.yolo_benchmark import benchmark_precision


PERMUTATION_INDICES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
    (0, 2, 1),
    (2, 1, 0),
    (1, 0, 2),
)


def balanced3_order(labels: Sequence[str], round_index: int) -> list[str]:
    """Return discarded-round order, then one of all six 3-engine permutations."""
    if len(labels) != 3:
        raise ValueError(f"balanced3 requires exactly 3 engines; found {len(labels)}")
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if round_index == 0:
        return list(labels)
    indices = PERMUTATION_INDICES[(round_index - 1) % len(PERMUTATION_INDICES)]
    return [labels[index] for index in indices]


def _option_value(argv: Sequence[str], name: str, default: str | None = None) -> str | None:
    try:
        index = list(argv).index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        raise ValueError(f"{name} requires a value")
    return argv[index + 1]


def validate_protocol(argv: Sequence[str]) -> None:
    """Reject invocations that would break the intended one-discard + six-retained design."""
    engine_count = sum(1 for token in argv if token == "--engine")
    if engine_count != 3:
        raise ValueError(f"balanced3 requires exactly 3 --engine arguments; found {engine_count}")

    mode = _option_value(argv, "--mode")
    if mode not in {"engine", "pipeline", "both"}:
        raise ValueError("--mode must be engine, pipeline, or both")

    discard_rounds = int(_option_value(argv, "--discard-rounds", "1"))
    if discard_rounds != 1:
        raise ValueError("balanced3 requires --discard-rounds 1")

    engine_rounds = int(_option_value(argv, "--engine-rounds", "4"))
    pipeline_rounds = int(_option_value(argv, "--pipeline-rounds", "4"))
    if mode in {"engine", "both"} and engine_rounds != 7:
        raise ValueError("balanced3 requires --engine-rounds 7 for engine mode")
    if mode in {"pipeline", "both"} and pipeline_rounds != 7:
        raise ValueError("balanced3 requires --pipeline-rounds 7 for pipeline mode")


def schedule_diagnostics(labels: Sequence[str]) -> dict[str, Counter]:
    """Return retained position and predecessor counts for validation/tests."""
    if len(labels) != 3:
        raise ValueError("schedule diagnostics require exactly 3 labels")
    position_counts: Counter = Counter()
    predecessor_counts: Counter = Counter()
    for round_index in range(1, 7):
        order = balanced3_order(labels, round_index)
        for position, label in enumerate(order, 1):
            position_counts[(label, position)] += 1
        for previous, current in zip(order, order[1:]):
            predecessor_counts[(previous, current)] += 1
    return {"positions": position_counts, "predecessors": predecessor_counts}


def main(argv: Sequence[str] | None = None) -> dict:
    args = list(sys.argv[1:] if argv is None else argv)
    validate_protocol(args)
    benchmark_precision.rotating_order = balanced3_order
    return benchmark_precision.main(args)


if __name__ == "__main__":
    main()
