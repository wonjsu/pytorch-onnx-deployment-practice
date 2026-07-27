"""CPU-only tests for the precision benchmark's pure helpers."""

import json

import pytest

from examples.yolo_benchmark.benchmark_precision import (
    calculate_pipeline_times,
    parse_args,
    parse_engine_spec,
    query_gpu_state,
    rotating_order,
    validate_engine_specs,
)
from examples.yolo_benchmark.statistics_utils import aggregate_rounds, describe, percentile


def test_engine_spec_and_validation(tmp_path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    assert parse_engine_spec(f"fp32={engine}")[0] == "fp32"
    assert validate_engine_specs([("fp32", engine)])[0][1] == engine.resolve()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_engine_specs([("same", engine), ("same", engine)])
    with pytest.raises(FileNotFoundError):
        validate_engine_specs([("missing", tmp_path / "missing.engine")])
    empty = tmp_path / "empty.engine"; empty.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_engine_specs([("empty", empty)])
    with pytest.raises(Exception):
        parse_engine_spec("invalid")


def test_round_and_discard_validation(tmp_path) -> None:
    engine = tmp_path / "model.engine"; engine.write_bytes(b"engine")
    base = ["--mode", "engine", "--engine", f"fp32={engine}"]
    assert parse_args(base + ["--engine-rounds", "3", "--discard-rounds", "1"]).engine_rounds == 3
    with pytest.raises(SystemExit):
        parse_args(base + ["--engine-rounds", "0"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--engine-rounds", "3", "--discard-rounds", "3"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--engine-iterations", "0"])


def test_statistics_and_discarded_rounds() -> None:
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    stats = describe([1, 2, 3])
    assert stats["mean"] == stats["median"] == 2
    assert stats["standard_deviation"] == pytest.approx((2 / 3) ** .5)
    rounds = [{"discarded": True, "raw": [{"x": 100}]},
              {"discarded": False, "raw": [{"x": 1}, {"x": 3}]}]
    aggregate = aggregate_rounds(rounds, ["x"])
    assert aggregate["x"]["all_iterations"]["mean"] == 2
    json.dumps(aggregate)


def test_rotation_and_pipeline_arithmetic() -> None:
    labels = ["fp32", "fp16", "int8"]
    assert rotating_order(labels, 0) == labels
    assert rotating_order(labels, 1) == ["fp16", "int8", "fp32"]
    assert rotating_order(labels, 2) == ["int8", "fp32", "fp16"]
    other, without_io, full = calculate_pipeline_times(19, 5, 2, 1, 3, 1, 2)
    assert (other, without_io, full) == (5, 14, 19)
    other, without_io, full = calculate_pipeline_times(1, 5, 2, 1, 3, 1, 2)
    assert other == 0 and without_io == 9 and full == 14


def test_nvidia_smi_na(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr("subprocess.run", missing)
    state = query_gpu_state()
    assert state and set(state.values()) == {"N/A"}

def test_throughput_console_unit(capsys) -> None:
    from examples.yolo_benchmark.benchmark_precision import print_summary
    stats={"all_iterations":{"mean":1.,"median":1.,"p95":1.},"round_means":{"standard_deviation":0.}}
    aggregate={field:stats for field in ("h2d_ms","gpu_compute_ms","d2h_ms","gpu_total_ms","host_latency_ms","throughput_fps")}
    print_summary("fp16","engine",aggregate)
    output=capsys.readouterr().out
    throughput_line=next(line for line in output.splitlines() if "throughput_fps:" in line)
    assert "FPS" in throughput_line and " ms" not in throughput_line
