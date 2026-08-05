"""CPU-only coverage for calibration, Q/DQ inspection, and orchestration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from examples.yolo_int8.generate_calibration_data import select_images
from examples.yolo_int8.quantize_int8_modelopt import LazyNpzCalibrationDataReader
from tools.run_precision_experiments import artifact_is_current, build_commands, parse_args, sha256

def test_calibration_selection_is_deterministic(tmp_path:Path):
    annotation=tmp_path/"instances.json";annotation.write_text(json.dumps({"images":[{"id":i,"file_name":f"{i}.jpg"} for i in range(20)]}))
    assert select_images(annotation,5,7)==select_images(annotation,5,7)
    assert len({x["id"] for x in select_images(annotation,5,7)})==5

def test_lazy_npz_reader_shape_dtype_and_rewind(tmp_path:Path):
    value=np.zeros((1,3,640,640),dtype=np.float32);np.savez(tmp_path/"batch_0000.npz",images=value)
    reader=LazyNpzCalibrationDataReader(tmp_path);batch=reader.get_next()
    assert list(batch)==["images"] and batch["images"].shape==(1,3,640,640) and batch["images"].dtype==np.float32
    assert reader.get_next() is None;reader.rewind();assert reader.get_next() is not None

def test_resume_requires_matching_source_hash(tmp_path:Path):
    source=tmp_path/"source.onnx";source.write_bytes(b"source");artifact=tmp_path/"model.engine";artifact.write_bytes(b"engine")
    metadata=tmp_path/"model.engine.json";metadata.write_text(json.dumps({"source_onnx_sha256":sha256(source)}))
    assert artifact_is_current(artifact,metadata,source)
    source.write_bytes(b"changed");assert not artifact_is_current(artifact,metadata,source)

def test_smoke_and_full_protocol_arguments():
    common=["--stage","int8","--calibration-images-dir","train","--calibration-annotation-path","train.json"]
    smoke=build_commands(parse_args(common));full=build_commands(parse_args(common+["--scope","full"]))
    smoke_bench=smoke[-1][1];full_bench=full[-1][1]
    assert "100" in smoke_bench and "2" in smoke_bench
    assert "5000" in full_bench and "500" in full_bench and "4" in full_bench

def test_int8_never_silently_defaults_calibration_paths():
    with pytest.raises(ValueError,match="explicit calibration"):
        build_commands(parse_args(["--stage","int8"]))

from examples.yolo_int8.run_calibration_matrix import (
    BUILDER_SETTINGS,
    build_benchmark_command,
    build_engine_command,
    build_quantize_command,
    calibration_image_ids_hash,
    is_nested,
    labels_are_unique,
    matrix_configurations,
    metadata_matches,
    select_master_images,
    subset_ids,
)


def test_matrix_master_selection_and_nested_prefixes(tmp_path: Path):
    annotation = tmp_path / "instances.json"
    annotation.write_text(json.dumps({"images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(1100)]}))
    master_a = select_master_images(annotation, 1024, 0)
    master_b = select_master_images(annotation, 1024, 0)
    assert master_a == master_b
    assert is_nested(master_a, [128, 256, 512, 1024])
    assert subset_ids(master_a, 128) == subset_ids(master_a, 256)[:128]


def test_matrix_configuration_ordering_labels_and_paths():
    configs = matrix_configurations(["entropy", "max"], [128, 256, 512, 1024], 0)
    assert [config.label for config in configs] == [
        "entropy_128", "entropy_256", "entropy_512", "entropy_1024",
        "max_128", "max_256", "max_512", "max_1024",
    ]
    assert labels_are_unique(configs)
    assert [config.relative_artifact_dir.as_posix() for config in configs] == [config.label for config in configs]


def test_matrix_command_construction_uses_explicit_qdq_and_final_builder_settings(tmp_path: Path):
    quant = build_quantize_command(tmp_path / "fp32.onnx", tmp_path / "out.onnx", tmp_path / "cal", "entropy")
    assert "examples.yolo_int8.quantize_int8_modelopt" in quant
    assert "--calibration-method" in quant and "entropy" in quant
    engine = build_engine_command(tmp_path / "out.onnx", tmp_path / "out.engine")
    for flag, value in {
        "--model-precision": "int8", "--tf32": "off", "--workspace-gb": "2",
        "--builder-optimization-level": "5", "--avg-timing-iterations": "8",
        "--max-num-tactics": "-1", "--max-aux-streams": "1",
    }.items():
        assert engine[engine.index(flag) + 1] == value
    assert BUILDER_SETTINGS["model_precision"] == "int8"


def test_matrix_resume_metadata_matching(tmp_path: Path):
    metadata = tmp_path / "artifact.json"
    metadata.write_text(json.dumps({"source_onnx_sha256": "abc", "calibration_count": 256, "seed": 0}))
    assert metadata_matches(metadata, {"source_onnx_sha256": "abc", "calibration_count": 256, "seed": 0})
    assert not metadata_matches(metadata, {"source_onnx_sha256": "abc", "calibration_count": 128, "seed": 0})
    assert not metadata_matches(tmp_path / "missing.json", {"source_onnx_sha256": "abc"})


def test_failure_continuation_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import examples.yolo_int8.run_calibration_matrix as matrix
    calls = []
    def fake_run(cmd, log):
        calls.append(cmd)
        if "bad.onnx" in " ".join(map(str, cmd)):
            raise RuntimeError("boom")
        return 0.1
    monkeypatch.setattr(matrix, "run_command", fake_run)
    assert calibration_image_ids_hash([1, 2, 3]) == calibration_image_ids_hash([1, 2, 3])


def test_eight_engine_full_benchmark_uses_nine_rounds_and_discards_one(tmp_path: Path):
    engines = [(f"e{i}", tmp_path / f"e{i}.engine") for i in range(8)]
    cmd = build_benchmark_command(engines, tmp_path / "benchmark.json", tmp_path / "benchmark.csv", "full")
    assert cmd[cmd.index("--engine-rounds") + 1] == "9"
    assert cmd[cmd.index("--discard-rounds") + 1] == "1"
    assert cmd[cmd.index("--engine-warmup") + 1] == "50"
    assert cmd[cmd.index("--engine-iterations") + 1] == "500"
    assert cmd.count("--engine") == 8
