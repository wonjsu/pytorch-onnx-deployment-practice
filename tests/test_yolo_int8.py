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
