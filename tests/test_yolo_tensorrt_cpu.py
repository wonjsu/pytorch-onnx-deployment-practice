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


def test_tensorrt_dtype_mapping_without_tensorrt_install() -> None:
    from types import SimpleNamespace
    from examples.yolo_tensorrt.tensorrt_runner import trt_dtype_to_torch

    fake_trt = SimpleNamespace(float32="f32", float16="f16", int8="i8", int32="i32", bool="bool")
    fake_torch = SimpleNamespace(float32=1, float16=2, int8=3, int32=4, bool=5)
    assert trt_dtype_to_torch(fake_trt.float32, fake_trt, fake_torch) == fake_torch.float32
    assert trt_dtype_to_torch(fake_trt.float16, fake_trt, fake_torch) == fake_torch.float16
    assert trt_dtype_to_torch(fake_trt.int8, fake_trt, fake_torch) == fake_torch.int8
    assert trt_dtype_to_torch(fake_trt.int32, fake_trt, fake_torch) == fake_torch.int32
    assert trt_dtype_to_torch(fake_trt.bool, fake_trt, fake_torch) == fake_torch.bool


def test_build_cli_validation(tmp_path) -> None:
    import pytest
    from examples.yolo_tensorrt.build_engine import parse_args, positive_workspace_gb

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    args = parse_args(["--onnx-path", str(onnx), "--engine-path", str(tmp_path / "model.engine"), "--tf32", "on"])
    assert args.tf32 == "on" and args.workspace_gb == 1.0
    with pytest.raises(SystemExit):
        parse_args(["--onnx-path", str(onnx), "--engine-path", str(tmp_path / "model.engine"), "--tf32", "maybe"])
    with pytest.raises(Exception):
        positive_workspace_gb("0")


def test_engine_path_and_static_metadata_validation(tmp_path) -> None:
    import pytest
    from examples.yolo_tensorrt.tensorrt_runner import validate_engine_path, validate_static_shape

    engine = tmp_path / "valid.engine"
    engine.write_bytes(b"engine")
    assert validate_engine_path(engine) == engine
    assert validate_static_shape([1, 3, 640, 640], "images") == (1, 3, 640, 640)
    with pytest.raises(ValueError, match="non-static"):
        validate_static_shape([1, 3, -1, -1], "images")
    empty = tmp_path / "empty.engine"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_engine_path(empty)


def test_builder_parameter_converters() -> None:
    import argparse
    import pytest
    from examples.yolo_tensorrt.build_engine import max_aux_streams, max_num_tactics, optimization_level, positive_integer

    assert [optimization_level(str(value)) for value in (0, 5)] == [0, 5]
    assert positive_integer("4") == 4
    assert [max_num_tactics(str(value)) for value in (-1, 1, 32)] == [-1, 1, 32]
    assert [max_aux_streams(value) for value in ("auto", "0", "2")] == ["auto", 0, 2]
    for converter, value in ((optimization_level, "6"), (positive_integer, "0"),
                             (max_num_tactics, "0"), (max_num_tactics, "-2"),
                             (max_aux_streams, "-1")):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            converter(value)


def test_sweep_configurations_paths_and_resume_metadata(tmp_path) -> None:
    from examples.yolo_tensorrt.run_builder_sweep import engine_path, metadata_matches, predefined_configurations

    first = predefined_configurations(3.0); second = predefined_configurations(3.0)
    assert first == second and first is not second
    assert next(item for item in first if item["label"] == "workspace1")["workspace_gb"] == 1.0
    paths = [engine_path(tmp_path, item["label"], "int8") for item in first]
    assert len(paths) == len(set(paths))
    settings = first[0]
    metadata = {"onnx_sha256": "abc", "workspace_bytes": int(3.0 * 2**30),
                **{key: settings[key] for key in ("builder_optimization_level", "avg_timing_iterations",
                                                   "max_num_tactics", "max_aux_streams", "max_aux_streams_mode")}}
    assert metadata_matches(metadata, "abc", settings)
    assert not metadata_matches({**metadata, "max_num_tactics": 8}, "abc", settings)


def test_sweep_summary_percent_changes_and_sorting() -> None:
    import pytest
    from examples.yolo_tensorrt.run_builder_sweep import create_summary

    configs = [{"label": "reference"}, {"label": "faster"}]
    fields = ("h2d_ms", "gpu_compute_ms", "d2h_ms", "gpu_total_ms", "host_latency_ms", "throughput_fps")
    def result(value):
        return {field: {"all_iterations": {"mean": value, "median": value, "p95": value}} for field in fields}
    benchmark = {"results": {"engine": {"reference": {"aggregate": result(10.0)},
                                           "faster": {"aggregate": result(8.0)}}}}
    builds = {label: {"engine_build_time_seconds": 1, "engine_file_size_bytes": 2} for label in ("reference", "faster")}
    summary = create_summary(configs, builds, benchmark)
    assert summary["winner_label"] == "faster"
    assert [row["label"] for row in summary["configurations"]] == ["faster", "reference"]
    assert summary["configurations"][0]["percent_difference_from_reference"]["compute_median_ms"] == pytest.approx(-20)
