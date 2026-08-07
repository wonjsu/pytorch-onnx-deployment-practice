"""CPU-only tests for selective-FP16 sensitivity infrastructure."""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from examples.yolo_int8.inspect_yolo_node_groups import (
    block_id_from_node_name,
    exact_nodes_for_blocks,
    group_nodes,
)
from examples.yolo_int8.quantize_int8_modelopt import quantize, resolve_excluded_node_names
from examples.yolo_int8.run_selective_fp16_sensitivity import (
    DEFAULT_GROUPS,
    build_quantize_command,
    exclusion_patterns,
    parse_args,
    parse_group,
    resume_identity,
)


def synthetic_model(path: Path) -> Path:
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["a"], name="/model.10/conv/Conv"),
        helper.make_node("Relu", ["a"], ["b"], name="/model.2/relu"),
        helper.make_node("Add", ["b", "bias"], ["y"], name="post/Add[0]"),
        helper.make_node("Identity", ["y"], ["z"], name=""),
    ]
    graph = helper.make_graph(
        nodes, "test", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])],
        initializer=[helper.make_tensor("w", TensorProto.FLOAT, [1], [1]),
                     helper.make_tensor("bias", TensorProto.FLOAT, [1], [0])],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def test_parses_model_block_names_without_assuming_all_names_are_usable():
    assert block_id_from_node_name("/model.0/conv/Conv") == 0
    assert block_id_from_node_name("prefix/model.22/output") == 22
    assert block_id_from_node_name("model.3") == 3
    assert block_id_from_node_name("postprocess/model.3x/Add") is None
    assert block_id_from_node_name("") is None


def test_grouping_is_integer_sorted_and_reports_ungrouped_nodes(tmp_path: Path):
    model = onnx.load(synthetic_model(tmp_path / "model.onnx"))
    report = group_nodes(model.graph.node)
    assert [item["block_id"] for item in report["blocks"]] == [2, 10]
    assert report["blocks"][1]["quantization_relevant_node_names"] == ["/model.10/conv/Conv"]
    assert report["ungrouped_named_nodes"]["node_names"] == ["post/Add[0]"]
    assert exact_nodes_for_blocks(report, [10, 2]) == ["/model.10/conv/Conv", "/model.2/relu"]


def test_group_parser_and_default_stage_one_groups(tmp_path: Path):
    assert parse_group("blocks_00_02=0,1,2") == ("blocks_00_02", (0, 1, 2))
    args = parse_args([
        "--onnx-path", "model.onnx", "--calibration-data-dir", "cal", "--eval-images-dir", "images",
        "--eval-annotation-path", "instances.json", "--runtime-python", "runtime-python",
        "--modelopt-python", "modelopt-python", "--output-dir", str(tmp_path),
    ])
    assert tuple(args.groups) == DEFAULT_GROUPS
    assert len(DEFAULT_GROUPS) == 5


def test_exact_node_exclusion_regexes_are_escaped_and_zero_match_rejected(tmp_path: Path):
    name = "/model.2/Add[0]"
    pattern = exclusion_patterns([name])[0]
    assert re.fullmatch(pattern, name)
    assert not re.fullmatch(pattern, "/modelX2/Add0")
    model = synthetic_model(tmp_path / "model.onnx")
    assert resolve_excluded_node_names(model, [r"^/model\.2/"]) == ["/model.2/relu"]
    with pytest.raises(ValueError, match="zero named ONNX nodes"):
        resolve_excluded_node_names(model, [r"typo-does-not-exist"])


def test_quantize_command_interpreter_routing_and_optional_exclusions(tmp_path: Path):
    modelopt = tmp_path / "modelopt-python"
    command = build_quantize_command(tmp_path / "fp32.onnx", tmp_path / "out.onnx", tmp_path / "cal", modelopt,
                                     [r"^/model\.0/Conv$"])
    assert command[0] == str(modelopt)
    assert command[command.index("--nodes-to-exclude") + 1] == r"^/model\.0/Conv$"
    assert "--nodes-to-exclude" not in build_quantize_command(
        tmp_path / "fp32.onnx", tmp_path / "out.onnx", tmp_path / "cal", modelopt)


def test_quantizer_only_passes_nodes_to_exclude_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = synthetic_model(tmp_path / "model.onnx")
    calibration = tmp_path / "cal"
    calibration.mkdir()
    import numpy as np
    np.savez(calibration / "batch_0000.npz", x=np.zeros((1,), dtype=np.float32))
    from examples.yolo_int8.generate_calibration_data import sha256
    (calibration / "metadata.json").write_text(json.dumps({
        "fp32_onnx_sha256": sha256(source), "count": 1, "seed": 0, "image_ids": [1]
    }))
    calls = []

    def fake_quantize(onnx_path, output_path, quantize_mode, calibration_method,
                      high_precision_dtype, calibration_data_reader, nodes_to_exclude=None):
        calls.append({"nodes_to_exclude": nodes_to_exclude, "mode": quantize_mode,
                      "dtype": high_precision_dtype})
        Path(output_path).write_bytes(source.read_bytes())

    import examples.yolo_int8.quantize_int8_modelopt as module
    monkeypatch.setattr(module.importlib, "import_module", lambda _: types.SimpleNamespace(quantize=fake_quantize))
    monkeypatch.setattr(module, "inspect_model", lambda _: {})
    monkeypatch.setitem(sys.modules, "modelopt", types.SimpleNamespace(__version__="test"))
    quantize(source, tmp_path / "plain.onnx", calibration, "entropy")
    quantize(source, tmp_path / "excluded.onnx", calibration, "entropy", [r"^/model\.2/relu$"])
    assert calls[0]["nodes_to_exclude"] is None
    assert calls[1]["nodes_to_exclude"] == [r"^/model\.2/relu$"]
    assert all(call["mode"] == "int8" and call["dtype"] == "fp16" for call in calls)


def test_resume_identity_changes_with_exclusion_set():
    common = ("source", "modelopt", {"count": 128, "seed": 0},)
    first = resume_identity(*common, [0], ["/model.0/Conv"], "full")
    second = resume_identity(*common, [1], ["/model.1/Conv"], "full")
    assert first != second


def test_sensitivity_runner_source_has_no_performance_benchmark_invocation():
    source = Path("examples/yolo_int8/run_selective_fp16_sensitivity.py").read_text(encoding="utf-8")
    forbidden = "benchmark" + "_precision"
    assert forbidden not in source
