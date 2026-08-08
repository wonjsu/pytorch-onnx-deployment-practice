"""Run Stage-3 complement ablations for YOLO INT8 selective-FP16.

Stage 3 starts from the Stage-1 model.22 winner and leaves exactly one model.22
subgroup quantized while the other three subgroups use FP16 fallback. Accuracy is
measured only; latency benchmarking is intentionally deferred.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.yolo_int8.generate_calibration_data import sha256
from examples.yolo_int8.inspect_yolo_node_groups import exact_nodes_for_blocks, inspect_onnx
from examples.yolo_int8.run_calibration_matrix import (
    BUILDER_SETTINGS,
    build_accuracy_command,
    build_engine_command,
    expected_builder_metadata,
    metadata_matches,
    query_modelopt_version,
    validate_python_interpreter,
)
from examples.yolo_int8.run_selective_fp16_sensitivity import METRICS, SMOKE_LIMIT, build_quantize_command, exclusion_patterns

GROUP_ORDER = ("cv2", "cv3", "dfl", "other")


def partition_block22(block22_names: Sequence[str]) -> dict[str, list[str]]:
    names = sorted(set(block22_names))
    groups = {
        "cv2": [name for name in names if name.startswith("/model.22/cv2.")],
        "cv3": [name for name in names if name.startswith("/model.22/cv3.")],
        "dfl": [name for name in names if name.startswith("/model.22/dfl/")],
    }
    assigned = set().union(*(set(values) for values in groups.values()))
    groups["other"] = [name for name in names if name not in assigned]
    empty = [name for name in GROUP_ORDER if not groups[name]]
    if empty:
        raise ValueError(f"Stage-3 groups resolved empty from model.22: {empty}")
    flattened = [node for group in GROUP_ORDER for node in groups[group]]
    if len(flattened) != len(names) or set(flattened) != set(names):
        raise RuntimeError("Stage-3 groups must partition every named model.22 node exactly once")
    return groups


def complement_variants(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return FP16 exclusion sets, leaving one subgroup INT8 in each variant."""
    variants: dict[str, list[str]] = {}
    for int8_group in GROUP_ORDER:
        fp16_names = sorted(
            node for group in GROUP_ORDER if group != int8_group for node in groups[group]
        )
        variants[f"leave_{int8_group}_int8"] = fp16_names
    return variants


def run_command(command: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")
    return time.perf_counter() - started


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def artifact_identity(source_sha: str, modelopt_version: str, calibration_metadata: dict[str, Any], exact_names: Sequence[str]) -> dict[str, Any]:
    return {
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy",
        "calibration_metadata": calibration_metadata,
        "calibration_count": calibration_metadata.get("count", calibration_metadata.get("calibration_count")),
        "calibration_seed": calibration_metadata.get("seed"),
        "excluded_exact_node_names": list(exact_names),
        "builder_settings": BUILDER_SETTINGS,
    }


def evaluation_identity(artifact: dict[str, Any], scope: str) -> dict[str, Any]:
    return {
        **artifact,
        "scope": scope,
        "evaluation_settings": {
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
            "limit": None if scope == "full" else SMOKE_LIMIT,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--calibration-data-dir", type=Path, required=True)
    parser.add_argument("--eval-images-dir", type=Path, required=True)
    parser.add_argument("--eval-annotation-path", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--modelopt-python", type=Path, required=True)
    parser.add_argument("--baseline-accuracy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_python_interpreter(args.runtime_python, "runtime")
    validate_python_interpreter(args.modelopt_python, "ModelOpt")
    for path, description, directory in (
        (args.onnx_path, "source ONNX", False),
        (args.calibration_data_dir, "calibration data", True),
        (args.eval_images_dir, "evaluation images", True),
        (args.eval_annotation_path, "evaluation annotations", False),
        (args.baseline_accuracy_json, "baseline accuracy", False),
    ):
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f"{description} not found: {path}")

    calibration_metadata = json.loads((args.calibration_data_dir / "metadata.json").read_text(encoding="utf-8"))
    calibration_count = calibration_metadata.get("count", calibration_metadata.get("calibration_count"))
    calibration_seed = calibration_metadata.get("seed")
    if calibration_count != 128 or calibration_seed != 0:
        raise ValueError(
            "Stage 3 requires entropy_128 calibration metadata with count=128 and seed=0; "
            f"found count={calibration_count!r}, seed={calibration_seed!r}"
        )

    graph_report = inspect_onnx(args.onnx_path)
    block22_names = exact_nodes_for_blocks(graph_report, [22])
    if not block22_names:
        raise ValueError("model.22 resolved to zero named ONNX nodes")
    groups = partition_block22(block22_names)
    variants = complement_variants(groups)

    source_sha = sha256(args.onnx_path)
    modelopt_version = query_modelopt_version(args.modelopt_python)
    artifact_root = args.output_dir / "artifacts"
    scope_root = args.output_dir / "results" / args.scope
    if args.force:
        if scope_root.exists():
            shutil.rmtree(scope_root)
        for label in variants:
            candidate = artifact_root / label
            if candidate.exists():
                shutil.rmtree(candidate)
    scope_root.mkdir(parents=True, exist_ok=True)

    baseline_accuracy = json.loads(args.baseline_accuracy_json.read_text(encoding="utf-8"))
    baseline_ap = baseline_accuracy.get("AP50:95")
    manifest: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 3 model.22 complement ablation",
        "scope": args.scope,
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy",
        "calibration_metadata": calibration_metadata,
        "builder_settings": BUILDER_SETTINGS,
        "evaluation_settings": {"conf_threshold": 0.001, "iou_threshold": 0.7, "limit": None if args.scope == "full" else SMOKE_LIMIT},
        "performance_benchmarking": False,
        "artifact_root": str(artifact_root),
        "scope_result_root": str(scope_root),
        "groups": [{"name": name, "exact_node_count": len(groups[name]), "exact_node_names": groups[name]} for name in GROUP_ORDER],
        "variants": [],
    }
    rows: list[dict[str, Any]] = []

    for label, names in variants.items():
        left_int8 = label.removeprefix("leave_").removesuffix("_int8")
        artifact_dir = artifact_root / label
        result_dir = scope_root / label
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        patterns = exclusion_patterns(names)
        artifact_meta = artifact_identity(source_sha, modelopt_version, calibration_metadata, names)
        eval_meta = evaluation_identity(artifact_meta, args.scope)
        row: dict[str, Any] = {
            "label": label,
            "status": "success",
            "failure_stage": None,
            "scope": args.scope,
            "smoke_result": args.scope == "smoke",
            "left_int8_group": left_int8,
            "fp16_group_names": [group for group in GROUP_ORDER if group != left_int8],
            "excluded_exact_node_count": len(names),
            "excluded_exact_node_names": names,
        }
        try:
            onnx_out = artifact_dir / "yolov8n_int8_qdq.onnx"
            qmeta = Path(str(onnx_out) + ".conversion.json")
            engine = artifact_dir / "yolov8n_int8.engine"
            logs = result_dir / "logs"
            quant_expected = {
                "source_sha256": source_sha,
                "modelopt_version": modelopt_version,
                "calibration_method": "entropy",
                "calibration_metadata": calibration_metadata,
                "calibration_count": calibration_count,
                "calibration_seed": calibration_seed,
                "calibration_image_ids": calibration_metadata.get("image_ids", calibration_metadata.get("calibration_image_ids")),
                "nodes_to_exclude_patterns": patterns,
                "resolved_excluded_node_names": names,
            }
            row["failure_stage"] = "quantization"
            if not args.resume or not onnx_out.is_file() or not metadata_matches(qmeta, quant_expected):
                row["quantization_duration_seconds"] = run_command(
                    build_quantize_command(args.onnx_path, onnx_out, args.calibration_data_dir, args.modelopt_python, patterns),
                    logs / "quantize.log",
                )
            row["failure_stage"] = "engine_build"
            engine_meta_path = Path(str(engine) + ".json")
            if not args.resume or not engine.is_file() or not metadata_matches(engine_meta_path, expected_builder_metadata(sha256(onnx_out))):
                row["engine_build_duration_seconds"] = run_command(
                    build_engine_command(onnx_out, engine, args.runtime_python), logs / "build_engine.log"
                )
            accuracy_path = result_dir / "accuracy.json"
            identity_path = result_dir / "evaluation_metadata.json"
            row["failure_stage"] = "accuracy_evaluation"
            accuracy_current = args.resume and accuracy_path.is_file() and metadata_matches(identity_path, eval_meta)
            if not accuracy_current:
                run_command(
                    build_accuracy_command(
                        engine, args.eval_images_dir, args.eval_annotation_path,
                        result_dir / "predictions.json", accuracy_path,
                        None if args.scope == "full" else SMOKE_LIMIT, args.runtime_python,
                    ),
                    logs / "accuracy.log",
                )
            accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
            row.update({metric: accuracy.get(metric) for metric in METRICS})
            row["ONNX SHA-256"] = sha256(onnx_out)
            row["engine SHA-256"] = sha256(engine)
            row["delta_AP50:95_vs_int8_baseline"] = None if baseline_ap is None or row.get("AP50:95") is None else row["AP50:95"] - baseline_ap
            write_json(identity_path, eval_meta)
            row["failure_stage"] = None
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)})
        rows.append(row)
        manifest["variants"] = rows
        write_json(scope_root / "sensitivity_manifest.json", manifest)

    ordered = sorted(rows, key=lambda row: (row.get("AP50:95") is None, -(row.get("AP50:95") or 0), row["label"]))
    best = next((row["label"] for row in ordered if row.get("AP50:95") is not None), None)
    summary = {
        "stage": "Stage 3 model.22 complement ablation",
        "scope": args.scope,
        "smoke_warning": "Correctness-only subset; not final accuracy." if args.scope == "smoke" else None,
        "int8_baseline": {metric: baseline_accuracy.get(metric) for metric in METRICS},
        "best_variant": best,
        "winner_note": "The best Stage-3 variant remains an accuracy candidate until a balanced latency benchmark is run.",
        "variants": ordered,
    }
    write_json(scope_root / "sensitivity_summary.json", summary)
    fields = list(dict.fromkeys(key for row in ordered for key in row)) if ordered else ["label"]
    with (scope_root / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["variants"] = rows
    write_json(scope_root / "sensitivity_manifest.json", manifest)
    return summary


if __name__ == "__main__":
    main()
