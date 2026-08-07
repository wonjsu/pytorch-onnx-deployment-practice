"""Run Stage-1 YOLO INT8 selective-FP16 accuracy-sensitivity experiments.

This runner deliberately evaluates COCO accuracy only; latency benchmarking is
reserved for later balanced-candidate experiments.
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

DEFAULT_GROUPS = (
    ("blocks_00_04", (0, 1, 2, 3, 4)),
    ("blocks_05_09", (5, 6, 7, 8, 9)),
    ("blocks_10_15", (10, 11, 12, 13, 14, 15)),
    ("blocks_16_21", (16, 17, 18, 19, 20, 21)),
    ("block_22", (22,)),
)
METRICS = ("AP50:95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100")
SMOKE_LIMIT = 128


def parse_group(value: str) -> tuple[str, tuple[int, ...]]:
    try:
        name, raw_ids = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("group must be NAME=BLOCK_IDS") from exc
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError("group NAME must contain only letters, digits, '.', '_' or '-'")
    try:
        ids = tuple(sorted({int(item) for item in raw_ids.split(",") if item != ""}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("group block IDs must be comma-separated integers") from exc
    if not ids or any(block_id < 0 for block_id in ids):
        raise argparse.ArgumentTypeError("group requires one or more non-negative block IDs")
    return name, ids


def exclusion_patterns(node_names: Sequence[str]) -> list[str]:
    """Return anchored regexes which match exactly the supplied ONNX names."""
    return [f"^{re.escape(name)}$" for name in sorted(set(node_names))]


def build_quantize_command(source: Path, output: Path, calibration_dir: Path,
                           modelopt_python: Path | str, patterns: Sequence[str] = ()) -> list[str]:
    command = [str(modelopt_python), "-m", "examples.yolo_int8.quantize_int8_modelopt",
               "--onnx-path", str(source), "--output-path", str(output),
               "--calibration-data-dir", str(calibration_dir), "--calibration-method", "entropy"]
    for pattern in patterns:
        command += ["--nodes-to-exclude", pattern]
    return command


def resume_identity(source_sha: str, modelopt_version: str, calibration_metadata: dict[str, Any],
                    block_ids: Sequence[int], exact_names: Sequence[str], scope: str) -> dict[str, Any]:
    return {
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy",
        "calibration_metadata": calibration_metadata,
        "calibration_count": calibration_metadata.get("count", calibration_metadata.get("calibration_count")),
        "calibration_seed": calibration_metadata.get("seed"),
        "excluded_block_ids": list(block_ids),
        "excluded_exact_node_names": list(exact_names),
        "builder_settings": BUILDER_SETTINGS,
        "scope": scope,
        "evaluation_settings": {"conf_threshold": 0.001, "iou_threshold": 0.7,
                                "limit": None if scope == "full" else SMOKE_LIMIT},
    }


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


def write_summary(output_dir: Path, rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    selective = [row for row in rows if row["label"] != "entropy_128_baseline"]
    ordered = sorted(selective, key=lambda row: (
        row.get("AP50:95") is None, -(row.get("AP50:95") or 0), row["label"]
    ))
    baseline = next((row for row in rows if row["label"] == "entropy_128_baseline"), None)
    winner = next((row["label"] for row in ordered if row.get("AP50:95") is not None), None)
    payload = {
        "stage": "Stage 1 selective-FP16 accuracy sensitivity",
        "scope": scope,
        "smoke_warning": "Correctness-only subset; not final accuracy." if scope == "smoke" else None,
        "int8_baseline": baseline,
        "best_accuracy_recovery_group": winner,
        "winner_note": "The Stage-1 winner is not the final deployment configuration.",
        "variants": ordered,
    }
    write_json(output_dir / "sensitivity_summary.json", payload)
    csv_rows = ([baseline] if baseline else []) + ordered
    fields = list(dict.fromkeys(key for row in csv_rows for key in row)) if csv_rows else ["label"]
    with (output_dir / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--calibration-data-dir", type=Path, required=True)
    parser.add_argument("--eval-images-dir", type=Path, required=True)
    parser.add_argument("--eval-annotation-path", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--modelopt-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--baseline-accuracy-json", type=Path)
    parser.add_argument("--group", action="append", type=parse_group, default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    groups = args.group or list(DEFAULT_GROUPS)
    if len({name for name, _ in groups}) != len(groups):
        parser.error("group names must be unique")
    args.groups = groups
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_python_interpreter(args.runtime_python, "runtime")
    validate_python_interpreter(args.modelopt_python, "ModelOpt")
    for path, description, directory in (
        (args.onnx_path, "source ONNX", False),
        (args.calibration_data_dir, "calibration data", True),
        (args.eval_images_dir, "evaluation images", True),
        (args.eval_annotation_path, "evaluation annotations", False),
    ):
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f"{description} not found: {path}")
    if args.baseline_accuracy_json and not args.baseline_accuracy_json.is_file():
        raise FileNotFoundError(f"baseline accuracy JSON not found: {args.baseline_accuracy_json}")
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(args.onnx_path)
    modelopt_version = query_modelopt_version(args.modelopt_python)
    calibration_metadata = json.loads((args.calibration_data_dir / "metadata.json").read_text(encoding="utf-8"))
    calibration_count = calibration_metadata.get("count", calibration_metadata.get("calibration_count"))
    calibration_seed = calibration_metadata.get("seed")
    if calibration_count != 128 or calibration_seed != 0:
        raise ValueError(
            "Stage 1 requires entropy_128 calibration metadata with count=128 and seed=0; "
            f"found count={calibration_count!r}, seed={calibration_seed!r}"
        )
    graph_report = inspect_onnx(args.onnx_path)
    available_blocks = {block["block_id"] for block in graph_report["blocks"]}
    manifest: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 1", "scope": args.scope,
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy", "calibration_metadata": calibration_metadata,
        "builder_settings": BUILDER_SETTINGS,
        "evaluation_settings": {"conf_threshold": 0.001, "iou_threshold": 0.7,
                                "limit": None if args.scope == "full" else SMOKE_LIMIT},
        "performance_benchmarking": False,
        "groups": [{"name": name, "block_ids": list(ids)} for name, ids in args.groups],
        "variants": [],
    }
    rows: list[dict[str, Any]] = []

    if args.baseline_accuracy_json:
        accuracy = json.loads(args.baseline_accuracy_json.read_text(encoding="utf-8"))
        baseline = {"label": "entropy_128_baseline", "status": "referenced", "failure_stage": None,
                    "scope": args.scope, **{key: accuracy.get(key) for key in METRICS},
                    "ONNX SHA-256": accuracy.get("onnx_sha256", accuracy.get("ONNX SHA-256")),
                    "engine SHA-256": accuracy.get("engine_sha256", accuracy.get("engine SHA-256")),
                    "baseline_accuracy_json": str(args.baseline_accuracy_json)}
        rows.append(baseline)
    else:
        args.groups = [("entropy_128_baseline", tuple())] + list(args.groups)

    baseline_ap = rows[0].get("AP50:95") if rows else None
    for label, block_ids in args.groups:
        is_baseline = label == "entropy_128_baseline"
        missing_blocks = sorted(set(block_ids) - available_blocks)
        names = exact_nodes_for_blocks(graph_report, block_ids)
        if missing_blocks:
            raise ValueError(f"{label} requests block IDs absent from named ONNX nodes: {missing_blocks}")
        if block_ids and not names:
            raise ValueError(f"{label} exclusion resolved to zero ONNX node names")
        patterns = exclusion_patterns(names)
        variant_dir = args.output_dir / label
        logs = variant_dir / "logs"
        variant_dir.mkdir(parents=True, exist_ok=True)
        identity = resume_identity(source_sha, modelopt_version, calibration_metadata,
                                   block_ids, names, args.scope)
        identity_path = variant_dir / "sensitivity_metadata.json"
        row: dict[str, Any] = {
            "label": label, "status": "success", "failure_stage": None, "scope": args.scope,
            "smoke_result": args.scope == "smoke", "excluded_block_ids": list(block_ids),
            "excluded_exact_node_count": len(names), "excluded_exact_node_names": names,
        }
        try:
            onnx_out = variant_dir / "yolov8n_int8_qdq.onnx"
            engine = variant_dir / "yolov8n_int8.engine"
            qmeta = Path(str(onnx_out) + ".conversion.json")
            quant_expected = {"source_sha256": source_sha, "modelopt_version": modelopt_version,
                              "calibration_method": "entropy",
                              "calibration_metadata": calibration_metadata,
                              "calibration_count": calibration_metadata.get("count", calibration_metadata.get("calibration_count")),
                              "calibration_seed": calibration_metadata.get("seed"),
                              "calibration_image_ids": calibration_metadata.get("image_ids", calibration_metadata.get("calibration_image_ids")),
                              "nodes_to_exclude_patterns": patterns,
                              "resolved_excluded_node_names": names}
            row["failure_stage"] = "quantization"
            if not args.resume or not onnx_out.is_file() or not metadata_matches(qmeta, quant_expected):
                row["quantization_duration_seconds"] = run_command(
                    build_quantize_command(args.onnx_path, onnx_out, args.calibration_data_dir,
                                           args.modelopt_python, patterns), logs / "quantize.log")
            row["failure_stage"] = "engine_build"
            engine_expected = expected_builder_metadata(sha256(onnx_out))
            if not args.resume or not engine.is_file() or not metadata_matches(Path(str(engine) + ".json"), engine_expected):
                row["engine_build_duration_seconds"] = run_command(
                    build_engine_command(onnx_out, engine, args.runtime_python), logs / "build_engine.log")
            accuracy_path = variant_dir / "accuracy.json"
            row["failure_stage"] = "accuracy_evaluation"
            accuracy_current = args.resume and accuracy_path.is_file() and metadata_matches(identity_path, identity)
            if not accuracy_current:
                run_command(build_accuracy_command(
                    engine, args.eval_images_dir, args.eval_annotation_path,
                    variant_dir / "predictions.json", accuracy_path,
                    None if args.scope == "full" else SMOKE_LIMIT, args.runtime_python), logs / "accuracy.log")
            accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
            row.update({key: accuracy.get(key) for key in METRICS})
            row["ONNX SHA-256"] = sha256(onnx_out)
            row["engine SHA-256"] = sha256(engine)
            write_json(identity_path, identity)
            row["failure_stage"] = None
            if is_baseline:
                baseline_ap = row.get("AP50:95")
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)})
        rows.append(row)
        for candidate in rows:
            candidate["delta_AP50:95_vs_int8_baseline"] = (
                None if baseline_ap is None or candidate.get("AP50:95") is None
                else candidate["AP50:95"] - baseline_ap
            )
        manifest["variants"] = rows
        write_json(args.output_dir / "sensitivity_manifest.json", manifest)
        write_summary(args.output_dir, rows, args.scope)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["variants"] = rows
    write_json(args.output_dir / "sensitivity_manifest.json", manifest)
    return write_summary(args.output_dir, rows, args.scope)


if __name__ == "__main__":
    main()
