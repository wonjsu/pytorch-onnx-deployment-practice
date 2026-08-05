"""Run a reproducible explicit-Q/DQ INT8 calibration matrix for YOLOv8n.

The runner generates one deterministic master calibration set and reuses nested
prefixes for each requested count.  It intentionally builds every TensorRT INT8
engine from its own explicit-Q/DQ ONNX model and never passes TensorRT
calibrators to the builder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.yolo_int8.generate_calibration_data import generate as generate_calibration
from examples.yolo_int8.generate_calibration_data import select_images, sha256

DEFAULT_METHODS = ("entropy", "max")
DEFAULT_COUNTS = (128, 256, 512, 1024)
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = Path("precision-experiment-results/calibration_matrix_int8")
DEFAULT_ONNX_PATH = Path("examples/yolo_onnx/artifacts/yolov8n.onnx")
BUILDER_SETTINGS = {
    "model_precision": "int8",
    "tf32": "off",
    "workspace_gb": 2,
    "builder_optimization_level": 5,
    "avg_timing_iterations": 8,
    "max_num_tactics": -1,
    "max_aux_streams": 1,
}


def expected_builder_metadata(onnx_sha256: str) -> dict[str, Any]:
    return {
        "source_onnx_sha256": onnx_sha256,
        "model_precision": BUILDER_SETTINGS["model_precision"],
        "tf32_enabled": False,
        "workspace_bytes": int(BUILDER_SETTINGS["workspace_gb"] * 2**30),
        "builder_optimization_level": BUILDER_SETTINGS["builder_optimization_level"],
        "avg_timing_iterations": BUILDER_SETTINGS["avg_timing_iterations"],
        "max_num_tactics": BUILDER_SETTINGS["max_num_tactics"],
        "max_aux_streams": BUILDER_SETTINGS["max_aux_streams"],
    }

@dataclass(frozen=True)
class MatrixConfig:
    method: str
    count: int
    seed: int = DEFAULT_SEED

    @property
    def label(self) -> str:
        return f"{self.method}_{self.count}"

    @property
    def relative_artifact_dir(self) -> Path:
        return Path(self.label)


def matrix_configurations(methods: Sequence[str], counts: Sequence[int], seed: int) -> list[MatrixConfig]:
    """Return deterministic method-major matrix ordering."""
    return [MatrixConfig(method, int(count), seed) for method in methods for count in counts]


def labels_are_unique(configs: Sequence[MatrixConfig]) -> bool:
    labels = [config.label for config in configs]
    return len(labels) == len(set(labels))


def calibration_image_ids_hash(image_ids: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in image_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_master_images(annotation_path: Path, max_count: int, seed: int) -> list[dict[str, Any]]:
    """Select the largest deterministic set once so smaller sets are nested prefixes."""
    return select_images(annotation_path, max_count, seed)


def subset_ids(master: Sequence[dict[str, Any]], count: int) -> list[int]:
    if count > len(master):
        raise ValueError(f"requested count {count} exceeds master calibration set of {len(master)} images")
    return [int(item["id"]) for item in master[:count]]


def is_nested(master: Sequence[dict[str, Any]], counts: Sequence[int]) -> bool:
    ordered = sorted(int(count) for count in counts)
    return all(subset_ids(master, smaller) == subset_ids(master, larger)[:smaller]
               for smaller, larger in zip(ordered, ordered[1:]))


def command_to_module(python: Path | str, module: str, *args: str) -> list[str]:
    return [str(python), "-m", module, *args]


def default_modelopt_python(root: Path = ROOT, platform: str = sys.platform) -> Path:
    if platform.startswith("win"):
        return root / ".venv-modelopt" / "Scripts" / "python.exe"
    posix = root / ".venv-modelopt" / "bin" / "python"
    return posix if posix.exists() else Path(sys.executable)


def validate_python_interpreter(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{role} Python interpreter does not exist: {path}")


def query_modelopt_version(modelopt_python: Path) -> str:
    cmd = [str(modelopt_python), "-c", "import modelopt; print(getattr(modelopt, '__version__', 'unknown'))"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"failed to query ModelOpt version with {modelopt_python}{suffix}")
    return completed.stdout.strip() or "unknown"


def build_quantize_command(source: Path, output: Path, calibration_dir: Path, method: str, modelopt_python: Path | str) -> list[str]:
    return command_to_module(modelopt_python, "examples.yolo_int8.quantize_int8_modelopt", "--onnx-path", str(source),
                             "--output-path", str(output), "--calibration-data-dir", str(calibration_dir),
                             "--calibration-method", method)


def build_engine_command(onnx_path: Path, engine_path: Path, runtime_python: Path | str) -> list[str]:
    s = BUILDER_SETTINGS
    return command_to_module(runtime_python, "examples.yolo_tensorrt.build_engine", "--onnx-path", str(onnx_path),
                             "--engine-path", str(engine_path), "--model-precision", s["model_precision"],
                             "--tf32", s["tf32"], "--workspace-gb", str(s["workspace_gb"]),
                             "--builder-optimization-level", str(s["builder_optimization_level"]),
                             "--avg-timing-iterations", str(s["avg_timing_iterations"]),
                             "--max-num-tactics", str(s["max_num_tactics"]),
                             "--max-aux-streams", str(s["max_aux_streams"]))


def build_accuracy_command(engine: Path, images_dir: Path, annotation_path: Path, predictions: Path,
                           metrics: Path, limit: int | None, runtime_python: Path | str) -> list[str]:
    cmd = command_to_module(runtime_python, "examples.yolo_coco.evaluate_coco", "--backend", "tensorrt", "--engine-path", str(engine),
                            "--images-dir", str(images_dir), "--annotation-path", str(annotation_path),
                            "--conf-threshold", "0.001", "--iou-threshold", "0.7", "--output-json", str(predictions),
                            "--metrics-json", str(metrics))
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def build_benchmark_command(engines: Sequence[tuple[str, Path]], output_json: Path, output_csv: Path, scope: str,
                            runtime_python: Path | str) -> list[str]:
    warmup, iterations, rounds = (50, 500, 9) if scope == "full" else (10, 100, 3)
    cmd = command_to_module(runtime_python, "examples.yolo_benchmark.benchmark_precision", "--mode", "engine",
                            "--engine-warmup", str(warmup), "--engine-iterations", str(iterations),
                            "--engine-rounds", str(rounds), "--discard-rounds", "1",
                            "--output-json", str(output_json), "--output-csv", str(output_csv))
    for label, path in engines:
        cmd += ["--engine", f"{label}={path}"]
    return cmd


def metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(cmd: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    duration = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    return duration


def apply_benchmark_results(rows: list[dict[str, Any]], benchmark_path: Path) -> None:
    if not benchmark_path.is_file():
        return
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    engine_results = payload.get("results", {}).get("engine", {})
    for row in rows:
        aggregate = engine_results.get(row["label"], {}).get("aggregate", {})
        for source, prefix in (("gpu_compute_ms", "compute"), ("gpu_total_ms", "GPU total"), ("host_latency_ms", "host latency")):
            stats = aggregate.get(source, {}).get("all_iterations", {})
            row[f"{prefix} median"] = stats.get("median")
            row[f"{prefix} P95"] = stats.get("p95")
        row["throughput"] = aggregate.get("throughput_fps", {}).get("all_iterations", {}).get("mean")


def summarize(rows: list[dict[str, Any]], output_dir: Path) -> None:
    baseline = next((row for row in rows if row["label"] == "entropy_256" and row.get("AP50:95") is not None), None)
    baseline_ap = baseline.get("AP50:95") if baseline else None
    for row in rows:
        row["accuracy_delta_relative_to_entropy_256"] = (None if baseline_ap is None or row.get("AP50:95") is None
                                                          else row["AP50:95"] - baseline_ap)
    write_json(output_dir / "matrix_summary.json", {
        "note": "entropy_256 is the historical comparison baseline, not necessarily the winner.",
        "failed_configurations": [r for r in rows if r["status"] != "success"], "rows": rows})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    p.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    p.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    p.add_argument("--modelopt-python", type=Path, default=default_modelopt_python())
    p.add_argument("--calibration-images-dir", type=Path, required=True)
    p.add_argument("--calibration-annotation-path", type=Path, required=True)
    p.add_argument("--eval-images-dir", type=Path, required=True)
    p.add_argument("--eval-annotation-path", type=Path, required=True)
    g = p.add_mutually_exclusive_group(); g.add_argument("--resume", action="store_true"); g.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if any(count <= 0 for count in args.counts): p.error("--counts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_python_interpreter(args.runtime_python, "runtime")
    validate_python_interpreter(args.modelopt_python, "ModelOpt")
    modelopt_version = query_modelopt_version(args.modelopt_python)
    if args.force and args.output_dir.exists(): shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_count = max(args.counts)
    if not args.calibration_images_dir.is_dir(): raise FileNotFoundError(f"calibration images not found: {args.calibration_images_dir}")
    if not args.calibration_annotation_path.is_file(): raise FileNotFoundError(f"calibration annotations not found: {args.calibration_annotation_path}")
    source_sha = sha256(args.onnx_path)
    master_dir = args.output_dir / "calibration" / f"seed{args.seed}_master{max_count}"
    master_meta = master_dir / "metadata.json"
    if not args.resume or not metadata_matches(master_meta, {"count": max_count, "seed": args.seed, "fp32_onnx_sha256": source_sha}):
        generate_calibration(args.calibration_images_dir, args.calibration_annotation_path, max_count, args.seed, master_dir, args.onnx_path)
    master = json.loads(master_meta.read_text(encoding="utf-8"))["images"]
    if not is_nested(master, args.counts): raise RuntimeError("internal error: calibration prefixes are not nested")
    configs = matrix_configurations(args.methods, args.counts, args.seed)
    if not labels_are_unique(configs): raise RuntimeError("matrix labels are not unique")
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "source_onnx_sha256": source_sha,
                "builder_settings": BUILDER_SETTINGS, "configurations": []}
    rows = []
    success_engines: list[tuple[str, Path]] = []
    eval_limit = None if args.scope == "full" else min(args.counts)
    for config in configs:
        cdir = args.output_dir / config.relative_artifact_dir; logs = cdir / "logs"; cdir.mkdir(parents=True, exist_ok=True)
        ids = subset_ids(master, config.count)
        expected = {"source_onnx_sha256": source_sha, "calibration_method": config.method, "calibration_count": config.count,
                    "seed": config.seed, "calibration_image_ids": ids, "builder_settings": BUILDER_SETTINGS}
        row = {"label": config.label, "method": config.method, "count": config.count, "seed": config.seed,
               "calibration_image_ids_hash": calibration_image_ids_hash(ids), "calibration_image_ids": ids,
               "status": "success", "failure_stage": None, "error_log_path": None}
        try:
            write_json(cdir / "calibration" / "metadata.json", {**expected, "reference_directory": str(master_dir), "prefix_count": config.count})
            onnx_out = cdir / "yolov8n_int8_qdq.onnx"; engine = cdir / "yolov8n_int8.engine"
            qmeta = Path(str(onnx_out) + ".conversion.json")
            if not args.resume or not metadata_matches(qmeta, {"source_sha256": source_sha, "calibration_method": config.method, "calibration_count": config.count, "calibration_seed": config.seed, "calibration_image_ids": ids, "modelopt_version": modelopt_version}):
                row["quantization_duration"] = run_command(build_quantize_command(args.onnx_path, onnx_out, cdir / "calibration", config.method, args.modelopt_python), logs / "quantize.log")
            emeta = Path(str(engine) + ".json")
            if not args.resume or not metadata_matches(emeta, expected_builder_metadata(sha256(onnx_out))):
                row["engine_build_duration"] = run_command(build_engine_command(onnx_out, engine, args.runtime_python), logs / "build_engine.log")
            row["ONNX SHA-256"] = sha256(onnx_out); row["engine SHA-256"] = sha256(engine)
            success_engines.append((config.label, engine))
            accuracy_path = cdir / "accuracy.json"
            run_command(build_accuracy_command(engine, args.eval_images_dir, args.eval_annotation_path, cdir / "predictions.json", accuracy_path, eval_limit, args.runtime_python), logs / "accuracy.log")
            if accuracy_path.is_file():
                accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
                accuracy.update({"calibration_method": config.method, "calibration_count": config.count,
                                 "calibration_seed": config.seed, "calibration_image_ids": ids})
                write_json(accuracy_path, accuracy); row.update(accuracy)
        except Exception as exc:
            row.update({"status": "failed", "failure_stage": row.get("failure_stage") or "configuration", "error_log_path": str(logs), "error": str(exc)})
        rows.append(row); manifest["configurations"].append(row); write_json(args.output_dir / "matrix_manifest.json", manifest); summarize(rows, args.output_dir)
    if success_engines:
        try:
            run_command(build_benchmark_command(success_engines, args.output_dir / "benchmark.json", args.output_dir / "benchmark.csv", args.scope, args.runtime_python), args.output_dir / "benchmark.log")
            apply_benchmark_results(rows, args.output_dir / "benchmark.json")
        except Exception as exc:
            manifest["benchmark_error"] = str(exc)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(); write_json(args.output_dir / "matrix_manifest.json", manifest); summarize(rows, args.output_dir)
    return manifest


if __name__ == "__main__":
    main()
