"""Build and benchmark a reproducible TensorRT builder-configuration sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REFERENCE_WORKSPACE_GB = 2.0
CONFIGURATIONS = (
    ("reference", 5, 8, -1, 2.0, 0), ("opt3", 3, 8, -1, 2.0, 0),
    ("opt4", 4, 8, -1, 2.0, 0), ("timing1", 5, 1, -1, 2.0, 0),
    ("timing4", 5, 4, -1, 2.0, 0), ("tactics8", 5, 8, 8, 2.0, 0),
    ("tactics16", 5, 8, 16, 2.0, 0), ("tactics32", 5, 8, 32, 2.0, 0),
    ("workspace1", 5, 8, -1, 1.0, 0), ("aux1", 5, 8, -1, 2.0, 1),
    ("aux_auto", 5, 8, -1, 2.0, "auto"),
)

EXTENDED_CONFIGURATIONS = (
    ("opt0", 0, 8, -1, 2.0, 0), ("opt1", 1, 8, -1, 2.0, 0),
    ("opt2", 2, 8, -1, 2.0, 0), ("timing2", 5, 2, -1, 2.0, 0),
    ("timing16", 5, 16, -1, 2.0, 0), ("tactics64", 5, 8, 64, 2.0, 0),
    ("workspace05", 5, 8, -1, 0.5, 0), ("workspace4", 5, 8, -1, 4.0, 0),
    ("aux2", 5, 8, -1, 2.0, 2), ("tactics8_aux1", 5, 8, 8, 2.0, 1),
    ("tactics8_opt3", 3, 8, 8, 2.0, 0), ("opt3_aux1", 3, 8, -1, 2.0, 1),
    ("tactics8_opt3_aux1", 3, 8, 8, 2.0, 1),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predefined_configurations(workspace_gb: float = REFERENCE_WORKSPACE_GB,
                              suite: str = "baseline") -> list[dict[str, Any]]:
    """Return fresh, deterministic configurations; workspace1 is never overridden."""
    selected = {"baseline": CONFIGURATIONS, "extended": EXTENDED_CONFIGURATIONS,
                "all": CONFIGURATIONS + EXTENDED_CONFIGURATIONS}[suite]
    return [{"label": label, "builder_optimization_level": level,
             "avg_timing_iterations": avg, "max_num_tactics": tactics,
             "workspace_gb": (_workspace if label in {item[0] for item in EXTENDED_CONFIGURATIONS}
                              else 1.0 if label == "workspace1" else workspace_gb),
             "max_aux_streams": aux,
             "max_aux_streams_mode": "auto" if aux == "auto" else "explicit"}
            for label, level, avg, tactics, _workspace, aux in selected]


def engine_path(output_dir: Path, label: str, precision: str) -> Path:
    return output_dir / "engines" / f"yolov8n_{precision}_{label}.engine"


def metadata_matches(metadata: dict[str, Any], onnx_digest: str, settings: dict[str, Any]) -> bool:
    expected = {"onnx_sha256": onnx_digest, **{key: settings[key] for key in (
        "builder_optimization_level", "avg_timing_iterations", "max_num_tactics",
        "max_aux_streams", "max_aux_streams_mode")},
        "workspace_bytes": int(settings["workspace_gb"] * 2**30)}
    return all(metadata.get(key) == value for key, value in expected.items())


def percent_change(value: float, reference: float) -> float:
    return (value - reference) / reference * 100.0


def create_summary(configurations: list[dict[str, Any]], builds: dict[str, dict[str, Any]],
                   benchmark: dict[str, Any], failed_configurations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = []
    metrics = {"h2d_median_ms": ("h2d_ms", "median"), "h2d_p95_ms": ("h2d_ms", "p95"),
               "compute_mean_ms": ("gpu_compute_ms", "mean"), "compute_median_ms": ("gpu_compute_ms", "median"),
               "compute_p95_ms": ("gpu_compute_ms", "p95"), "d2h_median_ms": ("d2h_ms", "median"),
               "d2h_p95_ms": ("d2h_ms", "p95"), "gpu_total_median_ms": ("gpu_total_ms", "median"),
               "gpu_total_p95_ms": ("gpu_total_ms", "p95"), "host_latency_median_ms": ("host_latency_ms", "median"),
               "host_latency_p95_ms": ("host_latency_ms", "p95"), "throughput_mean_fps": ("throughput_fps", "mean"),
               "throughput_median_fps": ("throughput_fps", "median")}
    for config in configurations:
        label = config["label"]
        if label not in builds:
            continue
        aggregate = benchmark["results"]["engine"][label]["aggregate"]
        row = {**config, **builds[label]}
        row.update({name: aggregate[field]["all_iterations"][stat] for name, (field, stat) in metrics.items()})
        rows.append(row)
    reference = next((row for row in rows if row["label"] == "reference"), None)
    if reference:
        for row in rows:
            row["percent_difference_from_reference"] = {
                metric: percent_change(row[metric], reference[metric]) for metric in metrics}
    rows.sort(key=lambda row: row["compute_median_ms"])
    winner = rows[0] if rows else None
    advantage = ((reference["compute_median_ms"] - winner["compute_median_ms"])
                 / reference["compute_median_ms"] * 100) if reference and winner else None
    return {"reference_label": "reference" if reference else None,
            "winner_label": winner["label"] if winner else None,
            "winner_advantage_over_reference_percent": advantage,
            "winner_advantage_smaller_than_2_percent": advantage < 2.0 if advantage is not None else None,
            "failed_configurations": failed_configurations or [], "configurations": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--model-precision", choices=("fp32", "mixed-fp16", "int8"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-gb", type=float, default=REFERENCE_WORKSPACE_GB)
    parser.add_argument("--engine-warmup", type=int)
    parser.add_argument("--engine-iterations", type=int)
    parser.add_argument("--engine-rounds", type=int)
    parser.add_argument("--discard-rounds", type=int)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--suite", choices=("baseline", "extended", "all"), default="baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.onnx_path.is_file(): parser.error(f"ONNX file not found: {args.onnx_path}")
    if args.workspace_gb <= 0: parser.error("--workspace-gb must be positive")
    defaults = (10, 100, 3, 1) if args.profile == "smoke" else (50, 500, 4, 1)
    for name, default in zip(("engine_warmup", "engine_iterations", "engine_rounds", "discard_rounds"), defaults):
        if getattr(args, name) is None: setattr(args, name, default)
    if args.engine_warmup < 0 or args.engine_iterations <= 0 or args.engine_rounds <= 0:
        parser.error("benchmark warm-up must be non-negative and iterations/rounds positive")
    if args.discard_rounds < 0 or args.discard_rounds >= args.engine_rounds:
        parser.error("discard rounds must be non-negative and less than engine rounds")
    return args


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    configurations = predefined_configurations(args.workspace_gb, args.suite); digest = sha256(args.onnx_path)
    started = datetime.now(timezone.utc).isoformat(); builds = {}; commands = []; failures = []
    manifest_path = args.output_dir / "builder_sweep_manifest.json"
    benchmark_json = args.output_dir / "builder_sweep_benchmark.json"; benchmark_csv = args.output_dir / "builder_sweep_benchmark.csv"
    # An interrupted run leaves a manifest intentionally; engines with matching
    # metadata remain resumable. Completed benchmark/summary files are protected.
    final_artifacts = (benchmark_json, benchmark_csv, args.output_dir / "builder_sweep_summary.json",
                       args.output_dir / "builder_sweep_summary.csv")
    existing_final = [artifact for artifact in final_artifacts if artifact.exists()]
    if existing_final and not args.force:
        raise RuntimeError("Refusing to overwrite existing sweep artifacts: " + ", ".join(map(str, existing_final)))

    def write_manifest(environment: dict[str, Any] | None = None) -> None:
        manifest = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                    "onnx_path": str(args.onnx_path), "onnx_sha256": digest,
                    "command": [sys.executable, *sys.argv], "suite": args.suite,
                    "build_commands": commands, "builder_settings": configurations,
                    "benchmark_settings": {key: getattr(args, key) for key in ("profile", "engine_warmup", "engine_iterations", "engine_rounds", "discard_rounds")},
                    "builds": builds, "failed_configurations": failures,
                    "environment": environment or {}}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for config in configurations:
        label = config["label"]; path = engine_path(args.output_dir, label, args.model_precision)
        metadata_path = Path(str(path) + ".json")
        command = [sys.executable, "-m", "examples.yolo_tensorrt.build_engine", "--onnx-path", str(args.onnx_path),
                   "--engine-path", str(path), "--model-precision", args.model_precision, "--workspace-gb", str(config["workspace_gb"]),
                   "--builder-optimization-level", str(config["builder_optimization_level"]), "--avg-timing-iterations", str(config["avg_timing_iterations"]),
                   "--max-num-tactics", str(config["max_num_tactics"]), "--max-aux-streams", str(config["max_aux_streams"])]
        commands.append(command)
        reused = False; elapsed = 0.0
        if path.exists() and not args.force:
            if not metadata_path.is_file() or not metadata_matches(json.loads(metadata_path.read_text(encoding="utf-8")), digest, config):
                raise RuntimeError(f"Existing engine metadata does not match requested configuration: {path}")
            reused = True
        else:
            log_paths = (args.output_dir / f"build_{label}.stdout.log", args.output_dir / f"build_{label}.stderr.log")
            collisions = [artifact for artifact in (path, metadata_path, Path(str(path) + ".inspector.json"), *log_paths)
                          if artifact.exists()]
            if collisions and not args.force:
                raise RuntimeError("Refusing to overwrite existing build artifacts: " + ", ".join(map(str, collisions)))
            path.parent.mkdir(parents=True, exist_ok=True); began = time.perf_counter()
            with log_paths[0].open("w", encoding="utf-8") as stdout, log_paths[1].open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
            elapsed = time.perf_counter() - began
            if completed.returncode:
                failures.append({"label": label, "returncode": completed.returncode,
                                 "stdout_log": str(log_paths[0]), "stderr_log": str(log_paths[1]),
                                 "error": f"Build failed with exit code {completed.returncode}"})
                write_manifest()
                continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        builds[label] = {"engine_path": str(path), "engine_sha256": sha256(path), "engine_file_size_bytes": path.stat().st_size,
                         "engine_build_time_seconds": metadata.get("build_time_seconds", elapsed), "reused": reused}
        write_manifest()
    benchmark_args = ["--mode", "engine", "--engine-warmup", str(args.engine_warmup), "--engine-iterations", str(args.engine_iterations),
                      "--engine-rounds", str(args.engine_rounds), "--discard-rounds", str(args.discard_rounds),
                      "--output-json", str(benchmark_json), "--output-csv", str(benchmark_csv)]
    for config in configurations:
        if config["label"] in builds:
            benchmark_args += ["--engine", f'{config["label"]}={engine_path(args.output_dir, config["label"], args.model_precision)}']
    if not builds:
        write_manifest()
        raise RuntimeError("All builder configurations failed; see builder_sweep_manifest.json")
    from examples.yolo_benchmark.benchmark_precision import main as benchmark_main
    benchmark = benchmark_main(benchmark_args)
    summary = create_summary(configurations, builds, benchmark, failures)
    (args.output_dir / "builder_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "builder_sweep_summary.csv", summary["configurations"])
    write_manifest(benchmark.get("environment", {}))
    return summary


if __name__ == "__main__":
    main()
