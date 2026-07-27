"""Benchmark one or more FP32-external-I/O TensorRT engines, without COCOeval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.yolo_benchmark.statistics_utils import aggregate_rounds  # noqa: E402
from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess, postprocess_output  # noqa: E402
from examples.yolo_tensorrt.tensorrt_runner import TensorRTRunner, validate_engine_path  # noqa: E402

DEFAULT_IMAGES_DIR = Path("input/coco/images/val2017")
DEFAULT_ANNOTATION_PATH = Path("input/coco/annotations/instances_val2017.json")
ENGINE_FIELDS = ("h2d_ms", "gpu_compute_ms", "d2h_ms", "gpu_total_ms", "host_latency_ms", "throughput_fps")
PIPELINE_FIELDS = (
    "image_load_ms", "preprocess_ms", "h2d_ms", "gpu_compute_ms", "d2h_ms",
    "postprocess_ms", "other_overhead_ms", "pipeline_without_io_ms", "full_end_to_end_ms", "throughput_fps",
)


def parse_engine_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--engine must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--engine requires a non-empty label and path")
    return label.strip(), Path(raw_path.strip())


def validate_engine_specs(specs: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    result = []
    for label, path in specs:
        if label in seen:
            raise ValueError(f"Duplicate engine label: {label}")
        seen.add(label)
        result.append((label, validate_engine_path(path).resolve()))
    if not result:
        raise ValueError("at least one --engine is required")
    return result


def rotating_order(labels: Sequence[str], round_index: int) -> list[str]:
    offset = round_index % len(labels)
    return list(labels[offset:]) + list(labels[:offset])


def calculate_pipeline_times(wall_ms: float, load_ms: float, preprocess_ms: float,
                             h2d_ms: float, compute_ms: float, d2h_ms: float,
                             postprocess_ms: float) -> tuple[float, float, float]:
    classified = load_ms + preprocess_ms + h2d_ms + compute_ms + d2h_ms + postprocess_ms
    other = max(0.0, wall_ms - classified)
    without_io = preprocess_ms + h2d_ms + compute_ms + d2h_ms + postprocess_ms + other
    return other, without_io, load_ms + without_io


def query_gpu_state() -> dict[str, str]:
    fields = ["name", "driver_version", "pstate", "temperature.gpu", "clocks.sm", "clocks.mem", "power.draw"]
    fallback = {field: "N/A" for field in fields}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    if completed.returncode or not completed.stdout.strip():
        return fallback
    values = [value.strip() or "N/A" for value in completed.stdout.splitlines()[0].split(",")]
    return {field: values[index] if index < len(values) else "N/A" for index, field in enumerate(fields)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjacent_metadata(path: Path) -> Any:
    candidate = Path(str(path) + ".json")
    if not candidate.is_file():
        candidate = path.with_suffix(".json")
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "path": str(candidate)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engine", "pipeline", "both"), required=True)
    parser.add_argument("--engine", action="append", type=parse_engine_spec, required=True)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--engine-warmup", type=int, default=50)
    parser.add_argument("--engine-iterations", type=int, default=500)
    parser.add_argument("--engine-rounds", type=int, default=4)
    parser.add_argument("--pipeline-rounds", type=int, default=4)
    parser.add_argument("--discard-rounds", type=int, default=1)
    parser.add_argument("--conf-threshold", type=float, default=.25)
    parser.add_argument("--iou-threshold", type=float, default=.45)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    try:
        args.engine = validate_engine_specs(args.engine)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.engine_warmup < 0 or args.engine_iterations <= 0:
        parser.error("warm-up must be non-negative and iterations must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.conf_threshold <= 1 or not 0 <= args.iou_threshold <= 1:
        parser.error("thresholds must be between 0 and 1")
    active_rounds = [args.engine_rounds] if args.mode == "engine" else [args.pipeline_rounds]
    if args.mode == "both": active_rounds = [args.engine_rounds, args.pipeline_rounds]
    if any(rounds <= 0 for rounds in active_rounds) or args.discard_rounds < 0:
        parser.error("round counts must be positive and discard count non-negative")
    if any(args.discard_rounds >= rounds for rounds in active_rounds):
        parser.error("--discard-rounds must be less than every active total round count")
    return args


def load_image_list(annotation_path: Path, images_dir: Path, limit: int | None) -> list[dict]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {images_dir}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(data.get("images", []), key=lambda item: int(item["id"]))
    images = images if limit is None else images[:limit]
    if not images:
        raise RuntimeError("No COCO images selected")
    return images


def engine_metadata(label: str, path: Path, runner: TensorRTRunner) -> dict:
    tensors = {name: {"mode": str(value["mode"]), "shape": list(value["shape"]),
                      "dtype": str(value["trt_dtype"])} for name, value in runner.metadata.items()}
    print(f"Engine {label} I/O metadata: {json.dumps(tensors)}")
    return {"label": label, "path": str(path), "sha256": sha256(path),
            "adjacent_metadata": adjacent_metadata(path), "io_tensors": tensors}


def run_engine_round(runner: TensorRTRunner, tensor: Any, iterations: int) -> list[dict]:
    raw = []
    for index in range(iterations):
        started = time.perf_counter()
        _, timing = runner.infer_timed(tensor)
        host_ms = (time.perf_counter() - started) * 1000
        raw.append({"iteration": index + 1, **timing,
                    "gpu_total_ms": timing["h2d_ms"] + timing["gpu_compute_ms"] + timing["d2h_ms"],
                    "host_latency_ms": host_ms, "throughput_fps": 1000 / host_ms})
    return raw


def run_pipeline_round(runner: TensorRTRunner, images: list[dict], images_dir: Path,
                       conf: float, iou: float) -> list[dict]:
    raw = []
    interval = max(1, len(images) // 10)
    for index, info in enumerate(images, 1):
        wall_start = time.perf_counter()
        stage = time.perf_counter()
        with Image.open(images_dir / info["file_name"]) as source:
            image = source.convert("RGB")
            image.load()
        load_ms = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        tensor, size, ratio, pad_x, pad_y = letterbox_preprocess(image)
        preprocess_ms = (time.perf_counter() - stage) * 1000
        output, gpu = runner.infer_timed(tensor)
        stage = time.perf_counter()
        postprocess_output(output, size, ratio, pad_x, pad_y, conf, iou)
        post_ms = (time.perf_counter() - stage) * 1000
        wall_ms = (time.perf_counter() - wall_start) * 1000
        other, without_io, full = calculate_pipeline_times(
            wall_ms, load_ms, preprocess_ms, gpu["h2d_ms"], gpu["gpu_compute_ms"], gpu["d2h_ms"], post_ms)
        raw.append({"image_index": index, "image_id": int(info["id"]), "filename": info["file_name"],
                    "image_load_ms": load_ms, "preprocess_ms": preprocess_ms, **gpu,
                    "postprocess_ms": post_ms, "other_overhead_ms": other,
                    "pipeline_without_io_ms": without_io, "full_end_to_end_ms": full,
                    "throughput_fps": 1000 / without_io})
        if index % interval == 0 or index == len(images):
            print(f"  Processed {index}/{len(images)} images")
    return raw


def print_summary(label: str, mode: str, aggregate: dict) -> None:
    print(f"\n{label} - {mode} summary (thresholds apply only to pipeline)")
    fields = ENGINE_FIELDS if mode == "engine" else PIPELINE_FIELDS
    for field in fields:
        stats = aggregate[field]["all_iterations"]
        round_sd = aggregate[field]["round_means"]["standard_deviation"]
        unit = "FPS" if field == "throughput_fps" else "ms"
        print(f"  {field}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
              f"P95={stats['p95']:.3f} {unit}; round-mean SD={round_sd:.3f} {unit}")
    print(f"  FPS (mean reciprocal latency): {aggregate['throughput_fps']['all_iterations']['mean']:.3f}")


def write_csv(path: Path, results: dict) -> None:
    rows = []
    for mode, by_engine in results.items():
        for label, result in by_engine.items():
            for round_result in result["rounds"]:
                for raw in round_result["raw"]:
                    rows.append({"engine_label": label, "mode": mode,
                                 "round_index": round_result["round_index"],
                                 "discarded": round_result["discarded"], **raw})
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    images = load_image_list(args.annotation_path, args.images_dir, args.limit)
    first_path = args.images_dir / images[0]["file_name"]
    with Image.open(first_path) as source:
        first_tensor, *_ = letterbox_preprocess(source.convert("RGB"))
    runners = {label: TensorRTRunner(path) for label, path in args.engine}
    metadata = {label: engine_metadata(label, path, runners[label]) for label, path in args.engine}
    import torch
    import tensorrt
    environment = {"TensorRT": tensorrt.__version__, "PyTorch": torch.__version__,
                   "CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
                   "Python": sys.version, "OS": platform.platform(), "driver": query_gpu_state()["driver_version"]}
    results: dict[str, dict] = {}
    labels = [label for label, _ in args.engine]
    for mode in (("engine", "pipeline") if args.mode == "both" else (args.mode,)):
        total_rounds = args.engine_rounds if mode == "engine" else args.pipeline_rounds
        results[mode] = {label: {"rounds": []} for label in labels}
        for round_index in range(total_rounds):
            order = rotating_order(labels, round_index)
            discarded = round_index < args.discard_rounds
            for label in order:
                print(f"Mode={mode} round={round_index + 1}/{total_rounds} discarded={discarded} "
                      f"engine={label} order={' -> '.join(order)}")
                runner = runners[label]
                if mode == "engine" and round_index == 0:
                    runner.warmup(first_tensor, args.engine_warmup)
                state_start = query_gpu_state(); began = time.perf_counter()
                raw = (run_engine_round(runner, first_tensor, args.engine_iterations) if mode == "engine"
                       else run_pipeline_round(runner, images, args.images_dir, args.conf_threshold, args.iou_threshold))
                elapsed = time.perf_counter() - began
                results[mode][label]["rounds"].append({"round_index": round_index + 1,
                    "discarded": discarded, "engine_order": order, "gpu_state_start": state_start,
                    "gpu_state_end": query_gpu_state(), "elapsed_seconds": elapsed, "raw": raw})
                print(f"Round completed in {elapsed:.2f}s")
        fields = ENGINE_FIELDS if mode == "engine" else PIPELINE_FIELDS
        for label in labels:
            aggregate = aggregate_rounds(results[mode][label]["rounds"], fields)
            results[mode][label]["aggregate"] = aggregate
            print_summary(label, mode, aggregate)
    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key != "engine"
    }
    serialized_arguments["engine"] = [f"{label}={path}" for label, path in args.engine]
    payload = {"started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
               "arguments": serialized_arguments,
               "engines": metadata, "environment": environment, "batch_size": 1, "input_shape": [1, 3, 640, 640],
               "confidence_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold,
               "note": "Latency thresholds are 0.25/0.45 by default; COCO accuracy evaluation uses 0.001/0.7.",
               "results": results}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv: write_csv(args.output_csv, results)
    return payload


if __name__ == "__main__":
    main()
