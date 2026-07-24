"""Benchmark TensorRT engine-only stages and complete image pipelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

import numpy as np

from PIL import Image
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess, letterbox_preprocess_image, postprocess_output
from examples.yolo_tensorrt.infer_tensorrt import TensorRTRunner


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile without optional dependencies."""
    if not values: raise ValueError("percentile requires at least one value")
    ordered = sorted(values); position = (len(ordered) - 1) * percent / 100.0
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    """Produce standard latency statistics in milliseconds."""
    return {"mean": mean(values), "median": median(values), "p95": percentile(values, 95), "min": min(values), "max": max(values)}


def timed(callable_: object) -> tuple[object, float]:
    start = perf_counter(); result = callable_()  # type: ignore[operator]
    return result, (perf_counter() - start) * 1000


def main() -> None:
    """Measure loaded-image and reopen-each-iteration TensorRT paths."""
    parser = argparse.ArgumentParser(); parser.add_argument("image_path", type=Path); parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--runs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=1, choices=[1])
    parser.add_argument("--conf-threshold", type=float, default=0.25); parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--json", type=Path); parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1: parser.error("--warmup must be >= 0 and --runs must be >= 1")
    cached = letterbox_preprocess_image(args.image_path)
    records: list[dict[str, float | str]] = []
    with TensorRTRunner(args.engine_path) as runner:
        for _ in range(args.warmup): runner.infer(cached[0])
        for mode in ("loaded_image_reuse", "reopen_each_iteration"):
            for _ in range(args.runs):
                total_start = perf_counter(); load_ms = 0.0
                if mode == "reopen_each_iteration":
                    image, load_ms = timed(lambda: Image.open(args.image_path).convert("RGB"))
                    loaded, preprocess_ms = timed(lambda: letterbox_preprocess(image))
                    tensor, size, ratio, px, py = loaded  # type: ignore[misc]
                else:
                    tensor, size, ratio, px, py = cached; preprocess_ms = 0.0
                raw, stages = runner.infer_timed(tensor)
                engine_ms = sum(stages.values())
                _, post_ms = timed(lambda: postprocess_output(raw.astype(np.float32), size, ratio, px, py, args.conf_threshold, args.iou_threshold))
                records.append({"mode": mode, "image_load_ms": load_ms, "preprocess_ms": preprocess_ms, **stages, "postprocess_ms": post_ms, "engine_only_ms": engine_ms, "end_to_end_ms": (perf_counter()-total_start)*1000})
    for mode in ("loaded_image_reuse", "reopen_each_iteration"):
        print(f"\n{mode}:")
        rows = [r for r in records if r["mode"] == mode]
        for key in ("image_load_ms", "preprocess_ms", "h2d_ms", "gpu_compute_ms", "d2h_ms", "postprocess_ms", "engine_only_ms", "end_to_end_ms"):
            stats = summarize([float(r[key]) for r in rows]); print(f"  {key}: " + ", ".join(f"{k}={v:.3f}" for k,v in stats.items()))
    payload = {"warmup": args.warmup, "runs": args.runs, "batch_size": args.batch_size, "records": records}
    if args.json: args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file: writer=csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__": main()
