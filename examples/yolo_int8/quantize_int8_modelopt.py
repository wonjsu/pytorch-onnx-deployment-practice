"""Quantize FP32 YOLO ONNX with ModelOpt's supported lazy-reader API."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.yolo_int8.generate_calibration_data import sha256
from examples.yolo_int8.inspect_int8_qdq_onnx import inspect_model


class LazyNpzCalibrationDataReader:
    def __init__(self, directory: Path):
        self.directory = directory
        metadata_path = directory / "metadata.json"
        prefix_count = None
        source_dir = directory
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if "reference_directory" in metadata:
                    source_dir = Path(metadata["reference_directory"])
                    prefix_count = int(metadata["prefix_count"])
            except (OSError, ValueError, json.JSONDecodeError):
                source_dir = directory
        self.files = sorted(source_dir.glob("batch_*.npz"))
        if prefix_count is not None:
            self.files = self.files[:prefix_count]
        self._index = 0
        if not self.files:
            raise ValueError(f"no calibration batches in {directory}")

    def get_next(self):
        if self._index >= len(self.files):
            return None
        import numpy as np
        path = self.files[self._index]
        self._index += 1
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}

    def rewind(self):
        self._index = 0


def resolve_excluded_node_names(source: Path, patterns: list[str] | None) -> list[str]:
    """Resolve ModelOpt regexes against named source nodes in stable name order."""
    if not patterns:
        return []
    import onnx
    try:
        regexes = [re.compile(pattern) for pattern in patterns]
    except re.error as exc:
        raise ValueError(f"invalid node exclusion regex: {exc}") from exc
    model = onnx.load(str(source), load_external_data=False)
    matches = sorted({node.name for node in model.graph.node
                      if node.name and any(regex.search(node.name) for regex in regexes)})
    if not matches:
        raise ValueError("nodes-to-exclude patterns matched zero named ONNX nodes")
    return matches


def quantize(source: Path, output: Path, calibration_dir: Path, method: str,
             nodes_to_exclude: list[str] | None = None) -> dict:
    metadata = json.loads((calibration_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata_source_sha = metadata.get("fp32_onnx_sha256", metadata.get("source_onnx_sha256"))
    if metadata_source_sha != sha256(source):
        raise ValueError("calibration metadata source SHA-256 does not match ONNX")
    patterns = list(nodes_to_exclude or [])
    resolved_names = resolve_excluded_node_names(source, patterns)

    module = importlib.import_module("modelopt.onnx.quantization")
    function = getattr(module, "quantize")
    signature = inspect.signature(function)
    supported = set(signature.parameters)
    requested = {
        "onnx_path": str(source),
        "quantize_mode": "int8",
        "calibration_method": method,
        "high_precision_dtype": "fp16",
        "calibration_data_reader": LazyNpzCalibrationDataReader(calibration_dir),
    }
    missing = [name for name in requested if name not in supported]
    output_key = next((name for name in ("output_path", "output_model_path") if name in supported), None)
    if missing or output_key is None:
        raise RuntimeError(
            f"Unsupported ModelOpt quantize signature {signature}; "
            f"missing supported parameters: {missing}, output_path"
        )
    if patterns and "nodes_to_exclude" not in supported:
        raise RuntimeError(
            f"Installed ModelOpt quantize signature {signature} does not support nodes_to_exclude"
        )
    if patterns:
        requested["nodes_to_exclude"] = patterns

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".onnx", dir=output.parent)
    os.close(fd)
    os.unlink(tmp)
    kwargs = dict(requested)
    kwargs[output_key] = tmp
    started = time.perf_counter()
    try:
        function(**kwargs)
        inspection = inspect_model(Path(tmp))
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    import modelopt
    import numpy
    import onnx
    result = {
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "modelopt_version": getattr(modelopt, "__version__", "unknown"),
        "onnx_version": onnx.__version__,
        "numpy_version": numpy.__version__,
        "quantize_mode": "int8",
        "calibration_method": method,
        "calibration_count": metadata.get("count", metadata.get("calibration_count")),
        "calibration_seed": metadata.get("seed"),
        "calibration_image_ids": metadata.get("image_ids", metadata.get("calibration_image_ids")),
        "calibration_metadata": metadata,
        "high_precision_fallback_dtype": "fp16",
        "external_io_dtype": "FP32",
        "nodes_to_exclude_patterns": patterns,
        "resolved_excluded_node_names": resolved_names,
        "resolved_excluded_node_count": len(resolved_names),
        "conversion_duration_seconds": time.perf_counter() - started,
        "qdq_inspection": inspection,
    }
    Path(str(output) + ".conversion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, default=Path("examples/yolo_onnx/artifacts/yolov8n.onnx"))
    parser.add_argument("--output-path", type=Path, default=Path("examples/yolo_int8/artifacts/yolov8n_int8_qdq.onnx"))
    parser.add_argument("--calibration-data-dir", type=Path, required=True)
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--nodes-to-exclude", action="append", default=[], metavar="REGEX")
    args = parser.parse_args(argv)
    print(json.dumps(quantize(args.onnx_path, args.output_path, args.calibration_data_dir,
                              args.calibration_method, args.nodes_to_exclude), indent=2))


if __name__ == "__main__":
    main()
