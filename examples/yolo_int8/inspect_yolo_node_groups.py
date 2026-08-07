"""Inspect YOLO ONNX nodes grouped by top-level ``model.N`` block."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

BLOCK_PATTERN = re.compile(r"(?:^|/)model\.(\d+)(?:/|$)")
QUANTIZATION_RELEVANT_OPS = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul"})


def block_id_from_node_name(name: str) -> int | None:
    match = BLOCK_PATTERN.search(name)
    return int(match.group(1)) if match else None


def group_nodes(nodes: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[int, list[Any]] = {}
    ungrouped = []
    for node in nodes:
        if not node.name:
            continue
        block_id = block_id_from_node_name(node.name)
        (ungrouped if block_id is None else grouped.setdefault(block_id, [])).append(node)

    def report(block_id: int | None, members: list[Any]) -> dict[str, Any]:
        names = sorted(node.name for node in members)
        return {
            "block_id": block_id,
            "total_named_nodes": len(members),
            "node_type_counts": dict(sorted(Counter(node.op_type for node in members).items())),
            "quantization_relevant_node_names": sorted(
                node.name for node in members if node.op_type in QUANTIZATION_RELEVANT_OPS
            ),
            "node_names": names,
        }

    return {
        "blocks": [report(block_id, grouped[block_id]) for block_id in sorted(grouped)],
        "ungrouped_named_nodes": report(None, ungrouped),
    }


def inspect_onnx(path: Path) -> dict[str, Any]:
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    result = group_nodes(model.graph.node)
    result["onnx_path"] = str(path)
    return result


def exact_nodes_for_blocks(report: dict[str, Any], block_ids: Iterable[int]) -> list[str]:
    wanted = set(block_ids)
    return sorted(name for block in report["blocks"] if block["block_id"] in wanted
                  for name in block["node_names"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    result = inspect_onnx(args.onnx_path)
    rendered = json.dumps(result, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
