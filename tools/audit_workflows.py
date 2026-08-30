"""Audit saved ComfyUI workflows against the running local installation.

The audit is read-only. It checks every UI graph and embedded subgraph for
registered node types, model references, accelerator/device nodes, and exact
duplicate files. Optional JSON output is useful for comparing later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "ComfyUI" / "user" / "default" / "workflows"
MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".sft", ".pt", ".pth", ".bin")
FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode", "Group"}
UUID_TYPE = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.IGNORECASE)
ACCELERATOR_WORDS = (
    "multigpu",
    "device",
    "offload",
    "unload",
    "cache",
    "compile",
    "nunchaku",
    "flash",
    "sage",
    "triton",
)
REMOTE_WORDS = ("seedance", "seedream", "wavespeed", "api")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8190")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def graphs(document: dict) -> list[tuple[str, dict]]:
    found = []
    for index, graph in enumerate(document.get("definitions", {}).get("subgraphs", []) or []):
        found.append((graph.get("name") or f"subgraph-{index}", graph))
    found.append(("root", document))
    return found


def widget_values(node: dict):
    values = node.get("widgets_values") or []
    if isinstance(values, dict):
        yield from values.values()
    elif isinstance(values, list):
        yield from values


def known_model_values(info: dict) -> set[str]:
    known = set()
    for spec in info.values():
        inputs = spec.get("input", {})
        for group in ("required", "optional"):
            for entry in (inputs.get(group, {}) or {}).values():
                if not isinstance(entry, list) or not entry:
                    continue
                if isinstance(entry[0], list):
                    known.update(value for value in entry[0] if isinstance(value, str))
                elif (
                    entry[0] == "COMBO"
                    and len(entry) > 1
                    and isinstance(entry[1], dict)
                ):
                    known.update(
                        value
                        for value in entry[1].get("options", [])
                        if isinstance(value, str)
                    )
    return known


def audit_file(path: Path, info: dict, known_models: set[str]) -> dict:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    node_types = Counter()
    missing_nodes = []
    models = []
    missing_models = []
    accelerator_nodes = []
    devices = []
    graph_count = 0

    for graph_name, graph in graphs(document):
        graph_count += 1
        for node in graph.get("nodes", []) or []:
            node_type = str(node.get("type") or "")
            if not node_type:
                continue
            node_types[node_type] += 1
            if (
                node_type not in info
                and node_type not in FRONTEND_ONLY
                and not UUID_TYPE.match(node_type)
            ):
                missing_nodes.append(
                    {"graph": graph_name, "id": node.get("id"), "type": node_type}
                )
            if any(word in node_type.lower() for word in ACCELERATOR_WORDS):
                accelerator_nodes.append(
                    {"graph": graph_name, "id": node.get("id"), "type": node_type}
                )
            for value in widget_values(node):
                if isinstance(value, str) and re.fullmatch(r"(?:cuda|gpu):\d+|cpu", value):
                    devices.append(
                        {"graph": graph_name, "id": node.get("id"), "type": node_type, "value": value}
                    )
                if isinstance(value, str) and value.lower().endswith(MODEL_EXTENSIONS):
                    entry = {"graph": graph_name, "id": node.get("id"), "type": node_type, "value": value}
                    models.append(entry)
                    if value not in known_models:
                        missing_models.append(entry)

    lowered_identity = " ".join(
        [str(path.relative_to(WORKFLOWS)), *node_types.keys()]
    ).lower()
    remote = any(word in lowered_identity for word in REMOTE_WORDS)
    return {
        "file": str(path.relative_to(WORKFLOWS)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "graphs": graph_count,
        "nodes": sum(node_types.values()),
        "remote_or_api": remote,
        "node_types": dict(node_types.most_common()),
        "missing_nodes": missing_nodes,
        "models": models,
        "missing_models": missing_models,
        "accelerator_nodes": accelerator_nodes,
        "devices": devices,
    }


def main() -> int:
    args = parse_args()
    with urllib.request.urlopen(f"{args.api}/object_info", timeout=180) as response:
        info = json.load(response)
    known_models = known_model_values(info)

    reports = []
    parse_errors = []
    for path in sorted(WORKFLOWS.rglob("*.json")):
        try:
            reports.append(audit_file(path, info, known_models))
        except Exception as exc:
            parse_errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    hashes = defaultdict(list)
    for report in reports:
        hashes[report["sha256"]].append(report["file"])
    duplicate_groups = [files for files in hashes.values() if len(files) > 1]

    result = {
        "workflow_root": str(WORKFLOWS),
        "api": args.api,
        "registered_node_types": len(info),
        "workflows": reports,
        "parse_errors": parse_errors,
        "duplicate_groups": duplicate_groups,
        "summary": {
            "files": len(reports),
            "remote_or_api": sum(report["remote_or_api"] for report in reports),
            "with_missing_nodes": sum(bool(report["missing_nodes"]) for report in reports),
            "with_missing_models": sum(bool(report["missing_models"]) for report in reports),
            "with_accelerator_nodes": sum(bool(report["accelerator_nodes"]) for report in reports),
            "parse_errors": len(parse_errors),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for report in reports:
        flags = []
        if report["remote_or_api"]:
            flags.append("remote/API")
        if report["missing_nodes"]:
            flags.append(f"missing-nodes={len(report['missing_nodes'])}")
        if report["missing_models"]:
            flags.append(f"missing-models={len(report['missing_models'])}")
        if report["accelerator_nodes"]:
            flags.append(f"accelerators={len(report['accelerator_nodes'])}")
        print(f"{report['file']}: {', '.join(flags) if flags else 'local/plain'}")
    for files in duplicate_groups:
        print("DUPLICATE: " + " == ".join(files))
    return 1 if parse_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
