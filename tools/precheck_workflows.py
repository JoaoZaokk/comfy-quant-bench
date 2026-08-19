"""Pre-flight the rewritten workflows against the running ComfyUI, before opening any of them.

Two failure modes the rewrite could have left behind, both of which only surface at Queue time
and both of which are answerable from /object_info plus the filesystem:

  * a node type that does not exist in this installation -- swap_to_nunchaku wrote the type name
    from a table, it never asked ComfyUI whether the node is installed
  * a widget value naming a model that is not in any search path

Also reports, per workflow, every model-ish widget value it can find, so a workflow that still
points at the old checkpoint somewhere else in the graph is visible rather than assumed absent.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path("F:/COMFY_PORTABLE")
WF = ROOT / "ComfyUI/user/default/workflows"
API = "http://127.0.0.1:8190"

with urllib.request.urlopen(f"{API}/object_info", timeout=120) as r:
    info = json.load(r)
print(f"{len(info)} node types registered\n")

# Every file ComfyUI would accept, from the loaders' own combo lists -- the authoritative answer,
# not a directory listing that might miss an extra_model_paths mount.
known_files = set()
for node, spec in info.items():
    for group in ("required", "optional"):
        for _name, entry in (spec.get("input", {}).get(group, {}) or {}).items():
            if isinstance(entry, list) and entry and isinstance(entry[0], list):
                known_files.update(str(v) for v in entry[0] if isinstance(v, str))

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".sft", ".pt", ".pth", ".bin")
# Frontend-only nodes. They are drawn by the browser and never reach the backend, so they are
# absent from /object_info by design -- flagging them made all six workflows look broken.
FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode", "Group"}
failures = 0
for path in sorted(WF.glob("*.nunchaku.json")):
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    graphs = (doc.get("definitions", {}).get("subgraphs", []) or []) + [doc]
    missing_nodes, missing_models, seen_models = [], [], []
    for graph in graphs:
        for node in graph.get("nodes", []) or []:
            ntype = node.get("type")
            # Subgraph instances carry a uuid as their type; those are not registered nodes.
            if ntype and ntype in info:
                pass
            elif (ntype and "-" not in str(ntype) and ntype not in info
                  and ntype not in FRONTEND_ONLY):
                missing_nodes.append(f"{ntype} (node {node.get('id')})")
            # widgets_values is a list on almost every node, but VHS_* nodes store a dict. A
            # plain `for` over a dict iterates its keys, so those nodes passed the check without
            # a single value being examined. Measured in the real workflow directory: 556 nodes
            # with a list, 3 with a dict.
            widgets = node.get("widgets_values") or []
            values = widgets.values() if isinstance(widgets, dict) else widgets
            for value in values:
                if isinstance(value, str) and value.lower().endswith(MODEL_EXT):
                    seen_models.append(value)
                    if value not in known_files:
                        missing_models.append(f"{value} (node {node.get('id')} {ntype})")

    status = "OK " if not missing_nodes and not missing_models else "BAD"
    if status == "BAD":
        failures += 1
    print(f"{status} {path.name}")
    for m in dict.fromkeys(seen_models):
        mark = "  !!" if any(m in x for x in missing_models) else "    "
        print(f"{mark} {m}")
    for n in dict.fromkeys(missing_nodes):
        print(f"  !! unknown node type: {n}")

print(f"\n{failures} workflow(s) with a problem, {len(list(WF.glob('*.nunchaku.json')))} checked")
sys.exit(1 if failures else 0)
