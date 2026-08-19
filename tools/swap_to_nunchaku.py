"""Rewrite a ComfyUI workflow to load an SVDQuant checkpoint instead of the BF16/FP8 one.

The swap is not "change a filename". `UNETLoader` and `NunchakuZImageDiTLoader` are different
node types with different input lists, and in these workflows the loader lives *inside a
subgraph* with its model widget promoted to a subgraph input. So four things have to move
together or the graph loads broken:

  1. the inner node's `type`, `inputs`, `widgets_values` and `properties`
  2. the subgraph's exposed input, renamed from `unet_name` to `model_name` -- its `id` is kept,
     because links inside the subgraph address it by slot index and reordering would silently
     rewire the graph
  3. the same rename on every instance of that subgraph in the outer graph
  4. `properties.models`, which is a download hint pointing at the old file, dropped rather than
     left lying about a file this node will never load

`weight_dtype` is dropped. That is safe here and checked rather than assumed: the audit found it
carries `link: null` and is not among the subgraph's exposed inputs, so nothing feeds it. The
script refuses the node if that stops being true.

Never writes over the input. Output goes to `<stem>.nunchaku.json` beside it, and an existing
output is refused rather than overwritten -- a workflow is hand-built work and a bad rewrite that
silently replaced it would be unrecoverable.

    python tools/swap_to_nunchaku.py --workflow TXT2IMG-ZIMG.json \
        --to svdq-int4_r32-z-image-turbo.safetensors
    python tools/swap_to_nunchaku.py --all --to svdq-int4_r32-z-image-turbo.safetensors --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = PORTABLE_ROOT / "ComfyUI" / "user" / "default" / "workflows"

# Which loader replaces which, keyed by a substring of the model name currently loaded. Only
# families whose SVDQuant checkpoint is a drop-in for the same task belong here: Qwen-Image-Edit
# and Qwen-Image-Layered are deliberately absent, because the base Qwen-Image SVDQuant is a
# different model and swapping it in would produce a workflow that runs and does the wrong thing.
FAMILIES = {
    "z_image": ("NunchakuZImageDiTLoader", "model_name"),
    "qwen_image": ("NunchakuQwenImageDiTLoader", "model_name"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workflow", help="file name under user/default/workflows")
    source.add_argument("--all", action="store_true", help="every workflow that matches --family")
    parser.add_argument("--to", required=True, help="the SVDQuant checkpoint to load instead")
    parser.add_argument("--family", default="z_image", choices=sorted(FAMILIES),
                        help="which loader family to rewrite")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    return parser.parse_args()


def subgraphs(doc: dict) -> list[dict]:
    return doc.get("definitions", {}).get("subgraphs", []) or []


def rewrite_node(node: dict, loader: str, widget: str, target: str) -> str | None:
    """Rewrite one loader node in place. Returns a reason string if it refused."""
    old_inputs = node.get("inputs", []) or []
    promoted = next((i for i in old_inputs if i.get("name") in ("unet_name", "model_name")), None)
    if promoted is None:
        return "no unet_name/model_name input to carry over"
    for extra in old_inputs:
        if extra is promoted:
            continue
        if extra.get("link") is not None:
            # Something feeds an input the replacement node does not have. Dropping it would
            # leave a link pointing at a slot that no longer exists.
            return f"input {extra.get('name')!r} is connected and has no equivalent"

    node["type"] = loader
    node["inputs"] = [{
        "localized_name": widget,
        "name": widget,
        "type": "COMBO",
        "widget": {"name": widget},
        "link": promoted.get("link"),
    }]
    node["widgets_values"] = [target]
    node["properties"] = {"Node name for S&R": loader}
    size = node.get("size")
    if isinstance(size, list) and len(size) == 2:
        # One widget instead of two. Leaving the old height only makes the node look wrong.
        node["size"] = [size[0], 58]
    return None


def identity(name: str) -> set[str]:
    """The tokens that say *which model* a checkpoint is, ignoring how it was stored.

    Matching on the family alone ("qwen_image") is too loose and produced a real near-miss: it
    matched `qwen_image_layered_bf16` against a Qwen-Image-**Edit** checkpoint. Both are
    "qwen image"; they are different models doing different jobs, and the swap would have
    produced a workflow that loads happily and does the wrong thing -- worse than one that fails.

    So the target's own filename decides. `svdq-int4_r128-qwen-image-edit-2509.safetensors`
    reduces to {qwen, image, edit, 2509}, and a candidate must contain all of it. Precision and
    storage tokens are dropped from both sides, because those are exactly what the swap changes.
    """
    stem = re.sub(r"\.(safetensors|ckpt|gguf|sft)$", "", name.lower())
    stem = re.sub(r"^svdq-(int4|fp4)_r\d+-", "", stem)
    noise = {"bf16", "fp16", "fp8", "fp8mixed", "e4m3fn", "e5m2", "scaled", "mixed", "int8",
             "gguf", "safetensors", "split", "files", "comfy", "org", "v1", "v2"}
    return {t for t in re.split(r"[^a-z0-9]+", stem) if t and t not in noise}


def rewrite(doc: dict, family: str, target: str) -> tuple[list[str], list[str]]:
    loader, widget = FAMILIES[family]
    wanted = identity(target)
    changed, refused = [], []
    for sub in subgraphs(doc) + [doc]:
        for node in sub.get("nodes", []) or []:
            if node.get("type") not in ("UNETLoader", "CheckpointLoaderSimple"):
                continue
            values = node.get("widgets_values") or []
            current = str(values[0]) if values else ""
            if not current:
                continue
            have = identity(current)
            if family not in current.lower().replace("-", "_"):
                continue
            if not wanted <= have:
                refused.append(f"{sub.get('name', 'root graph')}: node {node.get('id')} "
                               f"({current}) -- different model, missing "
                               f"{sorted(wanted - have)}")
                continue
            why = rewrite_node(node, loader, widget, target)
            where = sub.get("name", "root graph")
            if why:
                refused.append(f"{where}: node {node.get('id')} ({current}) -- {why}")
            else:
                changed.append(f"{where}: node {node.get('id')}  {current} -> {target}")

    # The exposed input keeps its id so slot order, and therefore every link addressing it,
    # is untouched. Only the label changes.
    for sub in subgraphs(doc):
        for exposed in sub.get("inputs", []) or []:
            if exposed.get("name") == "unet_name":
                exposed["name"] = widget
        sub_id = sub.get("id")
        for node in doc.get("nodes", []) or []:
            if node.get("type") != sub_id:
                continue
            for inp in node.get("inputs", []) or []:
                if inp.get("name") == "unet_name":
                    inp["name"] = widget
                    if isinstance(inp.get("widget"), dict):
                        inp["widget"]["name"] = widget
    return changed, refused


def process(path: Path, args) -> bool:
    # utf-8-sig, not utf-8: at least one workflow here was saved with a BOM, and plain utf-8
    # fails it with "Unexpected UTF-8 BOM". Reading it as utf-8-sig costs nothing on files
    # without one, and silently skipping a workflow is how an audit reports a clean result it
    # has not earned.
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    changed, refused = rewrite(doc, args.family, args.to)
    if not changed and not refused:
        return False
    print(f"--- {path.relative_to(WORKFLOWS)}")
    for line in changed:
        print(f"    swap    {line}")
    for line in refused:
        print(f"    REFUSED {line}")
    if not changed or args.dry_run:
        return bool(changed)
    out = path.with_suffix(".nunchaku.json")
    if out.exists():
        print(f"    refusing to overwrite {out.name}; delete it first")
        return False
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    written {out.name}")
    return True


def main() -> int:
    args = parse_args()
    model_dirs = [PORTABLE_ROOT / "ComfyUI" / "models" / "diffusion_models",
                  Path("D:/ComfyUI-Models/diffusion_models")]
    if not any((d / args.to).is_file() for d in model_dirs):
        # Writing a workflow that points at a file which is not there just moves the failure to
        # the moment the user presses Queue.
        print(f"'{args.to}' is not in any diffusion_models directory; refusing to write a "
              f"workflow that cannot run")
        return 1

    paths = sorted(WORKFLOWS.rglob("*.json")) if args.all else [WORKFLOWS / args.workflow]
    touched = 0
    for path in paths:
        if not path.is_file():
            print(f"{path} not found")
            return 1
        try:
            touched += bool(process(path, args))
        except Exception as exc:  # a malformed workflow should not stop the rest
            print(f"--- {path.name}: skipped ({type(exc).__name__}: {exc})")
    print(f"\n{touched} workflow(s) {'would be' if args.dry_run else ''} rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
