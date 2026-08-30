"""Build an optimized local workflow pack without touching source workflows.

The pack targets an RTX 3090 on cuda:0 for heavy diffusion models and an
RTX 3080 Ti on cuda:1 for text encoders, VAEs, and photo upscaling.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / ".scratch" / "workflow_pack_upstream"
WORKFLOWS = ROOT / "ComfyUI" / "user" / "default" / "workflows"
OUTPUT = WORKFLOWS / "ULTRA_IMAGE_VIDEO"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def all_nodes(document: dict):
    yield from document.get("nodes", []) or []
    for subgraph in document.get("definitions", {}).get("subgraphs", []) or []:
        yield from subgraph.get("nodes", []) or []


def write_json(name: str, document: dict) -> None:
    target = OUTPUT / name
    payload = json.dumps(document, ensure_ascii=False, indent=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and read_json(target) == document:
        print(f"unchanged: {target}")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing workflow: {target}")
    target.write_text(payload, encoding="utf-8")
    print(f"created: {target}")


def set_loader_metadata(node: dict, node_type: str, cnr_id: str) -> None:
    node["type"] = node_type
    properties = node.setdefault("properties", {})
    properties["Node name for S&R"] = node_type
    properties["cnr_id"] = cnr_id


def optimize_wan(source: str, target: str) -> None:
    document = read_json(UPSTREAM / source)
    counts = {"unet": 0, "clip": 0, "vae": 0}
    for node in all_nodes(document):
        node_type = node.get("type")
        values = node.get("widgets_values") or []
        if node_type == "UNETLoader":
            set_loader_metadata(node, "UNETLoaderDisTorch2MultiGPU", "comfyui-multigpu")
            node["widgets_values"] = [
                values[0],
                values[1],
                "cuda:0",
                17.0,
                "cpu",
                "",
                True,
            ]
            node["title"] = "Wan DiT -> RTX 3090 (17 GB VRAM + CPU offload)"
            counts["unet"] += 1
        elif node_type == "CLIPLoader":
            set_loader_metadata(node, "CLIPLoaderMultiGPU", "comfyui-multigpu")
            node["widgets_values"] = [values[0], values[1], "cuda:1"]
            node["title"] = "UMT5 FP8 -> RTX 3080 Ti"
            counts["clip"] += 1
        elif node_type == "VAELoader":
            set_loader_metadata(node, "VAELoaderMultiGPU", "comfyui-multigpu")
            vae_name = values if isinstance(values, str) else values[0]
            node["widgets_values"] = [vae_name, "cuda:1"]
            node["title"] = "Wan VAE -> RTX 3080 Ti"
            counts["vae"] += 1
    if counts != {"unet": 2, "clip": 1, "vae": 1}:
        raise RuntimeError(f"Unexpected Wan loader counts in {source}: {counts}")
    write_json(target, document)


def optimize_flux_fill(source: str, target: str) -> None:
    document = read_json(UPSTREAM / source)
    counts = {"unet": 0, "clip": 0, "vae": 0}
    for node in all_nodes(document):
        node_type = node.get("type")
        values = node.get("widgets_values") or []
        if node_type == "UNETLoader":
            set_loader_metadata(node, "UnetLoaderGGUFMultiGPU", "ComfyUI-GGUF")
            node["widgets_values"] = ["flux1-fill-dev-Q8_0.gguf", "cuda:0"]
            node["title"] = "FLUX Fill Q8 -> RTX 3090"
            counts["unet"] += 1
        elif node_type == "DualCLIPLoader":
            set_loader_metadata(node, "DualCLIPLoaderMultiGPU", "comfyui-multigpu")
            node["widgets_values"] = [
                "clip_l.safetensors",
                "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "flux",
                "cuda:1",
            ]
            node["title"] = "CLIP-L + T5 FP8 -> RTX 3080 Ti"
            counts["clip"] += 1
        elif node_type == "VAELoader":
            set_loader_metadata(node, "VAELoaderMultiGPU", "comfyui-multigpu")
            node["widgets_values"] = ["ae.safetensors", "cuda:1"]
            node["title"] = "FLUX VAE -> RTX 3080 Ti"
            counts["vae"] += 1
    if counts != {"unet": 1, "clip": 1, "vae": 1}:
        raise RuntimeError(f"Unexpected FLUX Fill loader counts in {source}: {counts}")
    write_json(target, document)


def optimize_qwen() -> None:
    source = WORKFLOWS / "Qwen_Edit_2511_DualGPU_3090_3080Ti_FIXED.json"
    document = read_json(source)
    changed = 0
    for node in all_nodes(document):
        if node.get("type") == "CLIPLoaderDisTorch2MultiGPU":
            set_loader_metadata(node, "CLIPLoaderMultiGPU", "comfyui-multigpu")
            node["widgets_values"] = [
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen_image",
                "cuda:1",
            ]
            node["title"] = "Qwen VL FP8 inteiro -> RTX 3080 Ti"
            changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one Qwen text encoder loader, changed {changed}")
    write_json("QWEN_EDIT_2511_NUNCHAKU_FP8_DUALGPU_3090_3080TI.json", document)


def build_photo_upscaler() -> None:
    document = {
        "id": "2813a7c7-a095-4d5b-b958-a5da0eeaa6af",
        "revision": 0,
        "last_node_id": 4,
        "last_link_id": 3,
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "pos": [0, 0],
                "size": [320, 430],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "slot_index": 0, "links": [2]},
                    {"name": "MASK", "type": "MASK", "slot_index": 1, "links": None},
                ],
                "properties": {"Node name for S&R": "LoadImage", "cnr_id": "comfy-core"},
                "widgets_values": ["example.png", "image"],
            },
            {
                "id": 2,
                "type": "multiGPU_UpscaleModelLoader",
                "pos": [360, 0],
                "size": [340, 90],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "slot_index": 0, "links": [1]}
                ],
                "title": "RealESRGAN x4 (compatível com multi-GPU)",
                "properties": {"Node name for S&R": "multiGPU_UpscaleModelLoader"},
                "widgets_values": ["RealESRGAN_x4plus.safetensors"],
            },
            {
                "id": 3,
                "type": "multiGPU_ImageUpscaleWithModelMultiGPU",
                "pos": [360, 150],
                "size": [360, 250],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": 1},
                    {"name": "image", "type": "IMAGE", "link": 2},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "slot_index": 0, "links": [3]}],
                "title": "Upscale em tiles -> RTX 3080 Ti",
                "properties": {"Node name for S&R": "multiGPU_ImageUpscaleWithModelMultiGPU"},
                "widgets_values": ["cuda:1", 1, 0.5, 512, 128, 32],
            },
            {
                "id": 4,
                "type": "SaveImage",
                "pos": [760, 0],
                "size": [420, 450],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 3}],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveImage", "cnr_id": "comfy-core"},
                "widgets_values": ["ULTRA_UPSCALE/RealESRGAN_4x"],
            },
        ],
        "links": [
            [1, 2, 0, 3, 0, "UPSCALE_MODEL"],
            [2, 1, 0, 3, 1, "IMAGE"],
            [3, 3, 0, 4, 0, "IMAGE"],
        ],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1.0, "offset": [40, 40]}},
        "version": 0.4,
    }
    write_json("UPSCALE_PHOTO_4X_3080TI_FAST.json", document)


def main() -> None:
    optimize_wan(
        "video_wan2_2_14B_t2v.json",
        "WAN22_T2V_14B_4STEP_DUALGPU_3090_3080TI.json",
    )
    optimize_wan(
        "video_wan2_2_14B_i2v.json",
        "WAN22_I2V_14B_4STEP_DUALGPU_3090_3080TI.json",
    )
    optimize_flux_fill(
        "flux_fill_inpaint_example.json",
        "FLUX_FILL_INPAINT_Q8_DUALGPU_3090_3080TI.json",
    )
    optimize_flux_fill(
        "flux_fill_outpaint_example.json",
        "FLUX_FILL_OUTPAINT_Q8_DUALGPU_3090_3080TI.json",
    )
    optimize_qwen()
    build_photo_upscaler()


if __name__ == "__main__":
    main()
