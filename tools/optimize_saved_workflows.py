"""Create conservative optimized copies of selected saved ComfyUI workflows.

Original workflow files are never changed. The transformations below only use
node classes and model choices registered by the running local ComfyUI.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "ComfyUI" / "user" / "default" / "workflows"


def all_nodes(document: dict):
    for graph in [*(document.get("definitions", {}).get("subgraphs", []) or []), document]:
        yield from graph.get("nodes", []) or []


def read(relative: str) -> dict:
    return json.loads((WORKFLOWS / relative).read_text(encoding="utf-8-sig"))


def write(relative: str, document: dict) -> None:
    target = WORKFLOWS / relative
    payload = json.dumps(document, ensure_ascii=False, indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8-sig") == payload:
            print(f"unchanged: {target}")
            return
        raise FileExistsError(f"Refusing to overwrite existing optimized copy: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    print(target)


def optimize_ltx25() -> None:
    document = read("LTX25-int8-acceptance.json")
    for node in all_nodes(document):
        if node.get("id") == 2 and node.get("type") == "CLIPLoaderMultiGPU":
            node["type"] = "CLIPLoaderDisTorch2MultiGPU"
            node["widgets_values"] = [
                "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "ltxv",
                "cuda:1",
                6.0,
                "cpu",
                "",
                True,
            ]
        elif node.get("id") in {5, 6} and node.get("type") == "VAELoader":
            node["type"] = "VAELoaderMultiGPU"
            node["widgets_values"] = [node["widgets_values"][0], "cuda:1"]
    write("LTX25-int8-acceptance_DualGPU_3090_3080Ti_OPTIMIZED.json", document)


def optimize_seedvr2() -> None:
    document = read("SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_safe_720p_Q8.json")
    for node in all_nodes(document):
        if node.get("type") == "SeedVR2LoadVAEModel":
            values = node["widgets_values"]
            values[1] = "cuda:1"
            values[9] = "none"
            values[10] = False
        elif node.get("type") == "SeedVR2LoadDiTModel":
            values = node["widgets_values"]
            values[1] = "cuda:1"
            values[2] = 0
            values[4] = "none"
            values[5] = False
            values[6] = "sageattn_2"
        elif node.get("type") == "SeedVR2VideoUpscaler":
            node["widgets_values"][-1] = False
    write("SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_OPTIMIZED.json", document)


def optimize_seedvr2_compile_profiles() -> None:
    source = read("SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_OPTIMIZED.json")

    for suffix, backend, mode in (
        ("INDUCTOR_DIT", "inductor", "default"),
        ("CUDAGRAPHS_DIT_EXPERIMENTAL", "cudagraphs", "default"),
    ):
        document = deepcopy(source)
        compile_node = None
        vae_loader = None
        for node in all_nodes(document):
            if node.get("type") == "SeedVR2TorchCompileSettings":
                compile_node = node
                node["mode"] = 0
                node["title"] = f"DiT compile: {backend} / {mode}"
                node["widgets_values"] = [backend, mode, False, False, 64, 128]
            elif node.get("type") == "SeedVR2LoadVAEModel":
                vae_loader = node

        if compile_node is None or vae_loader is None:
            raise RuntimeError("Expected SeedVR2 compile and VAE loader nodes were not found")

        vae_compile_link = vae_loader["inputs"][0].get("link")
        vae_loader["inputs"][0]["link"] = None
        if vae_compile_link is not None:
            document["links"] = [
                link for link in document.get("links", []) if link[0] != vae_compile_link
            ]
            compile_node["outputs"][0]["links"] = [
                link_id
                for link_id in (compile_node["outputs"][0].get("links") or [])
                if link_id != vae_compile_link
            ]

        write(
            f"SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_{suffix}.json",
            document,
        )


def optimize_zimage() -> None:
    document = read("Zimg-TXT2IMG-_multigpu.app.nunchaku.json")
    for node in all_nodes(document):
        if node.get("type") == "KSampler":
            node["widgets_values"][2] = 9
    write("Zimg-TXT2IMG-_multigpu.app.nunchaku_9step_OPTIMIZED.json", document)


def fix_qwen_clip(source: str, target: str) -> None:
    document = read(source)
    changed = 0
    for node in all_nodes(document):
        if node.get("type") == "CLIPLoader" and node.get("widgets_values"):
            if node["widgets_values"][0] == "qwen_2.5_vl_7b_fp8_scaled.safetensors":
                node["widgets_values"][0] = "qwen_2.5_vl_7b.safetensors"
                changed += 1
    if not changed:
        raise RuntimeError(f"Expected missing Qwen CLIP reference was not found in {source}")
    write(target, document)


def fix_ltx23_reference() -> None:
    document = read("Video-LTX2_MultiGPU.app.json")
    changed = 0
    for node in all_nodes(document):
        values = node.get("widgets_values") or []
        for index, value in enumerate(values):
            if value == "gemma_3_12B_it_heretic.safetensors":
                values[index] = "gemma_3_12B_it_heretic_fp8_e4m3fn.safetensors"
                changed += 1
    if not changed:
        raise RuntimeError("Expected missing LTX 2.3 text encoder reference was not found")
    write("Video-LTX2_MultiGPU_MODELREF_FIXED.json", document)


def main() -> None:
    optimize_ltx25()
    optimize_seedvr2()
    optimize_seedvr2_compile_profiles()
    optimize_zimage()
    fix_qwen_clip(
        "image_qwen_image_edit_2509_relight.nunchaku.json",
        "image_qwen_image_edit_2509_relight.nunchaku_MODELREF_FIXED.json",
    )
    fix_qwen_clip(
        "templates-image_to_real.nunchaku.json",
        "templates-image_to_real.nunchaku_MODELREF_FIXED.json",
    )
    fix_ltx23_reference()


if __name__ == "__main__":
    main()
