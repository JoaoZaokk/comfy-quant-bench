"""Build the quality-focused ComfyUI workflow suite without touching originals."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "ComfyUI" / "user" / "default" / "workflows"
OUT = WORKFLOWS / "ULTRA_IMAGE_VIDEO"
CUSTOM = ROOT / "ComfyUI" / "custom_nodes"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(relative: str, document: dict) -> None:
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2)
    if target.exists() and target.read_text(encoding="utf-8-sig") == payload:
        print(f"unchanged: {target}")
        return
    target.write_text(payload, encoding="utf-8")
    print(target)


def nodes(document: dict):
    for graph in [document, *(document.get("definitions", {}).get("subgraphs", []) or [])]:
        yield from graph.get("nodes", []) or []


def find(document: dict, node_id: int) -> dict:
    for node in nodes(document):
        if node.get("id") == node_id:
            return node
    raise KeyError(node_id)


def normalize_links(document: dict) -> None:
    by_id = {node["id"]: node for node in document["nodes"]}
    for node in document["nodes"]:
        for item in node.get("inputs", []) or []:
            item["link"] = None
        for item in node.get("outputs", []) or []:
            item["links"] = []
    for link in document["links"]:
        link_id, origin_id, origin_slot, target_id, target_slot, _ = link
        by_id[target_id]["inputs"][target_slot]["link"] = link_id
        output = by_id[origin_id]["outputs"][origin_slot]
        output.setdefault("links", []).append(link_id)


def build_classic_upscalers() -> None:
    source = load(OUT / "UPSCALE_PHOTO_4X_3080TI_FAST.json")
    document = deepcopy(source)
    find(document, 2)["type"] = "UpscaleModelLoader"
    find(document, 2)["widgets_values"] = ["4xRealWebPhoto_v4_dat2.pth"]
    find(document, 3)["type"] = "ImageUpscaleWithModel"
    find(document, 3)["widgets_values"] = []
    find(document, 4)["widgets_values"] = ["ULTRA_UPSCALE/DAT_RealWebPhoto_4x"]
    save("UPSCALE/UPSCALE_DAT_4X_PHOTO_QUALITY_3090.json", document)

    document = deepcopy(source)
    find(document, 2)["type"] = "UpscaleModelLoader"
    find(document, 2)["widgets_values"] = ["4x_NMKD-Siax_200k.pth"]
    find(document, 3)["type"] = "ImageUpscaleWithModel"
    find(document, 3)["widgets_values"] = []
    find(document, 4)["widgets_values"] = ["ULTRA_UPSCALE/NMKD_Siax_4x"]
    save("UPSCALE/UPSCALE_NMKD_SIAX_4X_BALANCED_3090.json", document)


def build_seedvr2() -> None:
    image_source = load(WORKFLOWS / "SeedVR2" / "SeedVR2_simple_image_upscale.json")
    document = deepcopy(image_source)
    find(document, 13)["widgets_values"] = [
        "ema_vae_fp16.safetensors", "cuda:1", True, 1024, 128,
        True, 1024, 128, "false", "none", False,
    ]
    find(document, 14)["widgets_values"] = [
        "seedvr2_ema_7b_sharp_fp16.safetensors", "cuda:0", 8,
        False, "cpu", True, "sageattn_2",
    ]
    find(document, 10)["widgets_values"] = [
        42, "fixed", 2048, 4096, 1, False, "lab", 0, 0, 0, 0, "cpu", False,
    ]
    find(document, 15)["widgets_values"] = ["ULTRA_UPSCALE/SeedVR2_7B_Sharp_Quality"]
    save("UPSCALE/UPSCALE_SEEDVR2_IMAGE_7B_SHARP_QUALITY_DUALGPU.json", document)

    video_source = load(
        WORKFLOWS / "SeedVR2" / "JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_OPTIMIZED.json"
    )
    document = deepcopy(video_source)
    find(document, 13)["widgets_values"] = [
        "ema_vae_fp16.safetensors", "cuda:1", True, 1024, 128,
        True, 768, 128, "false", "none", False,
    ]
    find(document, 14)["widgets_values"] = [
        "seedvr2_ema_7b_fp16.safetensors", "cuda:0", 8,
        False, "cpu", True, "sageattn_2",
    ]
    find(document, 10)["widgets_values"] = [
        42, "fixed", 1080, 2160, 5, True, "lab", 3, 0, 0, 0, "cpu", False,
    ]
    find(document, 23)["widgets_values"][0] = "ULTRA_UPSCALE/SeedVR2_7B_Video_Quality"
    save("UPSCALE/UPSCALE_SEEDVR2_VIDEO_7B_QUALITY_DUALGPU.json", document)


def build_flashvsr() -> None:
    source = load(
        WORKFLOWS / "SeedVR2" / "JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_OPTIMIZED.json"
    )
    sample = load(CUSTOM / "ComfyUI-FlashVSR_Stable" / "workflow" / "FlashVSR.json")
    init = deepcopy(next(node for node in sample["nodes"] if node["id"] == 3))
    run = deepcopy(next(node for node in sample["nodes"] if node["id"] == 2))
    init.update(id=14, pos=[540, 760], order=3)
    init["widgets_values"] = [
        "FlashVSR-v1.1", "tiny", "Wan2.2", False,
        "bf16", "cuda:1", "sparse_sage_attention",
    ]
    run.update(id=10, pos=[850, 500], order=5)
    run["widgets_values"] = [
        2, True, True, True, 384, 48, False,
        2.0, 3.0, 11, 42, "fixed", 48, False, False, 1.0,
    ]
    document = deepcopy(source)
    document["nodes"] = [
        node for node in document["nodes"] if node["id"] not in {10, 13, 14, 19, 20}
    ] + [init, run]
    document["links"] = [
        link for link in document["links"] if link[0] not in {10, 11, 16, 18, 19, 22}
    ]
    document["links"].extend(
        [
            [101, 14, 0, 10, 0, "PIPE"],
            [102, 17, 0, 10, 1, "IMAGE"],
            [103, 10, 0, 24, 0, "IMAGE"],
        ]
    )
    find(document, 23)["widgets_values"][0] = "ULTRA_UPSCALE/FlashVSR_2x_Fast"
    document["last_link_id"] = 103
    normalize_links(document)
    save("UPSCALE/UPSCALE_FLASHVSR_VIDEO_2X_FAST_3080TI.json", document)


def build_big_lama() -> None:
    source = load(CUSTOM / "comfyui-inpaint-nodes" / "workflows" / "inpaint-preprocess.json")
    keep = {33, 49, 52, 59}
    document = deepcopy(source)
    document["nodes"] = [deepcopy(node) for node in source["nodes"] if node["id"] in keep]
    document["links"] = [
        [1, 52, 0, 49, 0, "INPAINT_MODEL"],
        [2, 33, 0, 49, 1, "IMAGE"],
        [3, 33, 1, 49, 2, "MASK"],
        [4, 49, 0, 59, 0, "IMAGE"],
    ]
    find(document, 33)["widgets_values"] = ["3.webp", "image"]
    find(document, 52)["widgets_values"] = ["big-lama.pt"]
    find(document, 49)["widgets_values"] = [0, "randomize"]
    find(document, 59)["widgets_values"] = ["ULTRA_INPAINT/BigLama_Fast_Erase"]
    document["last_node_id"] = 59
    document["last_link_id"] = 4
    document["groups"] = []
    normalize_links(document)
    save("INPAINT_OUTPAINT/INPAINT_BIG_LAMA_FAST_ERASE.json", document)


def configure_checkpoint_loader(node: dict) -> None:
    node["type"] = "CheckpointLoaderAdvancedMultiGPU"
    node["widgets_values"] = ["Juggernaut-XL_v9.safetensors", "cuda:0", "cuda:1", "cuda:1"]
    node.setdefault("properties", {})["Node name for S&R"] = "CheckpointLoaderAdvancedMultiGPU"


def build_fooocus() -> None:
    source = load(CUSTOM / "comfyui-inpaint-nodes" / "workflows" / "inpaint-simple.json")
    document = deepcopy(source)
    configure_checkpoint_loader(find(document, 19))
    find(document, 33)["widgets_values"] = ["3.webp", "image"]
    find(document, 35)["widgets_values"] = ["fooocus_inpaint_head.pth", "inpaint_v26.fooocus.patch"]
    find(document, 9)["widgets_values"] = [
        "photorealistic seamless replacement, natural texture, matching light, matching perspective"
    ]
    find(document, 10)["widgets_values"] = [
        "illustration, cartoon, plastic skin, blurry, malformed, visible seam, watermark"
    ]
    find(document, 28)["type"] = "SaveImage"
    find(document, 28)["widgets_values"] = ["ULTRA_INPAINT/Fooocus_SDXL_Quality"]
    save("INPAINT_OUTPAINT/INPAINT_FOOOCUS_SDXL_QUALITY_DUALGPU.json", document)

    source = load(CUSTOM / "comfyui-inpaint-nodes" / "workflows" / "outpaint.json")
    document = deepcopy(source)
    configure_checkpoint_loader(find(document, 19))
    find(document, 33)["widgets_values"] = ["3.webp", "image"]
    find(document, 35)["widgets_values"] = ["fooocus_inpaint_head.pth", "inpaint_v26.fooocus.patch"]
    find(document, 67)["widgets_values"] = [256, 128, 256, 128, 64]
    find(document, 9)["widgets_values"] = [
        "photorealistic continuation of the same scene, consistent lighting, lens, perspective and depth"
    ]
    find(document, 10)["widgets_values"] = [
        "illustration, cartoon, blurry, visible seam, duplicated subject, watermark"
    ]
    find(document, 59)["widgets_values"] = ["ULTRA_OUTPAINT/Fooocus_SDXL_Quality"]
    save("INPAINT_OUTPAINT/OUTPAINT_FOOOCUS_SDXL_QUALITY_DUALGPU.json", document)


def build_ultimate_sdxl() -> None:
    source = load(CUSTOM / "comfyui_ultimatesdupscale" / "example_workflows" / "basic-usdu.json")
    inpaint_source = load(CUSTOM / "comfyui-inpaint-nodes" / "workflows" / "inpaint-preprocess.json")
    load_image = deepcopy(next(node for node in inpaint_source["nodes"] if node["id"] == 33))
    load_image.update(id=8, pos=[50, 650], order=0)
    load_image["widgets_values"] = ["3.webp", "image"]
    document = deepcopy(source)
    keep = {4, 6, 7, 17, 18, 19}
    document["nodes"] = [deepcopy(node) for node in source["nodes"] if node["id"] in keep] + [load_image]
    configure_checkpoint_loader(find(document, 4))
    find(document, 6)["widgets_values"] = [
        "photorealistic commercial photo, preserve identity, exact product geometry and text, natural microtexture"
    ]
    find(document, 7)["widgets_values"] = [
        "changed face, changed logo, fake text, illustration, CGI, plastic skin, oversharpening, halos"
    ]
    find(document, 18)["widgets_values"] = ["4xRealWebPhoto_v4_dat2.pth"]
    find(document, 17)["widgets_values"] = [
        2, 0, "fixed", 20, 5.5, "dpmpp_2m_sde_gpu", "karras", 0.18,
        "Chess", 1024, 1024, 16, 64, "None", 0.12, 128, 16, 32, True, True,
    ]
    output = find(document, 19)
    output["type"] = "SaveImage"
    output["widgets_values"] = ["ULTRA_UPSCALE/Ultimate_SDXL_Creative"]
    document["links"] = [
        [3, 4, 1, 6, 0, "CLIP"],
        [5, 4, 1, 7, 0, "CLIP"],
        [11, 8, 0, 17, 0, "IMAGE"],
        [12, 4, 0, 17, 1, "MODEL"],
        [13, 6, 0, 17, 2, "CONDITIONING"],
        [14, 7, 0, 17, 3, "CONDITIONING"],
        [15, 4, 2, 17, 4, "VAE"],
        [16, 18, 0, 17, 5, "UPSCALE_MODEL"],
        [17, 17, 0, 19, 0, "IMAGE"],
    ]
    document["groups"] = []
    normalize_links(document)
    save("UPSCALE/UPSCALE_ULTIMATE_SDXL_CREATIVE_DUALGPU.json", document)


def build_qwen_fast() -> None:
    source = load(OUT / "QWEN_EDIT_2511_NUNCHAKU_FP8_DUALGPU_3090_3080TI.json")
    document = deepcopy(source)
    find(document, 3)["widgets_values"] = [335442696528418, "randomize", 4, 1.0, "euler", "simple", 1.0]
    find(document, 117)["widgets_values"] = [
        "1", 1, True, "auto", True, False,
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 1.0,
    ]
    find(document, 119)["widgets_values"] = ["ULTRA_QWEN/Qwen_Edit_2511_4Step_Fast"]
    save("QWEN/QWEN_EDIT_2511_4STEP_FAST_DUALGPU.json", document)


def adapt_ltx25(document: dict, output_prefix: str) -> dict:
    document = deepcopy(document)
    for node in nodes(document):
        values = node.get("widgets_values") or []
        for index, value in enumerate(values):
            if value == "gemma4_e2b_it_bf16.safetensors":
                values[index] = "gemma4_e2b_it_int8_convrot.safetensors"
            elif value == "ltx-2.3-22b-dev.safetensors":
                values[index] = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
        if node.get("type") == "UNETLoader" and values and "ltx-2.5" in str(values[0]):
            node["type"] = "UNETLoaderDisTorch2MultiGPU"
            node["widgets_values"] = [
                "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                "default", "cuda:0", 17.0, "cpu", "", True,
            ]
        elif node.get("type") == "CLIPLoader" and values:
            if "with-proj-ltx-2.5" in str(values[0]):
                filename = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
                virtual_vram = 9.0
            elif "gemma4_e2b" in str(values[0]):
                filename = "gemma4_e2b_it_int8_convrot.safetensors"
                virtual_vram = 6.0
            else:
                continue
            node["type"] = "CLIPLoaderDisTorch2MultiGPU"
            node["widgets_values"] = [filename, "ltxv", "cuda:1", virtual_vram, "cpu", "", True]
        elif node.get("type") == "VAELoader" and values:
            node["type"] = "VAELoaderMultiGPU"
            node["widgets_values"] = [values[0], "cuda:1"]
        elif node.get("type") == "LatentUpscaleModelLoader":
            node["type"] = "LatentUpscaleModelLoaderMultiGPU"
            node["widgets_values"] = [
                "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "cuda:1"
            ]
        elif node.get("type") == "SaveVideo" and values:
            values[0] = output_prefix
    return document


def build_ltx25() -> None:
    base = CUSTOM / "ComfyUI-LTXVideo" / "example_workflows" / "2.5"
    inpaint = load(base / "LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json")
    outpaint = load(base / "LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json")
    save(
        "LTX/LTX25_VIDEO_INPAINT_TWO_STAGE_DUALGPU.json",
        adapt_ltx25(inpaint, "ULTRA_LTX25/Inpaint_Two_Stage"),
    )
    save(
        "LTX/LTX25_VIDEO_OUTPAINT_TWO_STAGE_DUALGPU.json",
        adapt_ltx25(outpaint, "ULTRA_LTX25/Outpaint_Two_Stage"),
    )


def build_zimage(kind: str) -> None:
    source = load(WORKFLOWS / "image_z_image_turbo_fun_union_controlnet.nunchaku.json")
    subgraph = source["definitions"]["subgraphs"][0]
    document = {key: deepcopy(value) for key, value in source.items() if key not in {"nodes", "links", "definitions"}}
    document["nodes"] = deepcopy(subgraph["nodes"])
    document["links"] = [
        [link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"], link["type"]]
        for link in subgraph["links"]
        if link["origin_id"] >= 0 and link["target_id"] >= 0
    ]
    document["links"] = [link for link in document["links"] if link[0] not in {65, 80}]
    load_image = deepcopy(next(node for node in source["nodes"] if node["id"] == 58))
    load_image.update(id=100, pos=[-900, 300], order=0)
    load_image["widgets_values"] = ["3.webp", "image"]
    save_template = load(OUT / "UPSCALE_PHOTO_4X_3080TI_FAST.json")
    save_node = deepcopy(find(save_template, 4))
    save_node.update(id=102, pos=[1500, 500], order=20)
    prefix = "ULTRA_ZIMAGE/Inpaint_Union21" if kind == "inpaint" else "ULTRA_ZIMAGE/Outpaint_Union21"
    save_node["widgets_values"] = [prefix]
    document["nodes"].extend([load_image, save_node])

    control = find(document, 60)
    control["type"] = "ZImageFunControlnet"
    control["inputs"] = [
        {"name": "model", "type": "MODEL", "link": 79},
        {"name": "model_patch", "type": "MODEL_PATCH", "link": 74},
        {"name": "vae", "type": "VAE", "link": 70},
        {"name": "strength", "type": "FLOAT", "widget": {"name": "strength"}, "link": None},
        {"name": "image", "type": "IMAGE", "shape": 7, "link": None},
        {"name": "inpaint_image", "type": "IMAGE", "shape": 7, "link": None},
        {"name": "mask", "type": "MASK", "shape": 7, "link": None},
    ]
    control["widgets_values"] = [0.75]
    find(document, 39)["type"] = "CLIPLoaderMultiGPU"
    find(document, 39)["widgets_values"] = ["qwen_3_4b.safetensors", "lumina2", "cuda:1"]
    find(document, 40)["type"] = "VAELoaderMultiGPU"
    find(document, 40)["widgets_values"] = ["ae.safetensors", "cuda:1"]
    find(document, 64)["widgets_values"] = ["Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"]
    find(document, 44)["widgets_values"] = [580358853373997, "randomize", 8, 1.0, "res_multistep", "simple", 1.0]
    find(document, 45)["widgets_values"] = [
        "Photorealistic seamless edit matching the original identity, lighting, lens, perspective, texture and color. Natural skin and fabric detail, physically plausible shadows."
    ]

    next_link = max(link[0] for link in document["links"]) + 1
    if kind == "outpaint":
        fooocus = load(CUSTOM / "comfyui-inpaint-nodes" / "workflows" / "outpaint.json")
        pad = deepcopy(find(fooocus, 67))
        pad.update(id=101, pos=[-480, 300], order=1)
        pad["widgets_values"] = [256, 128, 256, 128, 64]
        document["nodes"].append(pad)
        document["links"].extend(
            [
                [next_link, 100, 0, 101, 0, "IMAGE"],
                [next_link + 1, 101, 0, 60, 5, "IMAGE"],
                [next_link + 2, 101, 1, 60, 6, "MASK"],
                [next_link + 3, 101, 0, 69, 0, "IMAGE"],
            ]
        )
        next_link += 4
    else:
        document["links"].extend(
            [
                [next_link, 100, 0, 60, 5, "IMAGE"],
                [next_link + 1, 100, 1, 60, 6, "MASK"],
                [next_link + 2, 100, 0, 69, 0, "IMAGE"],
            ]
        )
        next_link += 3
    document["links"].append([next_link, 43, 0, 102, 0, "IMAGE"])
    document["last_node_id"] = 102
    document["last_link_id"] = next_link
    document["groups"] = []
    normalize_links(document)
    filename = "INPAINT_ZIMAGE_UNION21_NUNCHAKU_DUALGPU.json" if kind == "inpaint" else "OUTPAINT_ZIMAGE_UNION21_NUNCHAKU_DUALGPU.json"
    save(f"INPAINT_OUTPAINT/{filename}", document)


def main() -> None:
    build_classic_upscalers()
    build_seedvr2()
    build_flashvsr()
    build_big_lama()
    build_fooocus()
    build_ultimate_sdxl()
    build_qwen_fast()
    build_ltx25()
    build_zimage("inpaint")
    build_zimage("outpaint")


if __name__ == "__main__":
    main()
