"""Build a corrected, staged VOID workflow without modifying the source JSON."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def node(graph, node_id):
    return next(item for item in graph["nodes"] if item["id"] == node_id)


def subgraph(workflow, name):
    return next(item for item in workflow["definitions"]["subgraphs"] if item["name"] == name)


def link(graph, link_id):
    return next(item for item in graph["links"] if item["id"] == link_id)


def next_link_id(graph):
    return max((item["id"] for item in graph["links"]), default=0) + 1


def next_node_id(graph):
    return max((item["id"] for item in graph["nodes"] if isinstance(item["id"], int)), default=0) + 1


def add_link(graph, origin_id, origin_slot, target_id, target_slot, value_type):
    link_id = next_link_id(graph)
    graph["links"].append(
        {
            "id": link_id,
            "origin_id": origin_id,
            "origin_slot": origin_slot,
            "target_id": target_id,
            "target_slot": target_slot,
            "type": value_type,
        }
    )
    return link_id


def add_output_link(graph, node_id, output_slot, link_id):
    outputs = node(graph, node_id)["outputs"]
    links = outputs[output_slot].get("links") or []
    if link_id not in links:
        links.append(link_id)
    outputs[output_slot]["links"] = links


def mapped_widgets(graph_node):
    result = {}
    widget_index = 0
    for item in graph_node.get("inputs", []):
        if item.get("widget") is None:
            continue
        result[item["name"]] = widget_index
        widget_index += 1
    return result


def deno_unload_node(node_id, positive_link, clip_link, negative_link, positive_out, negative_out):
    return {
        "id": node_id,
        "type": "DenoTextEncoderUnload",
        "pos": [390, 250],
        "size": [330, 150],
        "flags": {},
        "order": 40,
        "mode": 0,
        "inputs": [
            {"name": "positive_conditioning", "type": "CONDITIONING", "link": positive_link},
            {"name": "text_encoder", "type": "CLIP", "link": clip_link},
            {"name": "negative_conditioning", "type": "CONDITIONING", "link": negative_link},
        ],
        "outputs": [
            {"name": "Positive Conditioning", "type": "CONDITIONING", "slot_index": 0, "links": [positive_out]},
            {"name": "Negative Conditioning", "type": "CONDITIONING", "slot_index": 1, "links": [negative_out]},
        ],
        "title": "Unload T5 after both prompt encodes",
        "properties": {"Node name for S&R": "DenoTextEncoderUnload"},
    }


def model_image_barrier(node_id, image_link, model_link, output_links):
    return {
        "id": node_id,
        "type": "VoidUnloadModelImage",
        "pos": [1510, -160],
        "size": [330, 130],
        "flags": {},
        "order": 41,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": image_link},
            {"name": "model", "type": "MODEL", "link": model_link},
            {"name": "stage", "type": "STRING", "widget": {"name": "stage"}, "link": None},
        ],
        "outputs": [{"name": "image", "type": "IMAGE", "slot_index": 0, "links": output_links}],
        "title": "HARD UNLOAD VOID Pass 1",
        "properties": {"Node name for S&R": "VoidUnloadModelImage"},
        "widgets_values": ["after_pass_1"],
        "widgets_values_named": {"stage": "after_pass_1"},
    }


def model_latent_barrier(node_id, latent_link, model_link, output_links):
    return {
        "id": node_id,
        "type": "VoidUnloadModelLatent",
        "pos": [1160, 1010],
        "size": [330, 130],
        "flags": {},
        "order": 43,
        "mode": 0,
        "inputs": [
            {"name": "latent", "type": "LATENT", "link": latent_link},
            {"name": "model", "type": "MODEL", "link": model_link},
            {"name": "stage", "type": "STRING", "widget": {"name": "stage"}, "link": None},
        ],
        "outputs": [{"name": "latent", "type": "LATENT", "slot_index": 0, "links": output_links}],
        "title": "HARD UNLOAD VOID Pass 2",
        "properties": {"Node name for S&R": "VoidUnloadModelLatent"},
        "widgets_values": ["after_pass_2"],
        "widgets_values_named": {"stage": "after_pass_2"},
    }


def optical_flow_barrier(node_id, latent_link, optical_flow_link, output_links):
    return {
        "id": node_id,
        "type": "VoidUnloadOpticalFlowLatent",
        "pos": [730, 1110],
        "size": [330, 130],
        "flags": {},
        "order": 42,
        "mode": 0,
        "inputs": [
            {"name": "latent", "type": "LATENT", "link": latent_link},
            {"name": "optical_flow", "type": "OPTICAL_FLOW", "link": optical_flow_link},
            {"name": "stage", "type": "STRING", "widget": {"name": "stage"}, "link": None},
        ],
        "outputs": [{"name": "latent", "type": "LATENT", "slot_index": 0, "links": output_links}],
        "title": "HARD UNLOAD RAFT",
        "properties": {"Node name for S&R": "VoidUnloadOpticalFlowLatent"},
        "widgets_values": ["after_raft"],
        "widgets_values_named": {"stage": "after_raft"},
    }


def patch_workflow(workflow, vae_device, duration_seconds=None, output_prefix=None):
    inner = subgraph(workflow, "Video Inpaint (VOID)")
    outer = subgraph(workflow, "VOID_SAM3_Video_Inpaint_DualGPU")

    # Keep the embedded operator notes aligned with the actual device routing.
    note_replacements = {
        "VOID Pass 1 + Pass 2 and CogVideoX VAE: **GPU 0 / RTX 3090**.":
            "VOID Pass 1 + Pass 2: **GPU 0 / RTX 3090**; CogVideoX VAE: **GPU 1 / RTX 3080 Ti**.",
        "T5 uses the FP8 scaled checkpoint on GPU 1; VOID/VAE stay on GPU 0.":
            "T5 and the CogVideoX VAE use GPU 1; both VOID passes use GPU 0.",
    }
    for note_id in (170, 171):
        note = node(workflow, note_id)
        for old, new in note_replacements.items():
            note["widgets_values"][0] = note["widgets_values"][0].replace(old, new)
            note["widgets_values_named"]["text"] = note["widgets_values_named"]["text"].replace(old, new)

    # Load T5 directly on the 3080 Ti. The core CLIPLoader + downstream
    # SelectCLIPDevice path clones the full text encoder, which this install
    # reports as a circularly referenced model and retains under RAM pressure.
    clip_loader = node(inner, 2)
    clip_loader["type"] = "CLIPLoaderMultiGPU"
    clip_loader["title"] = "Load T5 directly -> cuda:1 (no deep clone)"
    clip_loader["properties"]["Node name for S&R"] = "CLIPLoaderMultiGPU"
    clip_loader["widgets_values"][2] = "cuda:1"
    clip_loader["widgets_values_named"]["device"] = "cuda:1"

    clip_selector = node(inner, 220)
    clip_selector["mode"] = 4
    clip_selector["title"] = "BYPASSED — T5 already loaded directly on cuda:1"

    exposed_void = node(outer, 211)
    widget_map = mapped_widgets(exposed_void)
    exposed_void["widgets_values"][widget_map["void_unet_pass2"]] = "void_pass2.safetensors"
    exposed_void["widgets_values"][widget_map["clip_name"]] = "t5xxl_fp8_e4m3fn_scaled.safetensors"
    if "widgets_values_named" in exposed_void:
        exposed_void["widgets_values_named"]["void_unet_pass2"] = "void_pass2.safetensors"
        exposed_void["widgets_values_named"]["clip_name"] = "t5xxl_fp8_e4m3fn_scaled.safetensors"

    if duration_seconds is not None:
        exposed_void["widgets_values"][widget_map["duration_seconds"]] = duration_seconds
        if "widgets_values_named" in exposed_void:
            exposed_void["widgets_values_named"]["duration_seconds"] = duration_seconds

        root_void = node(workflow, 167)
        root_widget_map = mapped_widgets(root_void)
        root_void["widgets_values"][root_widget_map["duration_seconds"]] = duration_seconds
        if "widgets_values_named" in root_void:
            root_void["widgets_values_named"]["duration_seconds"] = duration_seconds

    if output_prefix is not None:
        node(workflow, 38)["widgets_values"][0] = output_prefix

    node(inner, 221)["widgets_values"] = [vae_device]
    node(inner, 221)["widgets_values_named"] = {"device": vae_device}
    node(inner, 221)["title"] = f"CogVideoX VAE -> {vae_device}"

    # Fixed ROI mode: keep the reusable SAM3 subgraph definition, but make both
    # invocations inactive in this production workflow.
    node(inner, 149)["mode"] = 2
    node(outer, 195)["mode"] = 2
    node(outer, 196)["mode"] = 2
    node(outer, 199)["widgets_values"] = [False]

    # The frontend validates every connected switch input even when the fixed
    # branch is selected. Feed the dormant auto inputs from the fixed values so
    # the specialized workflow stays valid without executing SAM3/bbox nodes.
    fixed_fallbacks = {
        434: 225,  # crop X -> switch on_true
        437: 192,  # crop Y -> switch on_true
        430: 193,  # crop width -> snap /8
        431: 194,  # crop height -> snap /8
    }
    bbox = node(outer, 196)
    for output in bbox["outputs"]:
        output["links"] = [item for item in (output.get("links") or []) if item not in fixed_fallbacks]
    for link_id, fixed_node_id in fixed_fallbacks.items():
        link(outer, link_id)["origin_id"] = fixed_node_id
        link(outer, link_id)["origin_slot"] = 0
        add_output_link(outer, fixed_node_id, 0, link_id)

    # Trim the lazy file-backed VIDEO before its first GetVideoComponents. The
    # original graph decoded the entire file, then selected the requested range
    # inside the inner VOID subgraph.
    slice_id = next_node_id(outer)
    zero_id = slice_id + 1
    recompose_id = slice_id + 2
    slice_start_link = add_link(outer, -10, 4, slice_id, 1, "INT")
    slice_duration_link = add_link(outer, -10, 5, slice_id, 2, "INT")
    sliced_video_link = add_link(outer, slice_id, 0, 222, 0, "VIDEO")
    recompose_source_link = add_link(outer, slice_id, 0, recompose_id, 0, "VIDEO")

    link(outer, 416)["target_id"] = slice_id
    link(outer, 416)["target_slot"] = 0
    link(outer, 460)["origin_id"] = slice_id
    link(outer, 460)["origin_slot"] = 0
    node(outer, 222)["inputs"][0]["link"] = sliced_video_link

    # The inner workflow must start at frame zero because the outer lazy slice
    # already applied the requested source-frame offset.
    link(outer, 418)["origin_id"] = zero_id
    link(outer, 418)["origin_slot"] = 0
    link(outer, 466)["origin_id"] = zero_id
    link(outer, 466)["origin_slot"] = 0

    for item in outer["inputs"]:
        if item["name"] == "source_video":
            item["linkIds"] = [416]
        elif item["name"] == "start_frame_index":
            item["linkIds"] = [slice_start_link]
        elif item["name"] == "duration_seconds":
            item["linkIds"] = [467, slice_duration_link]

    slice_node = {
        "id": slice_id,
        "type": "VoidSliceVideoByFrame",
        "pos": [-2320, -160],
        "size": [330, 150],
        "flags": {},
        "order": 1,
        "mode": 0,
        "inputs": [
            {"name": "video", "type": "VIDEO", "link": 416},
            {"name": "start_frame_index", "type": "INT", "link": slice_start_link},
            {"name": "duration_seconds", "type": "INT", "link": slice_duration_link},
        ],
        "outputs": [
            {
                "name": "video",
                "type": "VIDEO",
                "slot_index": 0,
                "links": [sliced_video_link, 460, recompose_source_link],
            }
        ],
        "title": "LAZY SLICE before any frame decode",
        "properties": {"Node name for S&R": "VoidSliceVideoByFrame"},
    }

    zero_node = copy.deepcopy(node(outer, 225))
    zero_node["id"] = zero_id
    zero_node["pos"] = [-1980, 80]
    zero_node["title"] = "Inner start frame = 0 (already sliced)"
    zero_node["inputs"][0]["link"] = None
    zero_node["outputs"][0]["links"] = [418, 466]
    zero_node["widgets_values"] = [0, "fixed"]
    zero_node["widgets_values_named"] = {"value": 0, "fixed": "fixed"}

    # The original outer graph retained decoded full-frame source tensors for
    # both pass-1 and pass-2 recomposition. Keep pass 1 raw and decode the
    # trimmed source only after pass 2/model unload, then paste in place once.
    for node_id in (204, 205, 206, 212, 213, 214, 215, 216, 217):
        node(outer, node_id)["mode"] = 2

    link(outer, 496)["origin_id"] = 211
    link(outer, 496)["origin_slot"] = 0
    add_output_link(outer, 211, 0, 496)
    node(outer, 216)["outputs"][0]["links"] = []

    link(outer, 479)["target_id"] = recompose_id
    link(outer, 479)["target_slot"] = 1
    node(outer, 213)["inputs"][0]["link"] = None
    link(outer, 525)["target_id"] = recompose_id
    link(outer, 525)["target_slot"] = 2
    node(outer, 215)["inputs"][3]["link"] = None
    link(outer, 529)["target_id"] = recompose_id
    link(outer, 529)["target_slot"] = 3
    node(outer, 215)["inputs"][4]["link"] = None
    link(outer, 499)["origin_id"] = recompose_id
    link(outer, 499)["origin_slot"] = 0
    node(outer, 217)["outputs"][0]["links"] = []

    recompose_node = {
        "id": recompose_id,
        "type": "VoidRecomposeCroppedVideo",
        "pos": [1740, 980],
        "size": [360, 170],
        "flags": {},
        "order": 50,
        "mode": 0,
        "inputs": [
            {"name": "source_video", "type": "VIDEO", "link": recompose_source_link},
            {"name": "edited_video", "type": "VIDEO", "link": 479},
            {"name": "x", "type": "INT", "link": 525},
            {"name": "y", "type": "INT", "link": 529},
        ],
        "outputs": [{"name": "video", "type": "VIDEO", "slot_index": 0, "links": [499]}],
        "title": "Decode source after Pass 2 + paste crop in place",
        "properties": {"Node name for S&R": "VoidRecomposeCroppedVideo"},
    }
    outer["nodes"].extend([slice_node, zero_node, recompose_node])

    # The pass-1 full-frame preview doubles final recomposition work and is not
    # part of the production output. It remains visible but disabled.
    preview = node(workflow, 48)
    preview["mode"] = 2
    preview["title"] = "DISABLED for low RAM — optional Pass 1 preview"

    # T5 barrier: positive and negative encodes must both complete before the
    # exact CLIP patcher is unloaded from GPU 1.
    deno_id = next_node_id(inner)
    clip_link = add_link(inner, 220, 0, deno_id, 1, "CLIP")
    positive_out = add_link(inner, deno_id, 0, 10, 0, "CONDITIONING")
    negative_out = add_link(inner, deno_id, 1, 10, 1, "CONDITIONING")
    link(inner, 8)["target_id"] = deno_id
    link(inner, 8)["target_slot"] = 0
    link(inner, 9)["target_id"] = deno_id
    link(inner, 9)["target_slot"] = 2
    node(inner, 10)["inputs"][0]["link"] = positive_out
    node(inner, 10)["inputs"][1]["link"] = negative_out
    add_output_link(inner, 220, 0, clip_link)
    inner["nodes"].append(deno_unload_node(deno_id, 8, clip_link, 9, positive_out, negative_out))

    # Pass 1 barrier after VAE decode, before RAFT and pass-1 video construction.
    pass1_id = next_node_id(inner)
    old_pass1_links = list(node(inner, 45)["outputs"][0]["links"])
    image_link = add_link(inner, 45, 0, pass1_id, 0, "IMAGE")
    model_link = add_link(inner, 190, 0, pass1_id, 1, "MODEL")
    for link_id in old_pass1_links:
        link(inner, link_id)["origin_id"] = pass1_id
        link(inner, link_id)["origin_slot"] = 0
    node(inner, 45)["outputs"][0]["links"] = [image_link]
    add_output_link(inner, 190, 0, model_link)
    inner["nodes"].append(model_image_barrier(pass1_id, image_link, model_link, old_pass1_links))

    # RAFT barrier after warped noise is materialized.
    raft_id = next_node_id(inner)
    old_raft_links = list(node(inner, 31)["outputs"][0]["links"])
    latent_link = add_link(inner, 31, 0, raft_id, 0, "LATENT")
    optical_link = add_link(inner, 142, 0, raft_id, 1, "OPTICAL_FLOW")
    for link_id in old_raft_links:
        link(inner, link_id)["origin_id"] = raft_id
        link(inner, link_id)["origin_slot"] = 0
    node(inner, 31)["outputs"][0]["links"] = [latent_link]
    add_output_link(inner, 142, 0, optical_link)
    inner["nodes"].append(optical_flow_barrier(raft_id, latent_link, optical_link, old_raft_links))

    # Pass 2 barrier before its VAE decode.
    pass2_id = next_node_id(inner)
    old_pass2_links = list(node(inner, 35)["outputs"][0]["links"])
    latent_link = add_link(inner, 35, 0, pass2_id, 0, "LATENT")
    model_link = add_link(inner, 191, 0, pass2_id, 1, "MODEL")
    for link_id in old_pass2_links:
        link(inner, link_id)["origin_id"] = pass2_id
        link(inner, link_id)["origin_slot"] = 0
    node(inner, 35)["outputs"][0]["links"] = [latent_link]
    add_output_link(inner, 191, 0, model_link)
    inner["nodes"].append(model_latent_barrier(pass2_id, latent_link, model_link, old_pass2_links))

    max_node = max(item["id"] for item in inner["nodes"] if isinstance(item["id"], int))
    max_link = max(item["id"] for item in inner["links"])
    for definition in workflow["definitions"]["subgraphs"]:
        definition["state"]["lastNodeId"] = max(definition["state"].get("lastNodeId", 0), max_node)
        definition["state"]["lastLinkId"] = max(definition["state"].get("lastLinkId", 0), max_link)

    return workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vae-device", choices=("gpu:0", "gpu:1"), default="gpu:1")
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    with args.source.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    workflow = patch_workflow(
        workflow,
        args.vae_device,
        duration_seconds=args.duration_seconds,
        output_prefix=args.output_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(workflow, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
