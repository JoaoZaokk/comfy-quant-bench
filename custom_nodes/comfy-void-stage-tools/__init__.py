"""Small, targeted memory barriers for the local VOID workflow."""

from __future__ import annotations

import gc
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

import comfy.model_management as model_management
import folder_paths
from comfy_api.latest import InputImpl, Types


def _loaded_models() -> list[dict[str, str]]:
    rows = []
    for patcher in model_management.loaded_models():
        model = getattr(patcher, "model", None)
        rows.append(
            {
                "model": type(model).__name__ if model is not None else "None",
                "load_device": str(getattr(patcher, "load_device", "unknown")),
                "offload_device": str(getattr(patcher, "offload_device", "unknown")),
            }
        )
    return rows


def _snapshot(stage: str, phase: str) -> dict[str, Any]:
    process = psutil.Process()
    process_memory = process.memory_info()
    system_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "allocated": torch.cuda.memory_allocated(index),
                        "reserved": torch.cuda.memory_reserved(index),
                        "global_free": free_bytes,
                        "total": total_bytes,
                    }
                )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "phase": phase,
        "pid": os.getpid(),
        "process_rss": process_memory.rss,
        "process_private": getattr(process_memory, "private", None),
        "system_total": system_memory.total,
        "system_available": system_memory.available,
        "swap_used": swap_memory.used,
        "loaded_models": _loaded_models(),
        "gpus": devices,
    }


def _record(stage: str, phase: str) -> None:
    audit_dir = Path(folder_paths.base_path).parent / ".scratch" / "void_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "runtime_barriers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_snapshot(stage, phase), ensure_ascii=False) + "\n")


def _unload(model: Any, stage: str) -> None:
    if model is None or not hasattr(model, "clone_base_uuid"):
        raise TypeError(f"{stage}: connected object is not a ComfyUI ModelPatcher")
    _record(stage, "before")
    model_management.unload_model_and_clones(model, all_devices=True)
    gc.collect()
    model_management.soft_empty_cache(force=True)
    _record(stage, "after")


class VoidUnloadModelImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "stage": ("STRING", {"default": "after_pass_1"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "unload"
    CATEGORY = "VOID/Memory"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def unload(self, image, model, stage):
        _unload(model, stage)
        return (image,)


class VoidUnloadModelLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "stage": ("STRING", {"default": "after_pass_2"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "unload"
    CATEGORY = "VOID/Memory"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def unload(self, latent, model, stage):
        _unload(model, stage)
        return (latent,)


class VoidUnloadOpticalFlowLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "optical_flow": ("OPTICAL_FLOW",),
                "stage": ("STRING", {"default": "after_raft"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "unload"
    CATEGORY = "VOID/Memory"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def unload(self, latent, optical_flow, stage):
        _unload(optical_flow, stage)
        return (latent,)


class VoidSliceVideoByFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "start_frame_index": ("INT", {"default": 0, "min": 0}),
                "duration_seconds": ("INT", {"default": 5, "min": 1}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "slice"
    CATEGORY = "VOID/Video"

    def slice(self, video, start_frame_index, duration_seconds):
        frame_rate = float(video.get_frame_rate())
        if frame_rate <= 0:
            raise ValueError("VOID video slice: source frame rate must be positive")
        start_time = start_frame_index / frame_rate
        trimmed = video.as_trimmed(start_time, duration_seconds, strict_duration=False)
        if trimmed is None:
            raise ValueError(
                f"VOID video slice failed: start frame {start_frame_index}, "
                f"duration {duration_seconds}s, source rate {frame_rate:g} fps"
            )
        return (trimmed,)


class VoidRecomposeCroppedVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("VIDEO",),
                "edited_video": ("VIDEO",),
                "x": ("INT", {"default": 0, "min": 0}),
                "y": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "recompose"
    CATEGORY = "VOID/Video"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def recompose(self, source_video, edited_video, x, y):
        _record("final_recompose", "before_decode")
        edited = edited_video.get_components()
        source = source_video.get_components()
        destination = source.images
        patch = edited.images

        frame_count = min(destination.shape[0], patch.shape[0])
        x = min(max(int(x), 0), destination.shape[2])
        y = min(max(int(y), 0), destination.shape[1])
        width = min(patch.shape[2], destination.shape[2] - x)
        height = min(patch.shape[1], destination.shape[1] - y)
        channels = min(destination.shape[3], patch.shape[3])
        if frame_count <= 0 or width <= 0 or height <= 0 or channels <= 0:
            raise ValueError("VOID video recompose: edited crop does not overlap the source video")

        destination[:frame_count, y : y + height, x : x + width, :channels].copy_(
            patch[:frame_count, :height, :width, :channels]
        )
        result = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=destination,
                audio=source.audio,
                frame_rate=source.frame_rate,
            ),
            bit_depth=source_video.get_bit_depth(),
        )
        _record("final_recompose", "after_copy")
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VoidUnloadModelImage": VoidUnloadModelImage,
    "VoidUnloadModelLatent": VoidUnloadModelLatent,
    "VoidUnloadOpticalFlowLatent": VoidUnloadOpticalFlowLatent,
    "VoidSliceVideoByFrame": VoidSliceVideoByFrame,
    "VoidRecomposeCroppedVideo": VoidRecomposeCroppedVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidUnloadModelImage": "VOID Unload Model -> Image",
    "VoidUnloadModelLatent": "VOID Unload Model -> Latent",
    "VoidUnloadOpticalFlowLatent": "VOID Unload RAFT -> Latent",
    "VoidSliceVideoByFrame": "VOID Slice Video by Frame",
    "VoidRecomposeCroppedVideo": "VOID Recompose Cropped Video (In Place)",
}
