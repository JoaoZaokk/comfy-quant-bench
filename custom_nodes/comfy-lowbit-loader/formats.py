"""Readers that turn every supported file into one ComfyUI state dict in the `lowbit_affine` format.

Recognized inputs, detected from the file contents, never from the file name:

    gemlite     `state_dict.pt` with `<layer>.W_q/.scales/.zeros/.metadata/.orig_shape` (Bonsai int2/int1)
    mlx         safetensors with `<layer>.weight` (uint32) + `.scales` + `.biases` (mflux / mx.quantize)
    lowbit      safetensors already saved in this format (`comfy_quant` = lowbit_affine)
    dense       any other checkpoint: 2-D weights whose groups hold only 2 or 3-4 evenly spaced values
                are packed losslessly (the Bonsai "unpacked" files, or any BF16 ternary checkpoint);
                everything else stays as it is

All three Bonsai packs describe the same numbers: w = code * scale + zero, codes LSB-first along K.
Measured on the 1-bit and 2-bit packs: gemlite and MLX reproduce the unpacked BF16 bit for bit.

Diffusers-named FLUX.2 checkpoints (the Bonsai layout) are renamed to the BFL names ComfyUI loads,
with the rule table derived by hashing the two official klein-4B releases against each other.
"""

import json
import re
import struct
from dataclasses import dataclass

import torch

import comfy.utils

from . import kernel
from .layout import FORMAT, shape_params


@dataclass
class LowBit:
    qdata: torch.Tensor  # uint8 (N, K * bits / 8)
    scale: torch.Tensor  # (N, K / G)
    zero: torch.Tensor   # (N, K / G)
    bits: int
    group_size: int


def _conf_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def to_comfy_state_dict(dense, lowbit):
    sd = dict(dense)
    for name, layer in lowbit.items():
        sd[f"{name}.weight"] = layer.qdata
        sd[f"{name}.weight_scale"] = layer.scale
        sd[f"{name}.weight_zeros"] = layer.zero
        sd[f"{name}.comfy_quant"] = _conf_tensor({"format": FORMAT})
    return sd


# ---------------------------------------------------------------- detection

def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def detect(path):
    if path.lower().endswith((".pt", ".pth", ".bin", ".ckpt")):
        return "gemlite"
    header = safetensors_header(path)
    keys = header.keys()
    if any(k.endswith(".comfy_quant") for k in keys) or FORMAT in json.dumps(header.get("__metadata__", {})):
        return "lowbit"
    if any(k.endswith(".biases") for k in keys) and any(k.endswith(".scales") for k in keys):
        return "mlx"
    return "dense"


# ---------------------------------------------------------------- readers

def read_gemlite(path):
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    dense, lowbit = {}, {}
    for key, value in sd.items():
        name, _, suffix = key.rpartition(".")
        if suffix == "W_q":
            meta = sd[f"{name}.metadata"].tolist()
            n, k = sd[f"{name}.orig_shape"].tolist()
            qdata = value.t().contiguous()
            scale = sd[f"{name}.scales"].t().contiguous()
            zero = sd[f"{name}.zeros"].t().contiguous()
            bits, group_size = shape_params(qdata, scale, k)
            if qdata.shape[0] != n or (meta[1], meta[2]) != (bits, group_size):
                raise ValueError(f"gemlite layer {name}: metadata {meta[:3]} disagrees with shapes (bits={bits}, G={group_size}, N={n})")
            lowbit[name] = LowBit(qdata, scale, zero, bits, group_size)
        elif suffix not in ("scales", "zeros", "metadata", "orig_shape"):
            dense[key] = value
    return dense, lowbit


def read_mlx(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    hidden = _hidden_size(sd)
    packed = [k[: -len(".biases")] for k in sd if k.endswith(".biases")]
    # The packed width fixes bits * K and the table fixes K / G; the attention q projection, whose input
    # is the hidden size, resolves the one remaining unknown for the whole file.
    probe = next(n for n in packed if re.search(r"(attn\.to_q|img_attn\.qkv|attn\.to_qkv)", n))
    bits, group_size = shape_params(sd[f"{probe}.weight"].view(torch.uint8), sd[f"{probe}.scales"], hidden)
    dense, lowbit = {}, {}
    for name in packed:
        qdata = sd.pop(f"{name}.weight").view(torch.uint8)
        scale, zero = sd.pop(f"{name}.scales"), sd.pop(f"{name}.biases")
        k = scale.shape[1] * group_size
        if qdata.shape[1] * 8 != bits * k:
            raise ValueError(f"MLX layer {name}: width {qdata.shape[1]} does not fit {bits}-bit codes over K={k}")
        lowbit[name] = LowBit(qdata, scale, zero, bits, group_size)
    dense.update(sd)
    return dense, lowbit


def read_lowbit(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    return sd, {}


def _hidden_size(sd):
    for key in ("x_embedder.weight", "img_in.weight"):
        if key in sd:
            return sd[key].shape[0]
    raise ValueError("could not find the input embedding (x_embedder / img_in) to size the packed layers")


# ---------------------------------------------------------------- lossless packing of dense low-bit weights

def pack_dense_layer(w, group_size=128, device="cpu"):
    """Pack a 2-D weight whose every group of `group_size` along K is exactly representable as
    code * scale + zero with 1-bit or 2-bit codes. Returns None when any element would change."""
    if w.ndim != 2 or w.shape[1] % group_size or not w.is_floating_point():
        return None
    n, k = w.shape
    x = w.to(device=device, dtype=torch.float32).view(n, k // group_size, group_size)
    lo, hi = x.amin(-1), x.amax(-1)
    for bits, levels in ((1, 1), (2, 2), (2, 3)):
        step = (hi - lo) / levels
        for store in (w.dtype, torch.float32):
            scale, zero = step.to(store), lo.to(store)
            safe = torch.where(scale == 0, torch.ones_like(scale), scale).to(torch.float32)
            codes = ((x - zero.to(torch.float32).unsqueeze(-1)) / safe.unsqueeze(-1)).round().clamp(0, (1 << bits) - 1)
            back = (codes * scale.to(torch.float32).unsqueeze(-1) + zero.to(torch.float32).unsqueeze(-1)).to(w.dtype)
            if torch.equal(back, x.to(w.dtype)):
                qdata = kernel.pack_codes(codes.view(n, k).to(torch.uint8), bits).cpu()
                return LowBit(qdata, scale.cpu(), zero.cpu(), bits, group_size)
    return None


def read_dense(path, device="cpu"):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    dense, lowbit = {}, {}
    for key, value in sd.items():
        layer = pack_dense_layer(value, device=device) if key.endswith(".weight") and value.ndim == 2 and min(value.shape) >= 256 else None
        if layer is None:
            dense[key] = value
        else:
            lowbit[key[: -len(".weight")]] = layer
    return dense, lowbit


READERS = {"gemlite": read_gemlite, "mlx": read_mlx, "lowbit": read_lowbit, "dense": read_dense}


# ---------------------------------------------------------------- diffusers FLUX.2 -> BFL names

_RENAMES = [(re.compile(f"^{src}$"), dst) for src, dst in [
    (r"transformer_blocks\.(\d+)\.attn\.norm_k\.weight", r"double_blocks.\1.img_attn.norm.key_norm.scale"),
    (r"transformer_blocks\.(\d+)\.attn\.norm_q\.weight", r"double_blocks.\1.img_attn.norm.query_norm.scale"),
    (r"transformer_blocks\.(\d+)\.attn\.norm_added_k\.weight", r"double_blocks.\1.txt_attn.norm.key_norm.scale"),
    (r"transformer_blocks\.(\d+)\.attn\.norm_added_q\.weight", r"double_blocks.\1.txt_attn.norm.query_norm.scale"),
    (r"transformer_blocks\.(\d+)\.attn\.to_out\.0", r"double_blocks.\1.img_attn.proj"),
    (r"transformer_blocks\.(\d+)\.attn\.to_add_out", r"double_blocks.\1.txt_attn.proj"),
    (r"transformer_blocks\.(\d+)\.ff\.linear_in", r"double_blocks.\1.img_mlp.0"),
    (r"transformer_blocks\.(\d+)\.ff\.linear_out", r"double_blocks.\1.img_mlp.2"),
    (r"transformer_blocks\.(\d+)\.ff_context\.linear_in", r"double_blocks.\1.txt_mlp.0"),
    (r"transformer_blocks\.(\d+)\.ff_context\.linear_out", r"double_blocks.\1.txt_mlp.2"),
    (r"single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj", r"single_blocks.\1.linear1"),
    (r"single_transformer_blocks\.(\d+)\.attn\.to_out", r"single_blocks.\1.linear2"),
    (r"single_transformer_blocks\.(\d+)\.attn\.norm_k\.weight", r"single_blocks.\1.norm.key_norm.scale"),
    (r"single_transformer_blocks\.(\d+)\.attn\.norm_q\.weight", r"single_blocks.\1.norm.query_norm.scale"),
    (r"double_stream_modulation_img\.linear", "double_stream_modulation_img.lin"),
    (r"double_stream_modulation_txt\.linear", "double_stream_modulation_txt.lin"),
    (r"single_stream_modulation\.linear", "single_stream_modulation.lin"),
    (r"norm_out\.linear", "final_layer.adaLN_modulation.1"),
    (r"proj_out", "final_layer.linear"),
    (r"x_embedder", "img_in"),
    (r"context_embedder", "txt_in"),
    (r"time_guidance_embed\.timestep_embedder\.linear_1", "time_in.in_layer"),
    (r"time_guidance_embed\.timestep_embedder\.linear_2", "time_in.out_layer"),
]]
# q, k, v are separate in diffusers and one fused projection (concatenated on the output axis) in BFL.
_FUSED = [
    (re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_([qkv])$"), r"double_blocks.\1.img_attn.qkv"),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn\.add_([qkv])_proj$"), r"double_blocks.\1.txt_attn.qkv"),
]


def is_diffusers_flux2(keys):
    return any(k.startswith("double_stream_modulation_img.linear") for k in keys)


def _rename(name):
    for pattern, dst in _RENAMES:
        if pattern.match(name):
            return pattern.sub(dst, name)
    raise KeyError(f"no BFL name for diffusers tensor {name}")


def diffusers_flux2_to_bfl(dense, lowbit):
    """Rename modules/tensors; fuse q/k/v (packed rows concatenate like dense rows); swap the two halves
    of the final modulation, which diffusers stores as [shift, scale] and BFL as [scale, shift]."""
    fused, out_dense, out_lowbit = {}, {}, {}
    for name, layer in lowbit.items():
        for pattern, dst in _FUSED:
            m = pattern.match(name)
            if m:
                fused.setdefault(pattern.sub(dst, name), {})[m.group(2)] = layer
                break
        else:
            out_lowbit[_rename(name)] = layer
    for key, value in dense.items():
        module, _, suffix = key.rpartition(".")
        hit = next(((p, dst) for p, dst in _FUSED if p.match(module)), None)
        if hit:
            fused.setdefault(hit[0].sub(hit[1], module), {})[hit[0].match(module).group(2)] = value
            continue
        new = _rename(key) if key.endswith(("norm_k.weight", "norm_q.weight", "norm_added_k.weight", "norm_added_q.weight")) else f"{_rename(module)}.{suffix}"
        if module == "norm_out.linear":
            value = torch.cat(value.chunk(2, dim=0)[::-1], dim=0)
        out_dense[new] = value
    for dst, parts in fused.items():
        q, k, v = parts["q"], parts["k"], parts["v"]
        if isinstance(q, LowBit):
            if len({(p.bits, p.group_size) for p in (q, k, v)}) != 1:
                raise ValueError(f"{dst}: q/k/v packed with different bits or group size")
            out_lowbit[dst] = LowBit(*(torch.cat([getattr(p, f) for p in (q, k, v)], 0) for f in ("qdata", "scale", "zero")), q.bits, q.group_size)
        else:
            out_dense[f"{dst}.weight"] = torch.cat([q, k, v], 0)
    return out_dense, out_lowbit


def load(path, pack_device="cpu"):
    """-> (kind, ComfyUI state dict, report dict)."""
    kind = detect(path)
    dense, lowbit = READERS[kind](path, pack_device) if kind == "dense" else READERS[kind](path)
    renamed = is_diffusers_flux2(list(dense) + list(lowbit))
    if renamed:
        dense, lowbit = diffusers_flux2_to_bfl(dense, lowbit)
    bits = {}
    for layer in lowbit.values():
        bits[layer.bits] = bits.get(layer.bits, 0) + 1
    packed_bytes = sum(t.numel() * t.element_size() for l in lowbit.values() for t in (l.qdata, l.scale, l.zero))
    dense_bytes = sum(t.numel() * t.element_size() for t in dense.values())
    report = {
        "source": kind,
        "diffusers_renamed": renamed,
        "packed_layers_by_bits": bits,
        "group_sizes": sorted({l.group_size for l in lowbit.values()}),
        "packed_gib": round(packed_bytes / 2**30, 3),
        "dense_gib": round(dense_bytes / 2**30, 3),
    }
    return kind, to_comfy_state_dict(dense, lowbit), report
