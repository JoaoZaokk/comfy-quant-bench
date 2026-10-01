"""Gera as fontes sinteticas da prova de equivalencia. Deterministico: mesma semente, mesmos bytes.

Escritor proprio (struct + json), chaves em ordem alfabetica, `__metadata__` primeiro. So CPU.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
FONTES = AQUI / "fontes"

ST = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int8: "I8",
      torch.uint8: "U8"}


def grava(path: Path, tensores: dict[str, torch.Tensor], meta: dict | None = None) -> None:
    header: dict = {}
    if meta:
        header["__metadata__"] = meta
    off = 0
    blobs = []
    for k in sorted(tensores):
        t = tensores[k].contiguous()
        raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
        header[k] = {"dtype": ST[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    hb += b" " * (-len(hb) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


class G:
    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    def w(self, *shape, dtype=torch.bfloat16, scale=0.05):
        return (torch.randn(*shape, generator=self.g) * scale).to(dtype)

    def norm(self, n, dtype=torch.bfloat16):
        return (torch.rand(n, generator=self.g) * 0.5 - 0.1).to(dtype)


def gemma() -> dict:
    g = G(1)
    t = {"model.embed_tokens.weight": g.w(64, 256), "model.norm.weight": g.norm(256)}
    for i in range(2):
        p = f"model.layers.{i}."
        t |= {p + "self_attn.q_proj.weight": g.w(256, 256), p + "self_attn.k_proj.weight": g.w(128, 256),
              p + "self_attn.v_proj.weight": g.w(128, 256), p + "self_attn.o_proj.weight": g.w(256, 256),
              p + "mlp.gate_proj.weight": g.w(512, 256), p + "mlp.up_proj.weight": g.w(512, 256),
              p + "mlp.down_proj.weight": g.w(256, 512),
              p + "input_layernorm.weight": g.norm(256),
              p + "post_attention_layernorm.weight": g.norm(256),
              p + "pre_feedforward_layernorm.weight": g.norm(256),
              p + "post_feedforward_layernorm.weight": g.norm(256),
              p + "self_attn.q_norm.weight": g.norm(64, torch.float32)}
    return t


def ltx() -> dict:
    g = G(2)
    t = {"adaln_single.linear.weight": g.w(512, 256), "proj_in.weight": g.w(256, 128),
         "proj_out.weight": g.w(128, 256), "scale_shift_table": g.w(2, 256, dtype=torch.float32),
         "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": g.w(256, 256),
         "video_embeddings_connector.transformer_1d_blocks.0.ff.net.2.weight": g.w(256, 512)}
    for i in range(2):
        p = f"transformer_blocks.{i}."
        for a in ("attn1", "attn2", "audio_to_video_attn"):
            for q in ("to_q", "to_k", "to_v", "to_out.0"):
                t[p + f"{a}.{q}.weight"] = g.w(256, 256)
            t[p + f"{a}.to_out.0.bias"] = g.w(256, dtype=torch.float32)
            t[p + f"{a}.norm_q.weight"] = g.norm(256)
        t[p + "ff.net.0.proj.weight"] = g.w(512, 256)
        t[p + "ff.net.2.weight"] = g.w(256, 512)
        t[p + "attn1.to_gate_logits.weight"] = g.w(32, 256)
    return t


def zimage() -> dict:
    g = G(3)
    t = {"x_embedder.weight": g.w(256, 64), "final_layer.linear.weight": g.w(64, 256),
         "cap_embedder.1.weight": g.w(256, 256)}
    for pilha, n in (("layers", 3), ("noise_refiner", 1)):
        for i in range(n):
            p = f"{pilha}.{i}."
            t |= {p + "attention.qkv.weight": g.w(768, 256), p + "attention.out.weight": g.w(256, 256),
                  p + "feed_forward.w1.weight": g.w(512, 256), p + "feed_forward.w2.weight": g.w(256, 512),
                  p + "feed_forward.w3.weight": g.w(512, 256),
                  p + "adaLN_modulation.0.weight": g.w(1024, 256),
                  p + "attention_norm1.weight": g.norm(256)}
    return t


def svdq() -> tuple[dict, dict]:
    g = G(4)
    t = {"layers.0.attention_norm1.weight": g.norm(64), "x_embedder.weight": g.w(64, 32)}

    def stem(s, inn, out, bias=False, rank=4):
        t[f"{s}.qweight"] = torch.randint(-128, 127, (out, inn // 2), generator=g.g, dtype=torch.int8)
        t[f"{s}.wscales"] = g.w(out)
        t[f"{s}.smooth_factor"] = g.norm(inn)
        t[f"{s}.smooth_factor_orig"] = g.norm(inn)
        t[f"{s}.proj_down"] = g.w(inn, rank)
        t[f"{s}.proj_up"] = g.w(out, rank)
        if bias:
            t[f"{s}.bias"] = g.w(out)
    stem("layers.0.attention.to_qkv", 64, 192)
    stem("layers.0.attention.to_out.0", 64, 64, bias=True)
    stem("layers.0.feed_forward.net.0.proj", 64, 256)
    stem("layers.0.feed_forward.net.2", 128, 64)
    meta = {"quantization_config": json.dumps({"weight": {"dtype": "int4", "group_size": 64}, "rank": 4}),
            "config": json.dumps({"n_heads": 2, "n_kv_heads": 2}), "model_class": "sint"}
    return t, meta


def svdq_ref() -> dict:
    g = G(5)
    return {"layers.0.attention.to_q.weight": g.w(64, 64), "layers.0.attention.to_k.weight": g.w(64, 64),
            "layers.0.attention.to_v.weight": g.w(64, 64), "layers.0.attention.to_out.0.weight": g.w(64, 64),
            "layers.0.feed_forward.w1.weight": g.w(128, 64), "layers.0.feed_forward.w3.weight": g.w(128, 64),
            "layers.0.feed_forward.w2.weight": g.w(64, 128)}


def checkpoint() -> dict:
    g = G(6)
    t = {"vae.decoder.conv.weight": g.w(8, 8), "text_encoders.t5.weight": g.w(16, 8)}
    for k in ("x_embedder.weight", "layers.0.attention.qkv.weight", "layers.0.norm.weight"):
        t["model.diffusion_model." + k] = g.w(32, 16)
    return t


def klein_par() -> tuple[dict, dict]:
    ga, gb = G(7), G(8)
    base, doador = {}, {}
    for i in range(2):
        for k, shp in ((f"double_blocks.{i}.txt_attn.qkv.weight", (192, 128)),
                       (f"double_blocks.{i}.img_attn.qkv.weight", (192, 128)),
                       (f"double_blocks.{i}.txt_mlp.0.weight", (256, 128)),
                       (f"double_blocks.{i}.norm.weight", (128,))):
            base[k] = ga.w(*shp)
            doador[k] = gb.w(*shp)
    return base, doador


def ternario(g: G, n: int, k: int) -> torch.Tensor:
    cod = torch.randint(-1, 2, (n, k // 128, 128), generator=g.g).float()
    esc = (torch.rand(n, k // 128, 1, generator=g.g) * 0.1 + 0.01).to(torch.bfloat16).float()
    return (cod * esc).reshape(n, k).to(torch.bfloat16)


def transplante() -> tuple[dict, dict]:
    out = []
    for seed in (9, 10):
        g = G(seed)
        t = {}
        for i in range(2):
            t[f"double_blocks.{i}.img_attn.qkv.weight"] = ternario(g, 64, 256)
            t[f"single_blocks.{i}.linear1.weight"] = ternario(g, 32, 128)
            t[f"double_blocks.{i}.img_attn.norm.query_norm.scale"] = g.norm(64)
        t["img_in.weight"] = g.w(64, 64)
        t["final_layer.linear.weight"] = g.w(32, 64)
        out.append(t)
    return out[0], out[1]


def calib_grava(zt: dict) -> dict:
    g = G(11)
    layers = {}
    for stem in ("layers.0.attention.out", "layers.1.feed_forward.w2", "layers.0.feed_forward.w1"):
        k = zt[stem + ".weight"].shape[1]
        layers[stem] = {"sample": g.w(96, k, scale=1.0)}
    layers["layers.2.attention.out"] = {"sample": None}
    return {"layers": layers}


def ja_quantizado() -> tuple[dict, dict]:
    g = G(12)
    t = {"a.weight": torch.randint(-127, 127, (64, 256), generator=g.g, dtype=torch.int8),
         "a.weight_scale": g.w(64, 1, dtype=torch.float32),
         "b.weight": torch.randint(-127, 127, (64, 64), generator=g.g, dtype=torch.int8),
         "b.weight_scale": g.w(64, dtype=torch.float32),
         "c.weight": g.w(64, 64)}
    layers = {"a": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
              "b": {"format": "convrot_w4a4", "convrot_groupsize": 64}}
    meta = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers},
                                                 separators=(",", ":")),
            "quantization": "sint"}
    return t, meta


def main() -> int:
    FONTES.mkdir(parents=True, exist_ok=True)
    meta_pt = {"format": "pt", "nota": "fonte sintética"}
    grava(FONTES / "gemma_sint_bf16.safetensors", gemma(), meta_pt)
    grava(FONTES / "ltx_sint_bf16.safetensors", ltx(), meta_pt)
    zt = zimage()
    grava(FONTES / "zimage_sint_native_bf16.safetensors", zt, meta_pt)
    st, sm = svdq()
    grava(FONTES / "svdq_sint.safetensors", st, sm)
    grava(FONTES / "svdq_ref_bf16.safetensors", svdq_ref())
    grava(FONTES / "checkpoint_sint.safetensors", checkpoint(), {"modelspec.title": "sint"})
    b, d = klein_par()
    grava(FONTES / "klein_base.safetensors", b, {"format": "pt"})
    grava(FONTES / "klein_doador.safetensors", d, {"format": "pt"})
    a, bb = transplante()
    grava(FONTES / "klein_tern_A.safetensors", a)
    grava(FONTES / "klein_tern_B.safetensors", bb, {"format": "pt"})
    torch.save(calib_grava(zt), FONTES / "calib_grava.pt")
    q, qm = ja_quantizado()
    grava(FONTES / "ja_quantizado.safetensors", q, qm)
    linhas = []
    for p in sorted(FONTES.iterdir()):
        if p.suffix in (".safetensors",):
            linhas.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (FONTES / "SHA256.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
