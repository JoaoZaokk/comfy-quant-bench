"""GPTQ for the convrot_w4a4 layers of an already converted checkpoint: only the int4 codes change.

Why this exists (fase 3, item 6, `.scratch/quantfunc_2026-10-01/otimizacao/resultado_fase3.md`): on a
held-out set of prompts/seeds, GPTQ with the per-row scale held fixed cut the W4A4 layer error of
Qwen-Image-2.1 by 12.8 % on average (0.1207 -> 0.1052, 20/20 layers better) once H came from 32768
rows; with the 128 rows of fase 2 the gain was 5.9 % and was written off. The scale per row is the
one already stored in --base, so the file keeps the exact format, loader path and kernel of the
original: runtime cost zero.

Per convrot_w4a4 layer of --base that has a Hessian:
  1. W (fp32) from --source, rotated with the comfy-kitchen ConvRot rotation the converter uses;
  2. RTN with the stored scale must reproduce the stored codes (proves --source and the rotation are
     the ones --base was made from; refuses otherwise);
  3. H_r = R^T H R (R = block-diagonal Hadamard), GPTQ column by column with the stored scale;
  4. codes packed exactly like the converter (low nibble = even column).
Every other tensor is copied verbatim from --base. Hessians come from
`tools/calibrate_activations.py --hessian` (X^T X of every row seen); a calibration that only has
`sample` rows is accepted too (H = sample^T sample), for small tests.

Qwen-Image-2.1 feeds to_q, to_k and to_v the same tensor (`comfy/ldm/qwen_image21/model.py`,
Attention.forward), so a Hessian captured on to_q serves all three (--share, on by default).

    python_embeded\\python.exe -s tools/gptq_w4a4.py --source P:/.../qwen_image_2.1_bf16.safetensors \\
        --base P:/.../qwen_image_2.1_bf16_w4a4_convrot.safetensors \\
        --hessian P:/ComfyBench/calib_fase3/h_attn_mlpin.calib.pt P:/.../h_mlpout_0.calib.pt ... \\
        --output P:/.../qwen_image_2.1_bf16_w4a4_convrot_gptq.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import _conversion as C  # noqa: E402

SHARE = ((re.compile(r"\.attn\.to_[kv]$"), ".attn.to_q"),)


def hadamard(size: int, device) -> torch.Tensor:
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard
    return _build_hadamard(size, device=device, dtype=torch.float32)


def rotate_weight(w: torch.Tensor, gs: int) -> torch.Tensor:
    from comfy_kitchen.backends.eager.convrot_w4a4 import _rotate_weight
    return _rotate_weight(w, hadamard(gs, w.device), gs)


def rotate_hessian(h: torch.Tensor, gs: int) -> torch.Tensor:
    """x_r = x @ Hd per group of gs (eager `_rotate_activation`) -> H_r = R^T H R, R block-diagonal."""
    k = h.shape[0]
    hd = hadamard(gs, h.device)
    g = k // gs
    t = h.reshape(g, gs, g, gs)
    t = torch.einsum("ji,ajbl->aibl", hd, t)          # R^T on the left (per row group)
    t = torch.einsum("aibl,lm->aibm", t, hd)          # R on the right (per column group)
    return t.reshape(k, k)


def unpack(q: torch.Tensor) -> torch.Tensor:
    from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
    return _unpack_int4_row_major(q).to(torch.int8)


def pack(codes: torch.Tensor) -> torch.Tensor:
    from comfy_kitchen.backends.eager.svdquant import _pack_int4_row_major
    return _pack_int4_row_major(codes)


def gptq_codes(w: torch.Tensor, h: torch.Tensor, scale: torch.Tensor, *, block: int, damp: float) -> torch.Tensor:
    """GPTQ (Frantar et al.) with a FIXED per-row scale. w (N,K) fp32 rotated, h (K,K) rotated, scale (N,).
    Returns int8 codes in [-7, 7]."""
    n, k = w.shape
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w = w.clone()
    w[:, dead] = 0
    h += torch.eye(k, device=w.device) * damp * torch.diag(h).mean()
    hinv = torch.linalg.cholesky(h)
    hinv = torch.cholesky_inverse(hinv)
    hinv = torch.linalg.cholesky(hinv, upper=True)
    s = scale.reshape(-1)
    codes = torch.empty(n, k, dtype=torch.int8, device=w.device)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            c = (col / s).round().clamp(-7, 7)
            codes[:, i1 + i] = c.to(torch.int8)
            e = (col - c * s) / hinv1[i, i]
            w1[:, i:] -= e.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes


def rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((got - ref).norm() / ref.norm())


def q4_token(x: torch.Tensor) -> torch.Tensor:
    s = (x.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-10)
    return (x / s).round().clamp(-7, 7) * s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="high-precision checkpoint --base was converted from")
    ap.add_argument("--base", type=Path, required=True, help="converted checkpoint with convrot_w4a4 layers")
    ap.add_argument("--hessian", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", default=None, help="regex: restrict GPTQ to matching layers (others copied)")
    ap.add_argument("--no-share", action="store_true", help="do not reuse to_q's Hessian for to_k/to_v")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--holdout", type=Path, default=None,
                    help="calibration with `sample` rows NOT used for H: reports RTN vs GPTQ W4A4 error per layer")
    ap.add_argument("--min-rtn-match", type=float, default=0.999,
                    help="fraction of RTN codes that must equal --base's (refuses below)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    args = ap.parse_args()
    dev = torch.device(args.device)

    sidecar = args.output.with_suffix(".quant.json")
    conv = C.Conversion(args.base, args.output, sidecar)
    conv.refuse_unsafe(allow_quantized_source=True)
    base_header, base_meta = conv.header, conv.metadata
    qmeta = json.loads(base_meta.get("_quantization_metadata", "{}"))
    layers = {k: v for k, v in qmeta.get("layers", {}).items() if v.get("format") == "convrot_w4a4"}
    if not layers:
        raise SystemExit(f"{args.base} has no convrot_w4a4 layer")
    only = re.compile(args.only) if args.only else None
    src_header, _ = C.read_header(args.source)
    src_start = C.data_start(args.source)
    base_start = conv.data_start

    def source_hessian_name(layer: str, available) -> str | None:
        if layer in available:
            return layer
        if not args.no_share:
            for pat, repl in SHARE:
                alt = pat.sub(repl, layer)
                if alt != layer and alt in available:
                    return alt
        return None

    holdout = None
    if args.holdout:
        holdout = torch.load(args.holdout, map_location="cpu", weights_only=False)["layers"]

    results: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}
    started = time.time()
    with open(args.source, "rb") as fsrc, open(args.base, "rb") as fbase:
        for hpath in args.hessian:
            calib = torch.load(hpath, map_location="cpu", weights_only=False)
            avail = calib["layers"]
            todo = [ly for ly in sorted(layers, key=lambda n: (int(n.split(".")[1]) if n.split(".")[1].isdigit() else -1, n))
                    if ly not in results and (only is None or only.search(ly)) and source_hessian_name(ly, avail)]
            print(f"{hpath.name}: {len(avail)} Hessians, {len(todo)} layers to do", flush=True)
            hcache: dict[str, torch.Tensor] = {}
            for ly in todo:
                hname = source_hessian_name(ly, avail)
                gs = int(layers[ly].get("convrot_groupsize", 256))
                if hname not in hcache:
                    e = avail[hname]
                    if "hessian" in e:
                        h, rows = e["hessian"].to(dev, torch.float32), int(e["hessian_rows"])
                    else:
                        x = e["sample"].to(dev, torch.float32)
                        h, rows = x.t() @ x, int(x.shape[0])
                    hcache = {hname: (rotate_hessian(h, gs), rows)}     # one at a time: K=12288 is 576 MiB
                    del h
                hr, rows = hcache[hname]
                info = src_header[ly + ".weight"]
                a, b = info["data_offsets"]
                w = C.read_tensor(fsrc, src_start + a, b - a, info["dtype"], info["shape"]).to(dev, torch.float32)
                wr = rotate_weight(w, gs)
                qi, si = base_header[ly + ".weight"], base_header[ly + ".weight_scale"]
                qa, qb = qi["data_offsets"]
                sa, sb = si["data_offsets"]
                q_base = C.read_tensor(fbase, base_start + qa, qb - qa, qi["dtype"], qi["shape"]).to(dev)
                scale = C.read_tensor(fbase, base_start + sa, sb - sa, si["dtype"], si["shape"]).to(dev, torch.float32).reshape(-1)
                codes_base = unpack(q_base)
                codes_rtn = (wr / scale[:, None]).round().clamp(-7, 7).to(torch.int8)
                match = float((codes_rtn == codes_base).float().mean())
                scale_chk = float(((wr.abs().amax(dim=1).clamp(min=1e-10) / 7) - scale).abs().max() / scale.abs().max())
                if match < args.min_rtn_match:
                    raise SystemExit(f"{ly}: RTN from --source reproduces only {match:.5f} of --base's codes "
                                     f"(scale check {scale_chk:.2e}); wrong --source or rotation, refusing")
                codes = gptq_codes(wr, hr, scale, block=args.block, damp=args.damp)
                rep = {"hessian": hname, "hessian_rows": rows, "rtn_match": round(match, 6),
                       "scale_check": scale_chk, "changed_codes": round(float((codes != codes_base).float().mean()), 4)}
                # proxy error on the Hessian's own rows: tr(D H D^T) / tr(W H W^T), D = W_q - W
                for tag, c in (("rtn", codes_base), ("gptq", codes)):
                    d = c.float() * scale[:, None] - wr
                    rep[f"proxy_{tag}"] = float(((d @ hr) * d).sum().clamp(min=0).sqrt() / ((wr @ hr) * wr).sum().sqrt())
                if holdout is not None and ly in holdout:
                    from comfy_kitchen.backends.eager.convrot_w4a4 import _rotate_activation
                    xh = _rotate_activation(holdout[ly]["sample"].to(dev, torch.float32), hadamard(gs, dev), gs)
                    exact = xh @ wr.t()
                    xq = q4_token(xh)
                    rep["holdout_rtn"] = rel(exact, xq @ (codes_base.float() * scale[:, None]).t())
                    rep["holdout_gptq"] = rel(exact, xq @ (codes.float() * scale[:, None]).t())
                    del xh, exact, xq
                results[ly] = pack(codes).cpu().contiguous()
                report[ly] = rep
                print(f"  {ly:42s} H={hname.split('.')[-1]} rows={rows} rtn_match={match:.5f} changed={rep['changed_codes']:.3f} "
                      f"proxy {rep['proxy_rtn']:.4f}->{rep['proxy_gptq']:.4f}"
                      + (f" holdout {rep['holdout_rtn']:.4f}->{rep['holdout_gptq']:.4f}" if "holdout_gptq" in rep else "")
                      + f" [{time.time() - started:.0f}s]", flush=True)
                del w, wr, q_base, codes_base, codes_rtn, codes
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            del calib, hcache
    skipped = sorted(set(layers) - set(results))
    print(f"\nGPTQ in {len(results)}/{len(layers)} convrot_w4a4 layers; kept RTN: {len(skipped)}", flush=True)
    if report:
        for key in ("proxy_rtn", "proxy_gptq", "holdout_rtn", "holdout_gptq"):
            vals = [r[key] for r in report.values() if key in r]
            if vals:
                print(f"  mean {key:13s} {sum(vals) / len(vals):.4f}  (n={len(vals)})")
    if args.dry_run or not results:
        print("dry run: nothing written" if args.dry_run else "no layer had a Hessian: nothing written")
        return 0

    entries = []
    for key, info in base_header.items():
        name = key.removesuffix(".weight")
        if key.endswith(".weight") and name in results:
            t = results[name]
            if list(t.shape) != list(info["shape"]) or info["dtype"] != "I8":
                raise SystemExit(f"{key}: packed {list(t.shape)} vs base {info['dtype']} {info['shape']}")
            entries.append(C.plan_write(key, t))
        else:
            entries.append(C.plan_copy(key, info))
    conv.guard(conv.planned_size(entries, base_meta))

    def manifesto() -> dict:
        base_side = args.base.with_suffix(".quant.json")
        prev = json.loads(base_side.read_text(encoding="utf-8")) if base_side.is_file() else {}
        return {**prev, "output": str(args.output), "output_size": conv.output_size,
                "gptq": {"base": str(args.base), "base_size": args.base.stat().st_size, "source": str(args.source),
                         "hessians": [str(p) for p in args.hessian], "damp": args.damp, "block": args.block,
                         "share_qkv": not args.no_share, "layers_gptq": len(results), "layers_rtn": skipped,
                         "per_layer": report, "seconds": round(time.time() - started, 1)}}

    conv.commit(entries, base_meta, sidecar=manifesto)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB) and {sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
