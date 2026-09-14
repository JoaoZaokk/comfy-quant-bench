"""Distancia entre condicionamentos salvos (models/embeddings/<prefixo>_pos|_neg.safetensors).

Para cada braco e cada lado (pos/neg): rel-L2 = ||a - r|| / ||r||, cosseno dos vetores achatados,
max |a - r|, fracao de elementos bit-iguais em bf16, formas. Tudo em float32. Sem GPU, sem torch
CUDA. Nao mede: o render (isso e o compara_av); a mascara de atencao; nada alem do tensor
`conditioning_data_0` e das opcoes salvas (impressas, nao comparadas).

Uso:
    python -s tools/compara_cond.py --ref ltx23condf --arm ltx23cond_gw4a8 --arm ltx23cond_hbf16 [--json saida.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "ComfyUI" / "models" / "embeddings"


def carrega(prefixo: str, lado: str):
    import torch
    from safetensors import safe_open
    p = EMB / f"{prefixo}_{lado}.safetensors"
    if not p.exists():
        raise SystemExit(f"nao existe {p}")
    with safe_open(str(p), framework="pt") as f:
        t = f.get_tensor("conditioning_data_0")
        meta = dict(f.metadata() or {})
    return t.float(), meta.get("options_0"), meta.get("encoder")


def compara(r, a):
    import torch
    if tuple(r.shape) != tuple(a.shape):
        return {"forma_ref": list(r.shape), "forma_arm": list(a.shape), "forma_igual": False}
    d = a - r
    rel = float(d.norm() / r.norm().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0))
    bit = float((a.to(torch.bfloat16) == r.to(torch.bfloat16)).float().mean())
    return {"forma_ref": list(r.shape), "forma_arm": list(a.shape), "forma_igual": True,
            "rel_l2": rel, "cosseno": cos, "max_abs": float(d.abs().max()), "bit_iguais_bf16": bit,
            "ref_min": float(r.min()), "ref_max": float(r.max())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    out = {"ref": a.ref, "bracos": {}}
    for lado in ("pos", "neg"):
        r, opt_r, enc_r = carrega(a.ref, lado)
        print(f"ref {a.ref}_{lado}: {list(r.shape)} encoder={enc_r} options={opt_r}")
        for arm in a.arm:
            t, opt, enc = carrega(arm, lado)
            c = compara(r, t)
            out["bracos"].setdefault(arm, {})[lado] = c | {"encoder": enc, "options": opt}
            if c.get("forma_igual"):
                print(f"  {arm:22s} {lado}: rel-L2 {c['rel_l2']:.4g}  cos {c['cosseno']:.6f}  max|d| {c['max_abs']:.3g}  "
                      f"bit-iguais(bf16) {c['bit_iguais_bf16']*100:.1f}%  encoder={enc}")
            else:
                print(f"  {arm:22s} {lado}: FORMA DIFERE {c['forma_ref']} vs {c['forma_arm']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("json em", a.json)
    print("NAO COBERTO: so o tensor conditioning_data_0 (pos e neg); a mascara e as opcoes sao impressas, nao comparadas; "
          "distancia de condicionamento nao e distancia de render.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
