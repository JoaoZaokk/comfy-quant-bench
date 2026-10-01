import json, statistics
from pathlib import Path

an = json.loads(Path("calib/qwen_image_edit_2511.analysis.json").read_text(encoding="utf-8"))
camadas = an["layers"] if isinstance(an, dict) and "layers" in an else an
print("chaves do topo:", list(an.keys())[:12] if isinstance(an, dict) else type(an))
print("camadas analisadas:", len(camadas))

primeira = next(iter(camadas.values())) if isinstance(camadas, dict) else camadas[0]
print("campos por camada:", list(primeira.keys()))

def col(nome):
    vals = []
    it = camadas.values() if isinstance(camadas, dict) else camadas
    for c in it:
        v = c.get(nome)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals

for m in ("err_bf16", "err_w4a4", "err_w4a8"):
    v = col(m)
    if v:
        v.sort()
        print(f"{m:10s} n={len(v):4d}  mediana={statistics.median(v):.4f}  "
              f"p25={v[len(v)//4]:.4f}  p75={v[3*len(v)//4]:.4f}  max={max(v):.4f}")

for side in ("qwen_image_edit_2511_w4a4", "qwen_image_edit_2511_mixed"):
    p = Path(f"ComfyUI/models/diffusion_models/{side}.quant.json")
    s = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n--- {side}.quant.json ---")
    for k in ("backend", "backend_linear", "convrot_groupsize", "source_identity_sha256",
              "parameters_total", "parameters_quantized", "comfy_kitchen", "torch"):
        if k in s:
            print(f"  {k}: {s[k]}")
    lay = s.get("layers") or s.get("_quantization_metadata", {}).get("layers") or {}
    fmts = {}
    for v in lay.values():
        f = v.get("format") if isinstance(v, dict) else str(v)
        fmts[f] = fmts.get(f, 0) + 1
    print("  formatos:", fmts, " total camadas:", len(lay))
