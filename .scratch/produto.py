import json, statistics
from pathlib import Path

ANALISES = {p.stem.replace(".analysis",""): p for p in Path("calib").glob("*.analysis.json")}

def carrega(an):
    a = json.loads(an.read_text(encoding="utf-8"))
    cam = a["layers"]
    it = cam.values() if isinstance(cam, dict) else cam
    return {c["layer"]: c for c in it}, a.get("convrot_groupsize")

def efetivo(sidecar, an):
    """erro EFETIVO: por camada, o erro do formato que ELA recebeu."""
    s = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    L = s.get("layers") or s.get("_quantization_metadata", {}).get("layers") or {}
    tab, cg_an = carrega(an)
    errs, faltam = [], 0
    for nome, v in L.items():
        fmt = v.get("format") if isinstance(v, dict) else str(v)
        cg = v.get("convrot_groupsize") if isinstance(v, dict) else None
        c = tab.get(nome) or tab.get(nome.replace(".weight",""))
        if not c: faltam += 1; continue
        if fmt == "convrot_w4a4":
            if cg is not None and cg_an is not None and cg != cg_an: faltam += 1; continue
            errs.append(c["err_w4a4"])
        elif fmt in ("asym_w4a8_int8","w4a8"):
            errs.append(c["err_w4a8"])
        else:
            errs.append(c.get("err_w4a4", 0.0))
    return (statistics.median(errs) if errs else float("nan")), len(L), faltam

CASOS = [
 ("Z-Image v2 cg256",  "ComfyUI/models/diffusion_models/zimage-v2-w4a4.quant.json",        "zimage_v2_native",     "funciona"),
 ("Z-Image v2 cg64",   "ComfyUI/models/diffusion_models/zimage-v2-teto-cg64.quant.json",   "zimage_v2_sigma_cg64", "funciona"),
 ("Z-Image v2 cg16",   "ComfyUI/models/diffusion_models/zimage-v2-teto-cg16.quant.json",   "zimage_v2_sigma_cg16", "DESTRUIDO"),
 ("Krea2 cg256",       "ComfyUI/models/diffusion_models/krea2_turbo_w4a4.quant.json",      "krea2_turbo",          "funciona"),
 ("Krea2 cg64",        "ComfyUI/models/diffusion_models/krea2_turbo_cg64.quant.json",      "krea2_turbo_cg64",     "funciona"),
 ("Krea2 cg16",        "ComfyUI/models/diffusion_models/krea2_turbo_cg16.quant.json",      "krea2_turbo_cg16",     "funciona"),
 ("Wan2.1 VACE",       None,                                                                "xfer_wan21_vace",      "DESTRUIDO"),
 ("Hunyuan15 w4a4",    "ComfyUI/models/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot.quant.json", "xfer_hunyuan15", "DESTRUIDO"),
 ("Hunyuan15 t015",    "ComfyUI/models/diffusion_models/hunyuan15-misto-t015.quant.json",  "xfer_hunyuan15",       "?"),
 ("Hunyuan15 t021",    "ComfyUI/models/diffusion_models/hunyuan15-misto-t021.quant.json",  "xfer_hunyuan15",       "?"),
 ("Hunyuan15 t025",    "ComfyUI/models/diffusion_models/hunyuan15-misto-t025.quant.json",  "xfer_hunyuan15",       "?"),
 ("Qwen Edit w4a4",    "ComfyUI/models/diffusion_models/qwen_image_edit_2511_w4a4.quant.json", "qwen_image_edit_2511", "DESTRUIDO"),
 ("Qwen Edit misto",   "ComfyUI/models/diffusion_models/qwen_image_edit_2511_mixed.quant.json","qwen_image_edit_2511", "DESTRUIDO"),
]

print(f"{'build':<20}{'camadas':>8}{'err_efetivo':>12}{'produto':>9}  veredito")
for nome, side, an_nome, ver in CASOS:
    an = ANALISES.get(an_nome)
    if an is None:
        print(f"{nome:<20} sem analise '{an_nome}'"); continue
    if side is None or not Path(side).exists():
        tab,_ = carrega(an)
        errs = [c["err_w4a4"] for c in tab.values()]
        med, n, faltam = statistics.median(errs), len(errs), 0
    else:
        med, n, faltam = efetivo(side, an)
    marca = f"  ({faltam} sem par)" if faltam else ""
    print(f"{nome:<20}{n:>8}{med:>12.4f}{med*n:>9.1f}  {ver}{marca}")
