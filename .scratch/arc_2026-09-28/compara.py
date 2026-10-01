"""MS-SSIM das 12 imagens da Arc contra as referências da 3090 (mesmo prompt/seed).

    python_embeded\\python.exe -s compara.py <pasta_arc> <saida.json>
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
from metricas_imagem import carrega  # noqa: E402
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim  # noqa: E402

REF = Path("F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria")
REFS = ("zen_v12", "w4a16_q4_1", "bf16")
arc = Path(sys.argv[1])
out = {}
for ref in REFS:
    v = {}
    for i in range(6):
        for s in (42, 7):
            a = sorted(arc.glob(f"p{i}_s{s}_*.png"))
            r = REF / ref / f"p{i}_s{s}_00001_.png"
            if a and r.exists():
                v[f"p{i}_s{s}"] = round(float(msssim(carrega(a[-1]), carrega(r), data_range=1.0)), 4)
    out[ref] = {"media": round(st.mean(v.values()), 4), "min": min(v.values()), "n": len(v), "por_imagem": v}
    print(ref, out[ref]["media"], out[ref]["min"], out[ref]["n"])
Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
