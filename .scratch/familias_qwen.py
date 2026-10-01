import json, re, statistics
from pathlib import Path

an = json.loads(Path("calib/qwen_image_edit_2511.analysis.json").read_text(encoding="utf-8"))
cam = an["layers"]
it = cam.values() if isinstance(cam, dict) else cam

def familia(nome):
    # tira indices de bloco, deixa so o papel da camada
    n = re.sub(r"\.\d+\.", ".N.", nome)
    n = re.sub(r"^transformer_blocks\.N\.", "", n)
    return n.replace(".weight", "")

grupos = {}
for c in it:
    f = familia(c["layer"])
    grupos.setdefault(f, []).append(c)

print(f"{'familia':<34}{'n':>5}{'w4a4':>9}{'w4a8':>9}{'razao':>8}{'shape':>22}")
linhas = []
for f, cs in grupos.items():
    w4 = statistics.median(x["err_w4a4"] for x in cs)
    w8 = statistics.median(x["err_w4a8"] for x in cs)
    linhas.append((w4, f, len(cs), w4, w8, cs[0]["shape"]))
for w4s, f, n, w4, w8, shape in sorted(linhas, reverse=True):
    print(f"{f:<34}{n:>5}{w4:>9.4f}{w8:>9.4f}{w4/w8:>8.2f}x{str(shape):>22}")

# quais foram promovidas a 8 bits no build misto
side = json.loads(Path("ComfyUI/models/diffusion_models/qwen_image_edit_2511_mixed.quant.json").read_text(encoding="utf-8"))
lay = side.get("layers") or side.get("_quantization_metadata", {}).get("layers") or {}
prom = {}
for nome, v in lay.items():
    fmt = v.get("format") if isinstance(v, dict) else str(v)
    if fmt == "asym_w4a8_int8":
        prom[familia(nome)] = prom.get(familia(nome), 0) + 1
print("\npromovidas a 8 bits no build misto, por familia:")
for f, n in sorted(prom.items(), key=lambda kv: -kv[1]):
    tot = len(grupos.get(f, []))
    print(f"  {f:<34}{n:>4} de {tot:<4} ({100*n/tot:.0f}%)" if tot else f"  {f:<34}{n:>4}")
