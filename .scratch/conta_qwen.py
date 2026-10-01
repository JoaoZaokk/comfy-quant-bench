import json, struct
from pathlib import Path

def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n

DT = {"F32":4,"F16":2,"BF16":2,"I8":1,"U8":1,"I32":4,"I64":8,"F64":8,"BOOL":1,"F8_E4M3":1,"F8_E5M2":1}

for nome in ["qwen_image_edit_2511_bf16","qwen_image_edit_2511_int8_convrot",
             "qwen_image_edit_2511_w4a4","qwen_image_edit_2511_mixed"]:
    p = Path("ComfyUI/models/diffusion_models")/f"{nome}.safetensors"
    h, _ = header(p)
    tens = {k:v for k,v in h.items() if k != "__metadata__"}
    params = 0
    dt = {}
    for k,v in tens.items():
        n = 1
        for d in v["shape"]: n *= d
        params += n
        dt[v["dtype"]] = dt.get(v["dtype"],0)+1
    b = p.stat().st_size
    print(f"{nome:38s} {b:>14,} B  {b/2**30:7.2f} GiB  tensores={len(tens):5d}  params={params:>15,}")
    print(f"{'':38s} dtypes={dt}")
