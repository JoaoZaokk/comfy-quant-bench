"""Selecao de camadas nos headers REAIS do disco, com o codigo de um tools dado. So le headers.

    selecao_real.py <tools_dir> <saida.json>
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

tools = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tools))

import quant_int8  # noqa: E402
import quant_mixed  # noqa: E402
import quant_w4a4  # noqa: E402
import quant_w4a8  # noqa: E402
from calibrate_activations import PROFILE_FILE_PATTERNS  # noqa: E402

RAIZES = [Path("F:/COMFY_PORTABLE/ComfyUI/models") / d for d in ("diffusion_models", "unet", "text_encoders", "clip")]
RAIZES += [Path("D:/ComfyUI-Models") / d for d in ("diffusion_models", "unet", "text_encoders", "clip")]


def header(p: Path) -> dict:
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 200_000_000:
            raise ValueError("header grande demais")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> int:
    out = {}
    for raiz in RAIZES:
        if not raiz.is_dir():
            continue
        for p in sorted(raiz.rglob("*.safetensors")):
            try:
                h = header(p)
            except Exception as e:  # noqa: BLE001
                out[str(p)] = {"erro": repr(e)}
                continue
            r: dict = {}
            for nome, mod in (("w4a4", quant_w4a4), ("w4a8", quant_w4a8)):
                try:
                    r[f"detect_{nome}"] = mod.detect_profile(p, list(h))
                except ValueError as e:
                    r[f"detect_{nome}"] = f"ValueError: {e}"
            for prof in quant_w4a8.PROFILE_PATTERNS:
                s = {
                    "w4a4": quant_w4a4.selected_layers(h, prof, 256),
                    "w4a4_cg64": quant_w4a4.selected_layers(h, prof, 64),
                    "w4a8": quant_w4a8.selected_layers(h, prof, 16, 256),
                    "int8c": quant_int8.selected_layers(h, prof, True, 256),
                    "int8": quant_int8.selected_layers(h, prof, False, 256),
                    "awq": [n for n, i in h.items() if quant_w4a8.PROFILE_PATTERNS[prof].fullmatch(n)
                            and i["dtype"] in quant_w4a8.HIGH_PRECISION_DTYPES and len(i["shape"]) == 2
                            and i["shape"][1] % 32 == 0],
                }
                s = {k: v for k, v in s.items() if v}
                if s:
                    r[prof] = s
            for prof in PROFILE_FILE_PATTERNS:
                sel = quant_mixed.selected_layers(h, prof, 256)
                if sel:
                    r[f"mixed_{prof}"] = sel
            out[str(p)] = r
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(out)} arquivos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
