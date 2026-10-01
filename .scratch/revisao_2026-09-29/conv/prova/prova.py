"""Prova de equivalencia antes/depois dos conversores, em fontes sinteticas. So CPU.

    prova.py antes     # roda os casos com conv/antes_tools (codigo congelado) -> prova/antes/
    prova.py depois    # roda os casos com tools/ atual               -> prova/depois/
    prova.py compara   # sha256 das saidas e sidecars normalizados, caso a caso
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = Path("F:/COMFY_PORTABLE")
PY = RAIZ / "python_embeded" / "python.exe"
F = AQUI / "fontes"
TOOLS = {"antes": RAIZ / ".scratch" / "antes_tools_conv_20260929", "depois": RAIZ / "tools"}


def casos(out: Path) -> list[tuple[str, str, list[str], dict, list[str]]]:
    """(nome, modulo, args, env extra, arquivos a copiar para o dir do caso antes de rodar)."""
    g = str(F / "gemma_sint_bf16.safetensors")
    lt = str(F / "ltx_sint_bf16.safetensors")
    z = str(F / "zimage_sint_native_bf16.safetensors")
    o = lambda c, n="saida.safetensors": str(out / c / n)  # noqa: E731
    return [
        ("w4a4_gemma", "quant_w4a4", ["--input", g, "--output", o("w4a4_gemma"), "--profile", "gemma"], {}, []),
        ("w4a4_gemma_cg64", "quant_w4a4", ["--input", g, "--output", o("w4a4_gemma_cg64"), "--profile", "gemma",
                                           "--convrot-groupsize", "64"], {}, []),
        ("w4a4_ltx", "quant_w4a4", ["--input", lt, "--output", o("w4a4_ltx"), "--profile", "ltx_2_5"], {}, []),
        ("w4a4_ltx_auto", "quant_w4a4", ["--input", lt, "--output", o("w4a4_ltx_auto")], {}, []),
        ("w4a8_gemma", "quant_w4a8", ["--input", g, "--output", o("w4a8_gemma")], {}, []),
        ("w4a8_gemma_nocb", "quant_w4a8", ["--input", g, "--output", o("w4a8_gemma_nocb"), "--no-codebook",
                                           "--group-size", "32"], {}, []),
        ("w4a8_ltx", "quant_w4a8", ["--input", lt, "--output", o("w4a8_ltx")], {}, []),
        ("int8_gemma", "quant_int8", ["--input", g, "--output", o("int8_gemma")], {}, []),
        ("int8_gemma_noconvrot", "quant_int8", ["--input", g, "--output", o("int8_gemma_noconvrot"),
                                                "--no-convrot"], {}, []),
        ("int8_ltx", "quant_int8", ["--input", lt, "--output", o("int8_ltx")], {}, []),
        ("mixed_zimage", "quant_mixed", ["--input", z, "--output", o("mixed_zimage"), "--analysis",
                                         str(F / "analise_mixed.json"), "--keep-bf16-error", "0.5",
                                         "--budget", "0.4", "--promote-error", "0.2"], {}, []),
        ("mixed_zimage_somente", "quant_mixed", ["--input", z, "--output", o("mixed_zimage_somente"), "--analysis",
                                                 str(F / "analise_mixed_somente.json"), "--somente-w4a4",
                                                 "--uncalibrated", "bf16"], {}, []),
        ("smooth_gemma", "quant_w4a4_smooth", ["--source", g, "--output", o("smooth_gemma"),
                                               "--calibrate-with", "x"], {"FAKE_CALIB": "1"}, []),
        ("smooth_gemma_bf16", "quant_w4a4_smooth", ["--source", g, "--output", o("smooth_gemma_bf16"),
                                                    "--calibrate-with", "x"],
         {"FAKE_CALIB": "1", "WRAP_BF16": "1"}, []),
        ("awq_gemma", "quant_awq_w4a16", ["--input", o("awq_gemma", "gemma_sint_bf16.safetensors"),
                                          "--profile", "gemma"], {}, ["gemma_sint_bf16.safetensors"]),
        ("weight_only", "quant_weight_only", ["--input", o("weight_only", "ja_quantizado.safetensors")], {},
         ["ja_quantizado.safetensors"]),
        ("te_int8", "quant_te_residuos", ["--input", g, "--format", "int8", "--output", o("te_int8")], {}, []),
        ("te_fp8", "quant_te_residuos", ["--input", g, "--format", "fp8", "--output", o("te_fp8")], {}, []),
        ("svdq", "svdq_to_bf16", ["--input", str(F / "svdq_sint.safetensors"), "--output", o("svdq"),
                                  "--device", "cpu", "--verify", "2", "--reference",
                                  str(F / "svdq_ref_bf16.safetensors")], {"FAKE_NUNCHAKU": "1"}, []),
        ("svdq_fused", "svdq_to_bf16", ["--input", str(F / "svdq_sint.safetensors"), "--output", o("svdq_fused"),
                                        "--device", "cpu", "--verify", "1", "--keep-fused"],
         {"FAKE_NUNCHAKU": "1"}, []),
        ("extrai", "extrai_transformer", [str(F / "checkpoint_sint.safetensors"), o("extrai")], {}, []),
        ("mistura", "mistura_klein", ["--base", str(F / "klein_base.safetensors"), "--doador",
                                      str(F / "klein_doador.safetensors"), "--regex",
                                      r"double_blocks\.\d+\.txt_attn\.qkv\.weight", "--saida", o("mistura"),
                                      "--esperadas", "2"], {}, []),
        ("mistura_rtn", "mistura_klein", ["--base", str(F / "klein_base.safetensors"), "--doador",
                                          str(F / "klein_doador.safetensors"), "--regex",
                                          r"double_blocks\.\d+\.txt_.*\.weight", "--saida", o("mistura_rtn"),
                                          "--rtn4", "64", "--bits", "3"], {}, []),
        ("transplanta", "transplanta_klein", ["--a", str(F / "klein_tern_A.safetensors"), "--b",
                                              str(F / "klein_tern_B.safetensors"), "--rotulo-b", "braco",
                                              "--dir", str(out / "transplanta")], {}, []),
        ("grava", "grava_pesos_recuperados", ["--modelo", z, "--calib", str(F / "calib_grava.pt"),
                                              "--saida", o("grava")], {}, []),
        ("grava_int8", "grava_pesos_recuperados", ["--modelo", z, "--calib", str(F / "calib_grava.pt"),
                                                   "--saida", o("grava_int8"), "--int8"], {}, []),
    ]


def gera_analises() -> None:
    sys.path.insert(0, str(RAIZ / "tools"))
    from calibrate_activations import safetensors_identity_digest

    z = F / "zimage_sint_native_bf16.safetensors"
    with z.open("rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    import re
    pat = re.compile(r"^(?:layers|noise_refiner|context_refiner)\.\d+\.(?:attention\.(?:qkv|out)|feed_forward\.w[123])$")
    stems = sorted(k.removesuffix(".weight") for k in h if k.endswith(".weight") and pat.match(k.removesuffix(".weight")))
    rows = []
    for i, s in enumerate(stems):
        if i == 3:
            rows.append({"layer": s, "shape": h[s + ".weight"]["shape"], "calibrated": False})
            continue
        e4 = 0.05 + 0.04 * (i % 7)
        e8 = e4 / 3 + (0.5 if i == 5 else 0.0)
        rows.append({"layer": s, "shape": h[s + ".weight"]["shape"], "calibrated": True, "err_bf16": 0.004,
                     "err_w4a4": e4, "err_w4a8": e8, "crest_p99": 3.0 + i, "rows": 128})
    base = {"profile": "zimage", "source": str(z), "source_identity_sha256": safetensors_identity_digest(z),
            "measure_dtype": "torch.bfloat16", "group_size": 16, "convrot_groupsize": 256,
            "calibration": {"sint": True}}
    (F / "analise_mixed.json").write_text(json.dumps(base | {"layers": rows}, indent=2), encoding="utf-8")
    somente = [{k: v for k, v in r.items() if k != "err_w4a8"} for r in rows]
    (F / "analise_mixed_somente.json").write_text(json.dumps(base | {"layers": somente}, indent=2),
                                                  encoding="utf-8")


def roda(rotulo: str, so: set[str] | None = None) -> int:
    out = AQUI / rotulo
    if not (F / "SHA256.txt").is_file():
        subprocess.run([str(PY), "-s", str(AQUI / "gera_fontes.py")], check=True,
                       env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"})
    if not (F / "analise_mixed.json").is_file():
        gera_analises()
    falhas = 0
    for nome, mod, args, env, copiar in casos(out):
        if so and nome not in so:
            continue
        d = out / nome
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for c in copiar:
            shutil.copyfile(F / c, d / c)
        p = subprocess.run([str(PY), "-s", str(AQUI / "roda_um.py"), str(TOOLS[rotulo]), mod, *args],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"} | env,
                           timeout=900)
        (d / "_stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        (d / "_rc.txt").write_text(str(p.returncode), encoding="utf-8")
        print(f"{rotulo:6s} {nome:24s} rc={p.returncode}")
        falhas += p.returncode != 0
    return falhas


VOLATEIS = {"conversion_seconds", "seconds", "comfy_version"}


def normaliza_sidecar(p: Path, rotulo: str) -> str:
    txt = p.read_text(encoding="utf-8")
    try:
        j = json.loads(txt)
    except json.JSONDecodeError:
        return txt

    def limpa(x):
        if isinstance(x, dict):
            return {k: limpa(v) for k, v in x.items() if k not in VOLATEIS}
        if isinstance(x, list):
            return [limpa(v) for v in x]
        if isinstance(x, str):
            return x.replace(str(AQUI / rotulo), "<OUT>").replace((str(AQUI / rotulo)).replace("\\", "/"), "<OUT>")
        return x
    return json.dumps(limpa(j), indent=2, ensure_ascii=False)


# Casos cujo codigo ANTIGO gravava com safetensors.save_file, que ordena o __metadata__ por um
# HashMap de semente aleatoria (medido: duas execucoes do codigo antigo, dois sha256).
SO_METADATA_EM_OUTRA_ORDEM = {"grava", "grava_int8"}


def equivalente(a: Path, b: Path) -> bool:
    import struct
    ra, rb = a.read_bytes(), b.read_bytes()
    na, nb = struct.unpack("<Q", ra[:8])[0], struct.unpack("<Q", rb[:8])[0]
    ha, hb = json.loads(ra[8:8 + na]), json.loads(rb[8:8 + nb])
    if ha != hb or list(k for k in ha if k != "__metadata__") != list(k for k in hb if k != "__metadata__"):
        return False
    return ra[8 + na:] == rb[8 + nb:]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compara() -> int:
    linhas, ruins = [], 0
    for nome, *_ in casos(AQUI / "antes"):
        a, d = AQUI / "antes" / nome, AQUI / "depois" / nome
        if not a.is_dir() or not d.is_dir():
            linhas.append(f"{nome:24s} FALTA ({'antes' if not a.is_dir() else 'depois'})")
            ruins += 1
            continue
        ra, rd = (a / "_rc.txt").read_text().strip(), (d / "_rc.txt").read_text().strip()
        arq = sorted({p.name for p in a.iterdir() if not p.name.startswith("_")}
                     | {p.name for p in d.iterdir() if not p.name.startswith("_")})
        partes = [f"rc {ra}/{rd}"]
        ok = ra == rd
        for n in arq:
            pa, pd = a / n, d / n
            if not pa.exists() or not pd.exists():
                partes.append(f"{n}: so {'depois' if pd.exists() else 'antes'}")
                ok = False
                continue
            if n.endswith(".json"):
                igual = normaliza_sidecar(pa, "antes") == normaliza_sidecar(pd, "depois")
                crlf = (bytes([13, 10]) in pa.read_bytes()) == (bytes([13, 10]) in pd.read_bytes())
                partes.append(f"{n}: sidecar {'IGUAL' if igual else 'DIFERENTE'}"
                              f"{'' if crlf else ' (fim de linha difere)'}")
                igual &= crlf
            else:
                igual = sha(pa) == sha(pd)
                if igual:
                    partes.append(f"{n}: IGUAL {sha(pa)[:16]}")
                elif nome in SO_METADATA_EM_OUTRA_ORDEM and equivalente(pa, pd):
                    igual = True
                    partes.append(f"{n}: EQUIVALENTE (header parseado e dados identicos; so a ordem "
                                  f"das chaves do __metadata__ difere -- save_file antigo e aleatorio)")
                else:
                    partes.append(f"{n}: DIFERENTE")
            ok &= igual
        ruins += not ok
        linhas.append(f"{'OK  ' if ok else 'DIF '} {nome:24s} " + "; ".join(partes))
    txt = "\n".join(linhas)
    print(txt)
    (AQUI / "comparacao.txt").write_text(txt + "\n", encoding="utf-8")
    return ruins


if __name__ == "__main__":
    modo = sys.argv[1]
    so = set(sys.argv[2:]) or None
    sys.exit(compara() if modo == "compara" else roda(modo, so))
