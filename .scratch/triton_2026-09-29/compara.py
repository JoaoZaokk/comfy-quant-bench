"""C2/C3/C4 do critério: ON × OFF, OFF × referência anterior à revisão, tempos. Lê os JSONL por prompt_id.

    python compara.py > resultado_metricas.json
"""
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "tools"))
from metricas_imagem import carrega, identica, medir  # noqa: E402

D = pathlib.Path(__file__).resolve().parent
OUT_DIR = RAIZ / "ComfyUI/output"
REFS = {  # referência anterior à revisão (mesmo grafo e seed, rodada fria)
    "Q1": "qwen21_bateria/w4a16_q4_1_nativo", "Q3": "qwen21_bateria/nosso_w4a8",
    "Q4": "qwen21_bateria/nosso_w4a4", "Q5": "qwen21_bateria/nosso_int8",
    "K1": "lowbit_2026-09-27/principal/ref_bf16_ternario", "K2": "lowbit_2026-09-27/principal/ternario_mlx",
}


def registros(flag):
    p = D / flag / "resultados.jsonl"
    regs = [json.loads(linha) for linha in p.read_text(encoding="utf-8").splitlines() if linha.strip()] if p.exists() else []
    return {(r["caso"], r["rodada"]): r for r in regs}  # o último registro de cada (caso, rodada) vale


def imagem(reg):
    fs = [f for f in reg.get("files") or [] if f.endswith(".png")]
    return OUT_DIR / fs[0] if len(fs) == 1 else None


def compara(a, b):
    x, r = carrega(a), carrega(b)
    if identica(x, r):
        return {"identica": True}
    return {"identica": False, **{k: round(v, 4) for k, v in medir(x, r, quais=("psnr", "msssim")).items()}}


def main():
    off, on = registros("off"), registros("on")
    saida = {}
    for (caso, rodada), r_off in sorted(off.items()):
        linha = {"off_status": r_off["status"], "off_s": r_off.get("server_side_s")}
        r_on = on.get((caso, rodada))
        if r_on:
            linha.update(on_status=r_on["status"], on_s=r_on.get("server_side_s"))
            if imagem(r_on) and imagem(r_off):
                linha["on_x_off"] = compara(imagem(r_on), imagem(r_off))
        if rodada == "frio" and caso in REFS and imagem(r_off):
            seed = r_off["seed"]
            refs = sorted((OUT_DIR / REFS[caso]).glob(f"p0_s{seed}_*.png"))
            if refs:
                linha["off_x_ref"] = compara(imagem(r_off), refs[0]) | {"ref": str(refs[0].relative_to(OUT_DIR))}
        saida[f"{caso}/{rodada}"] = linha
    print(json.dumps(saida, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
