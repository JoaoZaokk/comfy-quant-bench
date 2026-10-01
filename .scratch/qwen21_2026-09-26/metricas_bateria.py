"""Metricas da bateria Qwen-Image-2.1: cada DiT contra o bf16 do mesmo prompt/seed, mais tempo e memoria do log.

    python_embeded\\python.exe -s metricas_bateria.py <saida.json> <fonte> [<fonte> ...]

Cada <fonte> e uma de:
  resultados.jsonl  -- registros do executor (`tools/comfy_client.py`, `.scratch/roda_eros_2gpu.py`):
                       junta por grafo, tempo = server_side_s do /history, imagem = `files` do registro.
                       E o modo certo; nao depende de ordem nem de contador de arquivo.
  log:ordem         -- modo antigo, para as baterias ja rodadas sem JSONL: o stderr do ComfyUI casado
                       com `bateria/ordem*.txt` pela POSICAO. Recusa (exit 2) se o numero de
                       execucoes do log diferir do de grafos -- antes so avisava, e um prompt pulado
                       deslocava todos os tempos seguintes para o braco errado (revisao 2026-09-29).
                       Imagem resolvida por `imagem_unica` (sem `_00001_` fixo).
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
from metricas_imagem import carrega, grao, identica, imagem_unica, medir, sha_pixels  # noqa: E402

AQUI = Path(__file__).parent
SAIDA_COMFY = Path("F:/COMFY_PORTABLE/ComfyUI/output")
IMG = SAIDA_COMFY / "qwen21_bateria"
NOME = re.compile(r"^(\w+?)_p(\d+)_s(\d+)\.json$")


def _chave(grafo: str):
    m = NOME.search(Path(grafo.replace("\\", "/")).name)
    if not m:
        raise SystemExit(f"nome de grafo fora do padrao <dit>_p<i>_s<s>.json: {grafo}")
    dit, p_, s_ = m.groups()
    return dit, int(p_), int(s_)


def tempos(log: Path):
    """Uma entrada por prompt executado: it/s final do tqdm, segundos, e a ultima carga de modelo vista."""
    out, its, carga = [], None, None
    for linha in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"25/25 \[[^\]]*?([\d.]+)(it/s|s/it)\]", linha)
        if m:
            v = float(m.group(1))
            its = v if m.group(2) == "it/s" else 1 / v
        m = re.search(r"(loaded completely|loaded partially)[^\n]*", linha)
        if m:
            carga = m.group(0)[:200]
        # acima de um minuto o ComfyUI escreve hh:mm:ss em vez de "N seconds"
        m = re.search(r"Prompt executed in (?:([\d.]+) seconds|(\d+):(\d+):([\d.]+))", linha)
        if m:
            seg = float(m.group(1)) if m.group(1) else int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))
            out.append({"its": its, "seg": seg, "carga": carga})
            its = None
    return out


def de_log(log: Path, ordem: Path) -> list[dict]:
    grafos = [g for g in ordem.read_text().split() if g]
    ts = tempos(log)
    if len(ts) != len(grafos):
        raise SystemExit(f"RECUSADO {log.name}: {len(ts)} execucoes para {len(grafos)} grafos de "
                         f"{ordem.name}. O casamento e por posicao; com contagens diferentes todo "
                         "tempo depois do buraco cairia no braco errado. Use o JSONL do executor.")
    linhas = []
    for g, t_ in zip(grafos, ts):
        dit, p_, s_ = _chave(g)
        linhas.append({"dit": dit, "p": p_, "s": s_, **t_,
                       "arquivo": str(imagem_unica(IMG / dit, f"p{p_}_s{s_}"))})
    return linhas


def de_jsonl(caminho: Path, falhas: list) -> list[dict]:
    linhas = []
    for bruta in caminho.read_text(encoding="utf-8").splitlines():
        if not bruta.strip():
            continue
        r = json.loads(bruta)
        dit, p_, s_ = _chave(r["grafo"])
        imagens = [f for f in r.get("files") or [] if f.endswith(".png")]
        if r.get("status") != "success" or len(imagens) != 1:
            falhas.append({"grafo": r["grafo"], "status": r.get("status"), "erro": r.get("erro"),
                           "imagens": imagens})
            continue
        linhas.append({"dit": dit, "p": p_, "s": s_, "its": None, "carga": None,
                       # um cache hit nao cronometrou nada: a imagem vale, o tempo nao
                       "seg": None if r.get("cache_hit") else r.get("server_side_s"),
                       "cache_hit": bool(r.get("cache_hit")), "prompt_id": r.get("prompt_id"),
                       "placa": r.get("placa"),
                       "arquivo": str(SAIDA_COMFY / imagens[0].lstrip("/"))})
    return linhas


def _mediana(valores):
    v = [x for x in valores if x is not None]
    return st.median(v) if v else None


def main():
    saida = Path(sys.argv[1])
    execucoes, falhas = [], []
    for fonte in sys.argv[2:]:
        if fonte.endswith(".jsonl"):
            execucoes += de_jsonl(Path(fonte), falhas)
        else:
            # rsplit: um log com letra de drive (F:\...) nao pode partir no ':' do drive
            log, ordem = fonte.rsplit(":", 1)
            execucoes += de_log(Path(log), AQUI / ordem)
    por_dit = {}
    for L in execucoes:
        por_dit.setdefault(L.pop("dit"), []).append(L)
    if "bf16" not in por_dit:
        raise SystemExit("sem o braco bf16 nas fontes: nao ha referencia")

    res = {"por_imagem": {}, "resumo": {}, "falhas": falhas}
    refs = {}
    for L in por_dit["bf16"]:
        x = carrega(Path(L["arquivo"]))
        # hash dos PIXELS: o PNG leva o grafo nos metadados, entao bytes do arquivo sempre diferem
        L["sha256"] = sha_pixels(x)
        refs[(L["p"], L["s"])] = (x, grao(x))
    for dit, linhas in por_dit.items():
        for L in linhas:
            if dit == "bf16":
                continue
            ref = refs.get((L["p"], L["s"]))
            if ref is None:
                raise SystemExit(f"{dit} p{L['p']} s{L['s']}: sem a imagem bf16 correspondente")
            r, g_ref = ref
            x = carrega(Path(L["arquivo"]))
            L.update({"sha256": sha_pixels(x), "identica": identica(x, r), **medir(x, r, grao_ref=g_ref)})
        res["por_imagem"][dit] = linhas
        # a 1a imagem de cada DiT inclui a carga do modelo: fica fora da media de tempo
        quentes = linhas[1:] or linhas  # controle de 1 imagem: sem media quente
        r = {"n": len(linhas), "its_mediana": _mediana(L["its"] for L in quentes),
             "seg_mediana_quente": _mediana(L["seg"] for L in quentes), "seg_primeira": linhas[0]["seg"],
             "carga": linhas[-1]["carga"], "cache_hits": sum(bool(L.get("cache_hit")) for L in linhas)}
        if dit != "bf16":
            fin = [L["psnr"] for L in linhas if L["psnr"] != float("inf")]
            r.update({"identicas": sum(L["identica"] for L in linhas),
                      "psnr_media_finita": st.mean(fin) if fin else None,
                      "ssim_media": st.mean(L["ssim"] for L in linhas),
                      "msssim_media": st.mean(L["msssim"] for L in linhas),
                      "msssim_min": min(L["msssim"] for L in linhas),
                      "grao_media": st.mean(L["grao"] for L in linhas)})
        res["resumo"][dit] = r
    saida.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for dit, r in res["resumo"].items():
        print(dit, json.dumps(r))
    if falhas:
        print(f"FALHAS ({len(falhas)}), fora das metricas: {json.dumps(falhas)[:2000]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
