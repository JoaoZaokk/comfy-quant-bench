"""Gera o dataset público do professor FLUX.2-klein-4B BF16 no HF (roda NA VM, em /content/qat).

Pedido do dono (25/09): gravar UMA vez tudo o que os QATs já precisaram e guardar num dataset do HF, para
baixar (ou streamar) em vez de regravar a cada VM. Por amostra (prompt, semente):

    professor/<chave>.pt   o shard do professor (o mesmo formato de `grava_professor`: comum + 8 passos)
    images/<chave>.png     a imagem final do professor (diffusers, 8 passos, guidance 1)
    metadata.jsonl         uma linha por amostra: chave, prompt, semente, passos, size, lista, arquivos

`<chave>` = sha1(prompt)[:16] + `_s<semente>_n<passos>_<size>` -- a MESMA de `cache_professor` do
`qat_ternario_klein.py --professor-hf`, então o QAT reaproveita direto.

Retomável: pula o que o repo já tem; sobe em lotes de 64 numa thread (a gravação não para) e apaga o
local depois de subir. Config em /content/qat/ds_config.json. Log em /content/ds/ds.log, termina com FIM.
"""
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import torch

Q = Path("/content/qat")
sys.path.insert(0, str(Q))
import qat_ternario_klein as qk  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

cfg = json.loads((Q / "ds_config.json").read_text())
REPO, PASSOS, SIZE = cfg["repo"], cfg.get("passos", 8), cfg.get("size", 1024)
BASE = Path("/content/ds")
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=not cfg.get("publico", True), exist_ok=True)


def chave(p: str, s: int) -> str:
    return f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]}_s{s}_n{PASSOS}_{SIZE}"


remotos = {f.split("/", 1)[1].rsplit(".", 1)[0] for f in api.list_repo_files(REPO, repo_type="dataset")
           if f.startswith("professor/")}
meta: dict[str, dict] = {}
try:
    for ln in Path(hf_hub_download(REPO, "metadata.jsonl", repo_type="dataset")).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            meta[r["chave"]] = r
except Exception:  # noqa: BLE001 -- repo novo
    pass
qk.log(f"dataset {REPO}: {len(remotos)} amostras ja no repo, {len(meta)} linhas de metadata")

fila: queue.Queue = queue.Queue()
lote: list = []
trava = threading.Lock()


def sobe_lote(itens) -> None:
    t0 = time.perf_counter()
    stg = BASE / f"_stg_{int(t0 * 1000)}"
    (stg / "professor").mkdir(parents=True)
    (stg / "images").mkdir(parents=True)
    for c, arq, _ in itens:
        os.link(arq, stg / "professor" / f"{c}.pt")
        if arq.with_suffix(".png").is_file():
            os.link(arq.with_suffix(".png"), stg / "images" / f"{c}.png")
    with trava:
        for c, _, linha in itens:
            meta[c] = linha
        (stg / "metadata.jsonl").write_text("".join(json.dumps(meta[k], ensure_ascii=False) + "\n"
                                                    for k in sorted(meta)), encoding="utf-8")
    tam = sum(f.stat().st_size for f in stg.rglob("*") if f.is_file())
    for tent in range(3):
        try:
            api.upload_folder(folder_path=str(stg), repo_id=REPO, repo_type="dataset",
                              commit_message=f"{len(itens)} amostras do professor")
            break
        except Exception as e:  # noqa: BLE001
            qk.log(f"HF: lote falhou (tentativa {tent + 1}): {type(e).__name__}: {str(e)[:300]}")
            time.sleep(30)
    else:
        qk.log(f"FALHOU envio de lote ({len(itens)} amostras) -- arquivos locais mantidos em {stg}")
        return
    for _, arq, _ in itens:
        arq.unlink(missing_ok=True)
        arq.with_suffix(".png").unlink(missing_ok=True)
    shutil.rmtree(stg, ignore_errors=True)
    dt = time.perf_counter() - t0
    qk.log(f"HF: lote de {len(itens)} enviado ({tam / 2**30:.2f} GiB, {dt:.0f} s, "
           f"{tam / 2**20 / max(dt, 1e-9):.0f} MB/s) [MEDIDO]; total no repo ~{len(meta)}")


def trabalhador() -> None:
    while True:
        itens = fila.get()
        if itens is None:
            fila.task_done()
            return
        try:
            sobe_lote(itens)
        finally:
            fila.task_done()


threading.Thread(target=trabalhador, daemon=True).start()


def main() -> None:
    a = types.SimpleNamespace(professor=cfg["professor"], passos=PASSOS, size=SIZE, sementes=[1])
    raiz = Path(cfg["raiz"])
    dev = torch.device("cuda:0")
    for lst in cfg["listas"]:
        prompts = qk.le_prompts(Q / lst["arquivo"])
        for s in lst["sementes"]:
            faltam = [p for p in prompts if chave(p, s) not in remotos and chave(p, s) not in meta]
            qk.log(f"lista {lst['nome']} semente {s}: {len(prompts)} prompts, {len(faltam)} a gravar")
            if not faltam:
                continue
            a.sementes = [s]
            destino = BASE / f"{lst['nome']}_s{s}"

            def ao_gravar(i, p, sem, arq, _lst=lst["nome"]):
                c = chave(p, sem)
                lote.append((c, arq, {"chave": c, "prompt": p, "semente": sem, "passos": PASSOS, "size": SIZE,
                                      "lista": _lst, "file_name": f"images/{c}.png",
                                      "shard": f"professor/{c}.pt"}))
                if len(lote) >= 64:
                    fila.put(list(lote))
                    lote.clear()

            qk.grava_professor(a, raiz, faltam, destino, dev, imagens=True, ao_gravar=ao_gravar)
    if lote:
        fila.put(list(lote))
        lote.clear()
    fila.put(None)
    fila.join()
    qk.log(f"dataset pronto: {len(meta)} amostras em {REPO}")


if __name__ == "__main__":
    main()
    print(f"FIM {time.strftime('%H:%M:%S')}", flush=True)
