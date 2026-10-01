"""Gera o dataset público do professor FLUX.2-klein-4B BF16 no HF (roda NA VM, em /content/qat).

Pedido do dono (25/09): gravar UMA vez tudo o que os QATs já precisaram e guardar num dataset do HF, para
baixar (ou streamar) em vez de regravar a cada VM. Por amostra (prompt, semente):

    professor/<chave>.pt       o shard do professor (o mesmo formato de `grava_professor`: comum + 8 passos)
    images/<chave>.png         a imagem final do professor (diffusers, 8 passos, guidance 1)
    metadata/<vm>.jsonl        uma linha por amostra QUE ESTA VM GEROU: chave, prompt, semente, passos, size,
                               lista, arquivos e `camadas` (por nn.Linear, mínimo / médio / máximo / mediana
                               da ENTRADA nos 8 passos; por passo, no shard em `estat`)
    metadata.jsonl             (legado, só leitura) o índice único das gerações antigas

POR QUE UM ARQUIVO POR VM. Antes cada lote relia `metadata.jsonl`, juntava e o regravava inteiro: com duas
VMs em paralelo quem gravasse por último apagava as linhas da outra, e uma falha de leitura qualquer era
tratada como "repo novo" e sobrescrevia o índice com só as linhas locais. Agora cada VM só escreve o PRÓPRIO
arquivo; a leitura junta todos. Só "arquivo não existe" vira vazio; qualquer outro erro de leitura aborta.
(O visualizador de imagefolder do HF só associa `metadata.jsonl` da raiz: as amostras novas aparecem no
repo, mas não na prévia com colunas.)

`<chave>` = sha1(prompt)[:16] + `_s<semente>_n<passos>_<size>` -- a MESMA do nome local dos shards e do
`cache_professor` do QAT. Retomável: pula o que o repo já tem; sobe em lotes de 64 numa thread e apaga o
local depois de subir; shard local que sobrou de uma execução morta (até 63 no lote em curso) é ENVIADO no
próximo lançamento, não pulado. Config em /content/qat/ds_config.json. Log em /content/ds/ds.log (termina
com FIM) e estado em /content/ds/status.json.
"""
import json
import os
import queue
import shutil
import socket
import sys
import threading
import time
import types
from pathlib import Path

import torch

Q = Path("/content/qat")
BASE = Path("/content/ds")


def erro_de_ausencia(e: Exception) -> bool:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    return isinstance(e, (EntryNotFoundError, RepositoryNotFoundError))


class Metadados:
    """Índice do dataset: leitura de todos os arquivos (legado + um por VM), escrita só do desta VM."""

    def __init__(self, api, repo: str, vm: str, baixa):
        self.api, self.repo, self.vm, self._baixa = api, repo, vm, baixa
        self.trava = threading.Lock()
        self.todos: dict[str, dict] = {}
        self.meus: dict[str, dict] = {}

    @property
    def meu_arquivo(self) -> str:
        return f"metadata/{self.vm}.jsonl"

    def _le(self, caminho: str) -> list[dict]:
        try:
            txt = Path(self._baixa(self.repo, caminho, repo_type="dataset", force_download=True)).read_text()
        except Exception as e:  # noqa: BLE001 -- ausencia vira vazio; o resto aborta
            if erro_de_ausencia(e):
                return []
            raise SystemExit(f"FALHOU: leitura de {caminho} em {self.repo}: {type(e).__name__}: {str(e)[:300]}") from None
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]

    def rele(self) -> None:
        """Junta o que já existe no repo (outra VM pode estar gerando em paralelo, dividindo as sementes)."""
        try:
            arquivos = self.api.list_repo_files(self.repo, repo_type="dataset")
        except Exception as e:
            if not erro_de_ausencia(e):
                raise
            arquivos = []
        caminhos = [c for c in arquivos if c == "metadata.jsonl" or (c.startswith("metadata/") and c.endswith(".jsonl"))]
        with self.trava:
            for c in caminhos:
                for r in self._le(c):
                    if "camadas" in r or r["chave"] not in self.todos:
                        self.todos[r["chave"]] = r
                    if c == self.meu_arquivo:
                        self.meus[r["chave"]] = r

    def feito(self, chave: str) -> bool:
        # sem a coluna `camadas` conta como faltando (amostras gravadas antes das estatísticas)
        return "camadas" in self.todos.get(chave, {})

    def acrescenta(self, linhas: list[dict]) -> str:
        """Texto do arquivo desta VM com as linhas novas (só este arquivo é reescrito)."""
        with self.trava:
            for r in linhas:
                self.todos[r["chave"]] = r
                self.meus[r["chave"]] = r
            return "".join(json.dumps(self.meus[k], ensure_ascii=False) + "\n" for k in sorted(self.meus))


def camadas_de(estat) -> dict:
    camadas = {}
    for nome in (estat[0] if estat else {}):
        vals = [e[nome] for e in estat if nome in e]
        meds = sorted(v[3] for v in vals)
        camadas[nome] = {"minimo": min(v[0] for v in vals), "medio": sum(v[1] for v in vals) / len(vals),
                         "maximo": max(v[2] for v in vals), "mediana": meds[len(meds) // 2]}
    return camadas


class Gerador:
    def __init__(self, cfg: dict, api, meta: Metadados, log):
        self.cfg, self.api, self.meta, self.log = cfg, api, meta, log
        self.passos, self.size = cfg.get("passos", 8), cfg.get("size", 1024)
        self.fila: queue.Queue = queue.Queue()
        self.lote: list = []
        self.falhou = threading.Event()
        threading.Thread(target=self._trabalhador, daemon=True).start()

    def chave(self, p: str, s: int) -> str:
        from qat_klein.professor import chave
        return chave(p, s, self.passos, self.size)

    def enfileira(self, arq: Path, prompt: str, semente: int, estat, lista: str) -> None:
        c = self.chave(prompt, semente)
        self.lote.append((c, arq, {"chave": c, "prompt": prompt, "semente": semente, "passos": self.passos,
                                   "size": self.size, "lista": lista, "file_name": f"images/{c}.png",
                                   "shard": f"professor/{c}.pt", "camadas": camadas_de(estat)}))
        if len(self.lote) >= 64:
            self.fila.put(list(self.lote))
            self.lote.clear()

    def sobras(self, destino: Path, lista: str) -> int:
        """Shards locais de uma execução morta que o repo ainda não tem: vão para o lote, não são pulados."""
        from qat_klein.professor import lista_shards
        n = 0
        for arq in lista_shards(destino):
            sh = torch.load(arq, map_location="cpu", weights_only=False, mmap=True)
            if self.meta.feito(self.chave(sh["prompt"], sh["semente"])):
                continue
            if not sh.get("estat"):  # sem estatisticas: sai do caminho (nao apaga) para ser regravado
                lado = destino / "_sem_estat"
                lado.mkdir(exist_ok=True)
                os.replace(arq, lado / arq.name)
                self.log(f"AVISO: {arq.name} sem estatisticas; movido para {lado.name}/ e sera regravado")
                continue
            if arq.stem != self.chave(sh["prompt"], sh["semente"]):  # legado p####_s#: renomeia pela chave
                novo = arq.with_name(self.chave(sh["prompt"], sh["semente"]) + ".pt")
                os.replace(arq, novo)
                if arq.with_suffix(".png").is_file():
                    os.replace(arq.with_suffix(".png"), novo.with_suffix(".png"))
                arq = novo
            self.enfileira(arq, sh["prompt"], sh["semente"], sh["estat"], lista)
            n += 1
        return n

    def sobe_lote(self, itens) -> None:
        t0 = time.perf_counter()
        stg = BASE / f"_stg_{int(t0 * 1000)}"
        (stg / "professor").mkdir(parents=True)
        (stg / "images").mkdir(parents=True)
        (stg / "metadata").mkdir(parents=True)
        for c, arq, _ in itens:
            os.link(arq, stg / "professor" / f"{c}.pt")
            if arq.with_suffix(".png").is_file():
                os.link(arq.with_suffix(".png"), stg / "images" / f"{c}.png")
        (stg / self.meta.meu_arquivo).write_text(self.meta.acrescenta([linha for _, _, linha in itens]),
                                                 encoding="utf-8")
        tam = sum(f.stat().st_size for f in stg.rglob("*") if f.is_file())
        for tent in range(3):
            try:
                self.api.upload_folder(folder_path=str(stg), repo_id=self.cfg["repo"], repo_type="dataset",
                                       commit_message=f"{len(itens)} amostras do professor ({self.meta.vm})")
                break
            except Exception as e:  # noqa: BLE001
                self.log(f"HF: lote falhou (tentativa {tent + 1}): {type(e).__name__}: {str(e)[:300]}")
                time.sleep(30)
        else:
            raise RuntimeError(f"envio de lote ({len(itens)} amostras) falhou 3x -- arquivos locais mantidos em {stg}")
        for _, arq, _ in itens:
            arq.unlink(missing_ok=True)
            arq.with_suffix(".png").unlink(missing_ok=True)
        shutil.rmtree(stg, ignore_errors=True)
        dt = time.perf_counter() - t0
        self.log(f"HF: lote de {len(itens)} enviado ({tam / 2**30:.2f} GiB, {dt:.0f} s, "
                 f"{tam / 2**20 / max(dt, 1e-9):.0f} MB/s) [MEDIDO]; total no repo ~{len(self.meta.todos)}")

    def _trabalhador(self) -> None:
        while True:
            itens = self.fila.get()
            try:
                if itens is None:
                    return
                if not self.falhou.is_set():
                    self.sobe_lote(itens)
            except BaseException as e:  # noqa: BLE001 -- registra e para o gerador; nao morre calado
                self.log(f"FALHOU envio: {type(e).__name__}: {str(e)[:500]}")
                self.falhou.set()
            finally:
                self.fila.task_done()

    def confere(self) -> None:
        if self.falhou.is_set():
            raise SystemExit("FALHOU: o envio de um lote falhou; gerador parado (arquivos locais mantidos)")

    def roda(self, qk) -> None:
        a = types.SimpleNamespace(professor=self.cfg["professor"], passos=self.passos, size=self.size, sementes=[1])
        raiz = Path(self.cfg["raiz"])
        dev = torch.device("cuda:0")
        for lst in self.cfg["listas"]:
            prompts = qk.le_prompts(Q / lst["arquivo"])
            for s in lst["sementes"]:
                self.confere()
                self.meta.rele()
                destino = BASE / f"{lst['nome']}_s{s}"
                if destino.is_dir():
                    n = self.sobras(destino, lst["nome"])
                    if n:
                        self.log(f"{n} shards locais de uma execucao anterior entram no envio")
                faltam = [p for p in prompts if not self.meta.feito(self.chave(p, s))]
                self.log(f"lista {lst['nome']} semente {s}: {len(prompts)} prompts, {len(faltam)} a gravar")
                if not faltam:
                    continue
                a.sementes = [s]

                def ao_gravar(_i, p, sem, arq, estat, _lst=lst["nome"]):
                    self.enfileira(arq, p, sem, estat, _lst)
                    self.confere()

                qk.grava_professor(a, raiz, faltam, destino, dev, imagens=True, ao_gravar=ao_gravar, estat=True)
        if self.lote:
            self.fila.put(list(self.lote))
            self.lote.clear()
        self.fila.put(None)
        self.fila.join()
        self.confere()
        self.log(f"dataset pronto: {len(self.meta.todos)} amostras em {self.cfg['repo']}")


def main() -> None:
    sys.path.insert(0, str(Q))
    import qat_ternario_klein as qk
    from huggingface_hub import HfApi, hf_hub_download
    from qat_klein.status import Status
    BASE.mkdir(parents=True, exist_ok=True)
    status = Status(BASE, "gera_dataset_klein")
    try:
        cfg = json.loads((Q / "ds_config.json").read_text())
        api = HfApi()
        api.create_repo(cfg["repo"], repo_type="dataset", private=not cfg.get("publico", True), exist_ok=True)
        vm = cfg.get("vm_id") or socket.gethostname()
        meta = Metadados(api, cfg["repo"], vm, hf_hub_download)
        meta.rele()
        qk.log(f"dataset {cfg['repo']}: {len(meta.todos)} amostras no indice; esta VM grava {meta.meu_arquivo}")
        Gerador(cfg, api, meta, qk.log).roda(qk)
    except BaseException as e:
        status.fim("FALHOU", 1, f"{type(e).__name__}: {e}")
        raise
    status.fim("FIM", 0)
    print(f"FIM {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
