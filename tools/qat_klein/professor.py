"""Shards do professor: gravação, índice por CONTEÚDO, leitura, cache no HF e condição cruzada.

Um shard guarda, para (prompt, semente), a trajetória do professor: {"prompt", "semente", "comum", "passos":
[{"kw", "out"}...]} (+ "passos_n", "size" nos novos). Identidade:

    novo     `<sha1(prompt)[:16]>_s<semente>_n<passos>_<size>.pt`  -- a MESMA chave do cache no HF
    legado   `p<indice:04d>_s<semente>.pt`  -- o índice na lista de prompts de quem gravou

O nome legado não diz qual prompt está dentro: com outra lista (ou o subconjunto `faltam` do gerador do
dataset) o índice aponta para outro prompt, e o antigo `pula se o arquivo existe` pulava o prompt certo sem
aviso. Agora a decisão de gravar/baixar/subir é pelo PROMPT de dentro do shard (lido com mmap, sem os
tensores; cacheado em `indice_shards.json`). Shards legados continuam lidos e continuam no treino (a
corrida retomada vê o mesmo conjunto e a mesma ordem).
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import torch

from .util import grava_json_atomico, log, sha_texto

NOME_LEGADO = re.compile(r"^p(?P<i>\d{4,})_s(?P<s>\d+)$")
NOME_CHAVE = re.compile(r"^(?P<h>[0-9a-f]{16})_s(?P<s>\d+)_n(?P<n>\d+)_(?P<z>\d+)$")
INDICE = "indice_shards.json"


def chave(prompt: str, semente: int, passos: int, size: int) -> str:
    return f"{sha_texto(prompt)}_s{semente}_n{passos}_{size}"


def semente_do_nome(nome: str) -> int | None:
    stem = nome.removesuffix(".pt")
    m = NOME_LEGADO.match(stem) or NOME_CHAVE.match(stem)
    return int(m.group("s")) if m else None


def lista_shards(d: Path, sementes=None) -> list[Path]:
    """Shards da pasta (legados primeiro, em ordem de nome -- a ordem de sempre --, depois os por chave),
    só das `sementes` pedidas."""
    d = Path(d)
    if not d.is_dir():
        return []
    leg, novos = [], []
    for a in d.glob("*.pt"):
        s = semente_do_nome(a.name)
        if s is None or (sementes is not None and s not in set(sementes)):
            continue
        (leg if NOME_LEGADO.match(a.stem) else novos).append(a)
    return sorted(leg) + sorted(novos)


def valida_shard(sh: dict, arq, prompt=None, semente=None, passos=None, size=None) -> None:
    """Recusa (ValueError) shard cujo conteúdo não é o pedido. `size` só é conferido se o shard o grava."""
    erros = []
    if prompt is not None and sh.get("prompt") != prompt:
        erros.append(f"prompt {sh.get('prompt')!r:.60} != {prompt!r:.60}")
    if semente is not None and sh.get("semente") != semente:
        erros.append(f"semente {sh.get('semente')} != {semente}")
    n = sh.get("passos_n", len(sh.get("passos", ())) or None)
    if passos is not None and n != passos:
        erros.append(f"{n} passos != {passos}")
    if size is not None and sh.get("size") is not None and sh["size"] != size:
        erros.append(f"size {sh['size']} != {size}")
    if erros:
        raise ValueError(f"shard {Path(arq).name}: " + "; ".join(erros))


def le_cabecalho(arq: Path) -> dict:
    """prompt/semente/passos_n/size de um shard sem ler os tensores (mmap)."""
    sh = torch.load(arq, map_location="cpu", weights_only=False, mmap=True)
    return {"prompt_sha": sha_texto(sh["prompt"]), "prompt": sh["prompt"], "semente": sh["semente"],
            "passos_n": sh.get("passos_n", len(sh["passos"])), "size": sh.get("size")}


class Indice:
    """(sha do prompt, semente) -> arquivo, para os shards de uma pasta, legados incluídos.

    Nomes por chave dão tudo pelo nome; legados são abertos uma vez (mmap) e cacheados por (nome, tamanho,
    mtime) em `indice_shards.json` -- a pasta pode ser um symlink compartilhado entre braços; a gravação do
    cache é atômica e o conteúdo é o mesmo para quem gravar por último."""

    def __init__(self, d: Path, sementes=None):
        self.d = Path(d)
        self.arqs = lista_shards(self.d, sementes)
        self.cab: dict[str, dict] = {}
        cache_arq = self.d / INDICE
        cache = {}
        if cache_arq.is_file():
            try:
                import json
                cache = json.loads(cache_arq.read_text(encoding="utf-8"))
            except ValueError:
                cache = {}
        mudou = False
        for a in self.arqs:
            m = NOME_CHAVE.match(a.stem)
            if m:
                self.cab[a.name] = {"prompt_sha": m.group("h"), "semente": int(m.group("s")),
                                    "passos_n": int(m.group("n")), "size": int(m.group("z"))}
                continue
            st = a.stat()
            marca = [st.st_size, st.st_mtime_ns]
            c = cache.get(a.name)
            if not c or c.get("marca") != marca:
                c = {k: v for k, v in le_cabecalho(a).items() if k != "prompt"}
                c["marca"] = marca
                cache[a.name] = c
                mudou = True
            self.cab[a.name] = c
        if mudou:
            try:
                grava_json_atomico(cache_arq, cache)
            except OSError as e:  # pasta só de leitura: o índice vale só nesta execução
                log(f"indice de shards nao gravado em {self.d}: {type(e).__name__}")
        self.por_chave: dict[tuple[str, int], Path] = {}
        for a in self.arqs:
            c = self.cab[a.name]
            self.por_chave.setdefault((c["prompt_sha"], c["semente"]), a)

    def tem(self, prompt: str, semente: int) -> Path | None:
        return self.por_chave.get((sha_texto(prompt), semente))

    def prompts_sha(self) -> set[str]:
        return {c["prompt_sha"] for c in self.cab.values()}

    def confere_legados(self, prompts: list[str]) -> list[str]:
        """Avisos para shards legados cujo prompt de dentro não é o prompts[indice] da lista atual."""
        avisos = []
        for a in self.arqs:
            m = NOME_LEGADO.match(a.stem)
            if not m:
                continue
            i = int(m.group("i"))
            if i < len(prompts) and sha_texto(prompts[i]) != self.cab[a.name]["prompt_sha"]:
                avisos.append(a.name)
        return avisos


def _ganchos_estat(tr, por_passo: list) -> list:
    """Pre-hooks em todo nn.Linear do transformer: a cada chamada (passo) anota, por camada, mínimo, média,
    máximo e mediana da ENTRADA. Mediana de uma amostra fixa de ~1M valores (a exata custaria mais que o
    próprio passo); mín/média/máx exatos. Os valores ficam em tensores até o fim do shard (sem .item())."""
    alcas = []

    def fab(nome):
        def gancho(_m, args):
            x = args[0].detach()
            v = x.reshape(-1)
            passo_ = max(1, v.numel() // 1_000_000)
            amostra = v[::passo_].float()
            por_passo[-1][nome] = torch.stack([v.min().float(), v.float().mean(), v.max().float(),
                                               amostra.median()])
        return gancho

    for nome, m in tr.named_modules():
        if isinstance(m, torch.nn.Linear):
            alcas.append(m.register_forward_pre_hook(fab(nome)))
    return alcas


def faltantes(a, prompts: list[str], destino: Path) -> list[tuple[int, str, int]]:
    """(índice, prompt, semente) sem shard VÁLIDO na pasta (por conteúdo, não por índice)."""
    idx = Indice(destino, a.sementes)
    ruins = idx.confere_legados(prompts)
    if ruins:
        log(f"AVISO: {len(ruins)} shards legados em {Path(destino).name} guardam prompt diferente do da lista "
            f"atual no mesmo indice (p.ex. {ruins[:3]}); os prompts da lista serao gravados por chave de conteudo")
    return [(i, p, s) for i, p in enumerate(prompts) for s in a.sementes if idx.tem(p, s) is None]


def grava_professor(a, raiz: Path, prompts: list[str], destino: Path, dev, imagens: bool = False,
                    ao_gravar=None, estat: bool = False) -> None:
    """Um shard por (prompt, semente), nome por chave de conteúdo: {"comum": kw que não muda entre passos,
    "passos": [...]}.

    `imagens=True` também decodifica a imagem final e grava `<chave>.png` ao lado (o transformer é
    chamado igual; só o VAE roda a mais). `ao_gravar(i, prompt, semente, arq_pt, estat)` é chamado depois de
    cada shard (o gerador do dataset usa para subir em lotes).

    O `encoder_hidden_states` do klein é um tap de 3 camadas do Qwen3 (7680 de largura) e é o MESMO
    tensor nos 8 passos de uma chamada: guardar uma vez por chamada corta ~7x o disco."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    faltam = faltantes(a, prompts, destino)
    if not faltam:
        log(f"professor: {destino.name} completo, nada a gravar")
        return
    from ajusta_denso_diffusers import carrega_transformer, classe_do_pipeline
    nome, Pipe = classe_do_pipeline(raiz)
    tr = carrega_transformer(Path(a.professor), raiz / "transformer" / "config.json",
                             torch.bfloat16, torch.device("cpu")).to(dev)
    pipe = Pipe.from_pretrained(str(raiz), transformer=tr, torch_dtype=torch.bfloat16)
    pipe.to(dev)
    log(f"professor {nome}: {len(faltam)} shards a gravar em {destino}")
    orig = tr.forward
    atual: list = []

    por_passo: list = []
    alcas = _ganchos_estat(tr, por_passo) if estat else []

    def espia(*args, **kw):
        por_passo.append({})
        out = orig(*args, **kw)
        saida = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        atual.append(({k: v for k, v in kw.items()}, saida.detach().to("cpu").clone()))
        return out

    tr.forward = espia
    t0 = time.perf_counter()
    try:
        for j, (i, p, s) in enumerate(faltam):
            atual.clear()
            por_passo.clear()
            with torch.no_grad():
                res = pipe(prompt=p, height=a.size, width=a.size, num_inference_steps=a.passos,
                           guidance_scale=1.0, generator=torch.Generator(device="cpu").manual_seed(s),
                           output_type="pil" if imagens else "latent")
            kw0 = atual[0][0]
            comum = {k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
                     for k, v in kw0.items()
                     if torch.is_tensor(v) and all(torch.is_tensor(kw.get(k))
                                                   and kw[k].data_ptr() == v.data_ptr()
                                                   for kw, _ in atual)}
            passos = []
            for kw, out in atual:
                varia = {k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
                         for k, v in kw.items() if k not in comum}
                passos.append({"kw": varia, "out": out})
            arq = destino / f"{chave(p, s, a.passos, a.size)}.pt"
            tmp = arq.with_suffix(".pt.partial")
            extra = {}
            if estat:
                # [passo][camada] -> [minimo, medio, maximo, mediana] da entrada da camada
                extra["estat"] = [{k: [round(float(x), 6) for x in v.cpu()] for k, v in d.items()} for d in por_passo]
            torch.save({"prompt": p, "semente": s, "passos_n": len(passos), "size": a.size, "comum": comum,
                        "passos": passos, **extra}, tmp)
            os.replace(tmp, arq)
            if imagens:
                res.images[0].save(arq.with_suffix(".png"))
            if ao_gravar is not None:
                ao_gravar(i, p, s, arq, extra.get("estat"))
            if j == 0 or (j + 1) % 16 == 0 or j == len(faltam) - 1:
                log(f"  professor {j + 1}/{len(faltam)}  {len(atual)} chamadas  "
                    f"{(time.perf_counter() - t0) / (j + 1):.1f} s/shard")
    finally:
        tr.forward = orig
        for h in alcas:
            h.remove()
    pipe.to("cpu")
    del pipe, tr
    torch.cuda.empty_cache()


class Exemplos:
    """(comum, kw_do_passo, alvo) lidos do DISCO por demanda, com um cache pequeno de shards.

    A primeira versão carregava tudo na RAM: com 124 prompts eram ~6 GB e cabia; com o PartiPrompts
    (~1.750 prompts) seriam ~42 GB. Um shard tem ~24 MB e `torch.load` dele custa décimos de segundo contra
    segundos de passo. Só entram os shards das `sementes` pedidas (legados e por chave)."""

    def __init__(self, d: Path, sementes, passos: int, cache: int = 64, size: int | None = None):
        self.arqs = lista_shards(d, sementes)
        self.passos = passos
        self.size = size
        self._le = lru_cache(maxsize=cache)(
            lambda a: torch.load(a, map_location="cpu", weights_only=False))

    def __len__(self):
        return len(self.arqs) * self.passos

    def __getitem__(self, i):
        arq = self.arqs[i // self.passos]
        sh = self._le(arq)
        try:
            valida_shard(sh, arq, passos=self.passos, size=self.size)
        except ValueError as e:
            raise SystemExit(f"RECUSADO: {e}") from None
        ps = sh["passos"][i % self.passos]
        return sh["comum"], ps["kw"], ps["out"]


def recusa_vazamento(treino_dir: Path, holdout_dir: Path, sementes) -> None:
    """Treino e holdout não podem ter PROMPT em comum nos shards de fato usados (a trava por arquivo de
    prompts não via shards compartilhados por symlink ou gravados por outra lista)."""
    comum = Indice(treino_dir, sementes).prompts_sha() & Indice(holdout_dir, sementes).prompts_sha()
    if comum:
        raise SystemExit(f"RECUSADO: {len(comum)} prompts nos shards de treino E de holdout (sha1 {sorted(comum)[:2]})")


# ------------------------------------------------------------------------- condicao cruzada (fase 2)
# O aluno so' via o professor NA TRAJETORIA DELE: x_t do prompt i sempre com a condicao do prompt i.
# A condicao cruzada grava, para o MESMO (x_t, t) do shard i, a saida do professor com a condicao do shard
# vizinho j (mesma semente, indice seguinte na lista de shards, circular -- o mesmo par da guarda `sens`).
# Arquivo separado, com o MESMO nome do shard i: os shards originais nao se reescrevem.

def pares_vizinhos(arqs: list[Path]) -> dict[Path, Path]:
    por_semente: dict[int, list[Path]] = {}
    for arq in arqs:
        por_semente.setdefault(semente_do_nome(arq.name), []).append(arq)
    return {L[i]: L[(i + 1) % len(L)] for L in por_semente.values() if len(L) >= 2 for i in range(len(L))}


def forward_aluno(aluno, comum, kw, dev, contexto):
    todos = {**comum, **kw}
    todos = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in todos.items()}
    todos["return_dict"] = False
    with contexto:
        out = aluno(**todos)
    return out[0] if isinstance(out, tuple) else out


def grava_cruzado(a, raiz: Path, origem: Path, destino: Path, dev, carrega=None) -> None:
    arqs = lista_shards(origem, a.sementes)
    par = pares_vizinhos(arqs)
    destino.mkdir(parents=True, exist_ok=True)
    faltam = [x for x in arqs if x in par and not (destino / x.name).is_file()]
    if not faltam:
        log(f"cruzado: {destino.name} completo")
        return
    from contextlib import nullcontext
    if carrega is None:
        from ajusta_denso_diffusers import carrega_transformer
        tr = carrega_transformer(Path(a.professor), raiz / "transformer" / "config.json",
                                 torch.bfloat16, torch.device("cpu")).to(dev).eval()
    else:
        tr = carrega().to(dev).eval()
    log(f"cruzado: {len(faltam)} shards em {destino}")
    t0 = time.perf_counter()
    for n, x in enumerate(faltam):
        shi = torch.load(x, map_location="cpu", weights_only=False)
        shj = torch.load(par[x], map_location="cpu", weights_only=False)
        with torch.no_grad():
            outs = [forward_aluno(tr, shj["comum"], ps["kw"], dev, nullcontext()).detach().to("cpu").clone()
                    for ps in shi["passos"]]
        tmp = destino / (x.name + ".partial")
        torch.save({"par": par[x].name, "outs": outs}, tmp)
        os.replace(tmp, destino / x.name)
        if n == 0 or (n + 1) % 64 == 0 or n == len(faltam) - 1:
            log(f"  cruzado {n + 1}/{len(faltam)}  {(time.perf_counter() - t0) / (n + 1):.2f} s/shard")
    tr.to("cpu")
    del tr
    torch.cuda.empty_cache()


class ExemplosCruzados:
    """(comum DO VIZINHO, kw do passo do shard i, saida cruzada do professor). So' shards com arquivo cruzado."""

    def __init__(self, base: Exemplos, d: Path, passos_sel: list[int] | None = None):
        self.base, self.d = base, d
        self.arqs = [x for x in base.arqs if (d / x.name).is_file()]
        self.passos = list(passos_sel) if passos_sel is not None else list(range(base.passos))
        self._le = lru_cache(maxsize=64)(lambda p: torch.load(p, map_location="cpu", weights_only=False))

    def __len__(self):
        return len(self.arqs) * len(self.passos)

    def __getitem__(self, i):
        x = self.arqs[i // len(self.passos)]
        pj = self.passos[i % len(self.passos)]
        cz = self._le(self.d / x.name)
        shi = self.base._le(x)
        shj = self.base._le(x.parent / cz["par"])
        return shj["comum"], shi["passos"][pj]["kw"], cz["outs"][pj]


def junta_lote(itens: list[tuple[dict, dict, torch.Tensor]]) -> tuple[dict, torch.Tensor]:
    """N exemplos -> um lote. Tensor com dimensão 0 == 1 é concatenado no eixo 0 (é o eixo de lote:
    hidden_states, encoder_hidden_states, timestep). Tensor sem eixo de lote (ids posicionais) TEM de
    ser idêntico entre os exemplos -- a 1024 px e com o texto preenchido até 512 tokens ele é; se não
    for, recusa em vez de misturar posições de exemplos diferentes."""
    dicts = [{**c, **k} for c, k, _ in itens]
    lote = {}
    for nome, v0 in dicts[0].items():
        vs = [d[nome] for d in dicts]
        if not torch.is_tensor(v0) or len(vs) == 1:
            lote[nome] = v0
        elif v0.ndim >= 1 and v0.shape[0] == 1:
            lote[nome] = torch.cat(vs, dim=0)
        elif all(v.shape == v0.shape and torch.equal(v, v0) for v in vs[1:]):
            lote[nome] = v0
        else:
            raise SystemExit(f"RECUSADO: '{nome}' {tuple(v0.shape)} nao tem eixo de lote e difere "
                             f"entre exemplos; lote > 1 misturaria posicoes")
    return lote, torch.cat([t for _, _, t in itens], dim=0)


# ----------------------------------------------------------------------------- cache no HF (--professor-hf)

def cache_professor(a, prompts: list[str], destino: Path, sentido: str, api=None) -> None:
    """Reaproveita shards do professor entre VMs via um repo de DADOS no HF (`--professor-hf`).

    Chave no repo = a do nome local: `professor/<sha1[:16]>_s<sem>_n<passos>_<size>.pt`. `baixa` traz os
    (prompt, semente) da lista que não têm shard válido local; `sobe` envia os shards locais da lista cuja
    chave -- calculada do PROMPT DE DENTRO do shard, não do índice -- o repo ainda não tem."""
    if not a.professor_hf:
        return
    import shutil
    import tempfile
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    repo = a.professor_hf
    api.create_repo(repo, repo_type="dataset", private=not a.hf_publico, exist_ok=True)
    remotos = {f.split("/", 1)[1] for f in api.list_repo_files(repo, repo_type="dataset")
               if f.startswith("professor/")}
    destino.mkdir(parents=True, exist_ok=True)
    idx = Indice(destino, a.sementes)
    t0 = time.perf_counter()
    if sentido == "baixa":
        falta = {chave(p, s, a.passos, a.size) + ".pt" for p in prompts for s in a.sementes
                 if idx.tem(p, s) is None}
        falta &= remotos
        if not falta:
            log(f"professor-hf: nada a baixar para {destino.name} ({len(remotos)} no repo)")
            return
        from huggingface_hub import snapshot_download
        tmp = Path(tempfile.mkdtemp(dir=destino.parent))
        snapshot_download(repo, repo_type="dataset", local_dir=str(tmp), max_workers=16,
                          allow_patterns=[f"professor/{c}" for c in sorted(falta)])
        for c in falta:
            os.replace(tmp / "professor" / c, destino / c)
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"professor-hf: {len(falta)} shards baixados para {destino.name} em {time.perf_counter() - t0:.0f} s")
        return
    novos = {}
    for p in prompts:
        for s in a.sementes:
            arq = idx.tem(p, s)
            if arq is None:
                continue
            c = idx.cab[arq.name]
            nome = f"{c['prompt_sha']}_s{c['semente']}_n{c['passos_n']}_{c.get('size') or a.size}.pt"
            if nome not in remotos:
                novos[nome] = arq
    if not novos:
        return
    tmp = Path(tempfile.mkdtemp(dir=destino.parent))
    for nome, arq in novos.items():
        os.link(arq, tmp / nome)
    tam = sum((tmp / c).stat().st_size for c in novos)
    try:
        api.upload_folder(folder_path=str(tmp), path_in_repo="professor", repo_id=repo,
                          repo_type="dataset", commit_message=f"{len(novos)} shards do professor")
        log(f"professor-hf: {len(novos)} shards enviados ({tam / 2**30:.2f} GiB, "
            f"{time.perf_counter() - t0:.0f} s) [MEDIDO]")
    except Exception as e:  # noqa: BLE001 -- cache falho nao derruba o treino
        log(f"professor-hf: envio NAO completou ({type(e).__name__}: {str(e)[:500]})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
