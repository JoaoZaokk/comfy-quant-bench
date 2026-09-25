"""QAT ternário do FLUX.2-klein por destilação contra o próprio BF16. Retomável, para rodar em casa ou no Colab.

DE ONDE VEM. O braço 1 (`ajusta_denso_diffusers.py`) congelou o corpo ternário e ajustou só 9 tensores
densos: fechou 54% da distância PTQ→Bonsai no epsilon fora da amostra e **não gerou imagem
utilizável** (`bench/render_braco1_resultado_2026-09-22.md`). O Bonsai trocou 2,7 M de sinais no
corpo — treinou os códigos. Este tool deixa o corpo treinar:

    professor   klein BF16, congelado; grava (latente ruidoso, t, condição) -> saída, na trajetória DELE
    aluno       inicializado do MESMO BF16 (não do ternário: o valor contínuo diz quem está perto
                do limiar); no forward cada Linear do corpo vira ternário por STE; o denso fica em
                precisão normal, como no braço 1
    perda       MSE em fp32 entre as duas saídas
    um backprop -> o mestre anda (contínuo) -> quando cruza o limiar, o código troca

TERNÁRIO = o mesmo do braço 0 e do Bonsai: absmean por grupo de 128 no eixo K, código
`round(clamp(w/d, -1, 1))`, escala ótima L2 por grupo dado o código (`constroi_ternario_ingenuo.py`).
A escala é RECALCULADA a cada forward a partir do mestre ("escala fixa" no sentido de não ser
parâmetro); escala aprendida é outro eixo, não este.

STE EXATO. `w + (q(w) - w).detach()` em bf16 NÃO dá exatamente q(w) — dois arredondamentos — e o
forward deixaria de ser ternário. Usa-se uma `autograd.Function` que devolve q(w) no forward e passa
o gradiente inteiro no backward.

PESO MESTRE, o eixo que já custou um braço: bf16 puro com lr 1e-5 deixou 63,4% dos elementos do braço
1 sem mudar um bit. Opções (`--otim`):
    adamw8bit-sr   peso bf16 + AdamW8bit do torchao com arredondamento estocástico   ~6 B/param
    adamw4bit-sr   peso bf16 + AdamW4bit com SR -- o 8-bit NAO coube na 3090: pico 23,47 GiB e o WDDM
                   paginou (20 s/passo, 190-230 W), smoke de 2026-09-22 22:08                  ~5 B/param
    adamw8bit      peso fp32 + AdamW8bit                                            ~10 B/param
    adam-fp32      peso fp32 + Adam do torch                                        ~16 B/param
Para 3.875.544.576 params: 21,7 / 36,1 / 57,7 GiB SÓ de peso+grad+estado, fora ativação.

RETOMÁVEL, porque o recurso some sem aviso (memória `intervalo-de-gravacao-e-a-perda-maxima`):
  * o professor grava UM shard por (prompt, semente) e pula os que já existem;
  * o aluno grava checkpoint atômico (`.partial` + `os.replace`) a cada `--ckpt-min` minutos e retoma
    do último, com o estado do otimizador. O intervalo É a perda máxima: não subir sem motivo.
  * `journal.jsonl` recebe uma linha por log; o log termina com `FIM HH:MM:SS` (contrato do
    `probe_background_job.py` do supervisor do Colab).

MÉTRICAS que o critério pede, além da perda: perda num conjunto HOLDOUT de prompts que o treino não
vê; fração de códigos ternários diferentes do código inicial; fração que trocou desde o último log
(estabilidade).

NÃO COBRE: nenhuma imagem e nenhum epsilon no protocolo do ComfyUI — isso é feito depois, fora, com
`aplica_mapa_diffusers_bfl.py` + `probe_epsilon_ckpt_ab.py` + `quality_ladder.py`. Nenhum tempo de
1,58 bit: o corpo roda desempacotado em bf16.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ ternário

def pilhas_reais(nomes) -> set[str]:
    ind: dict[str, set[str]] = {}
    for k in nomes:
        m = BLOCO.match(k)
        if m:
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


# Níveis do código: 1 = ternário (absmean, o do braço 0 e do Bonsai); 7 = INT4 simétrico (absmax/7,
# o mesmo RTN do controle C1 de 25/09). Definido em main() por --formato; a escala é sempre a ótima L2
# por grupo dado o código.
NIVEIS = 1


def codigo_e_escala(w: torch.Tensor, grupo: int):
    """(código int8 [n, k], escala fp32 [n, k//grupo, 1]). Ternário: mesma receita do braço 0."""
    n, k = w.shape
    g = w.float().reshape(n, k // grupo, grupo)
    if NIVEIS == 1:
        d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
    else:
        d = (g.abs().amax(dim=2, keepdim=True) / NIVEIS).clamp(min=1e-30)
    t = (g / d).clamp(-NIVEIS, NIVEIS).round()
    num = (g * t).sum(dim=2, keepdim=True)
    den = (t * t).sum(dim=2, keepdim=True)
    s = torch.where(den > 0, num / den, torch.zeros_like(num))
    return t.reshape(n, k).to(torch.int8), s


def ternariza(w: torch.Tensor, grupo: int) -> torch.Tensor:
    t, s = codigo_e_escala(w, grupo)
    n, k = w.shape
    return (t.float().reshape(n, k // grupo, grupo) * s).reshape(n, k).to(w.dtype)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, grupo):
        return ternariza(w, grupo)

    @staticmethod
    def backward(ctx, g):
        return g, None


# Desligado SO para medir a referencia da guarda de condicionamento com o mestre ainda igual ao
# BF16 (antes do primeiro passo). Nunca durante o treino.
STE_LIGADO = True


def instala_ste(modelo: torch.nn.Module, corpo: set[str], grupo: int, dtype_calc) -> int:
    """Troca o forward de cada nn.Linear do corpo por F.linear(x, STE(w)). Devolve quantos."""
    n = 0
    for nome, m in modelo.named_modules():
        if isinstance(m, torch.nn.Linear) and f"{nome}.weight" in corpo:
            def fwd(x, _m=m):
                w = _STE.apply(_m.weight, grupo) if STE_LIGADO else _m.weight
                w = w.to(dtype_calc)
                b = None if _m.bias is None else _m.bias.to(dtype_calc)
                return F.linear(x.to(dtype_calc), w, b)
            m.forward = fwd
            n += 1
    return n


def instala_escalas(modelo: torch.nn.Module, corpo: set[str], grupo: int, dtype_calc) -> int:
    """Modo `--so-escalas`: congela o código de cada Linear do corpo no arredondamento do BF16 e treina SÓ
    uma escala por grupo (parâmetro `_esc`, inicializado na escala ótima L2). O peso original fica
    congelado e só é usado com STE_LIGADO=False (referência BF16 do sens). Refino pós-quantização: a grade
    não se move, só o tamanho do degrau de cada grupo."""
    n = 0
    for nome, m in modelo.named_modules():
        if isinstance(m, torch.nn.Linear) and f"{nome}.weight" in corpo:
            with torch.no_grad():
                cod, esc = codigo_e_escala(m.weight, grupo)
            m.register_buffer("_cod", cod, persistent=False)
            m._esc = torch.nn.Parameter(esc.float())
            m.weight.requires_grad_(False)

            def fwd(x, _m=m):
                if STE_LIGADO:
                    nn_, kk = _m._cod.shape
                    w = (_m._cod.float().view(nn_, kk // grupo, grupo) * _m._esc).view(nn_, kk)
                else:
                    w = _m.weight
                w = w.to(dtype_calc)
                b = None if _m.bias is None else _m.bias.to(dtype_calc)
                return F.linear(x.to(dtype_calc), w, b)
            m.forward = fwd
            n += 1
    return n


# ----------------------------------------------------------------------------------------- professor

def le_prompts(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def grava_professor(a, raiz: Path, prompts: list[str], destino: Path, dev, imagens: bool = False,
                    ao_gravar=None) -> None:
    """Um shard por (prompt, semente): {"comum": kw que não muda entre passos, "passos": [...]}.

    `imagens=True` também decodifica a imagem final e grava `p####_s#.png` ao lado (o transformer é
    chamado igual; só o VAE roda a mais). `ao_gravar(i, prompt, semente, arq_pt)` é chamado depois de
    cada shard (o gerador do dataset usa para subir em lotes).

    O `encoder_hidden_states` do klein é um tap de 3 camadas do Qwen3 (7680 de largura) e é o MESMO
    tensor nos 8 passos de uma chamada: guardar uma vez por chamada corta ~7x o disco."""
    destino.mkdir(parents=True, exist_ok=True)
    faltam = [(i, p, s) for i, p in enumerate(prompts) for s in a.sementes
              if not (destino / f"p{i:04d}_s{s}.pt").is_file()]
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

    def espia(*args, **kw):
        out = orig(*args, **kw)
        saida = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        atual.append(({k: v for k, v in kw.items()}, saida.detach().to("cpu").clone()))
        return out

    tr.forward = espia
    t0 = time.perf_counter()
    for j, (i, p, s) in enumerate(faltam):
        atual.clear()
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
        arq = destino / f"p{i:04d}_s{s}.pt"
        tmp = arq.with_suffix(".pt.partial")
        torch.save({"prompt": p, "semente": s, "comum": comum, "passos": passos}, tmp)
        os.replace(tmp, arq)
        if imagens:
            res.images[0].save(arq.with_suffix(".png"))
        if ao_gravar is not None:
            ao_gravar(i, p, s, arq)
        if j == 0 or (j + 1) % 16 == 0 or j == len(faltam) - 1:
            log(f"  professor {j + 1}/{len(faltam)}  {len(atual)} chamadas  "
                f"{(time.perf_counter() - t0) / (j + 1):.1f} s/shard")
    tr.forward = orig
    pipe.to("cpu")
    del pipe, tr
    torch.cuda.empty_cache()


class Exemplos:
    """(comum, kw_do_passo, alvo) lidos do DISCO por demanda, com um cache pequeno de shards.

    A primeira versão carregava tudo na RAM: com 124 prompts eram ~6 GB e cabia; com o PartiPrompts
    (~1.750 prompts) seriam ~42 GB, que não cabem ao lado do resto nem aqui nem com folga no Colab.
    Um shard tem ~24 MB e `torch.load` dele custa décimos de segundo contra segundos de passo.
    Só entram no índice os shards das `sementes` pedidas, para uma corrida com menos sementes poder
    reaproveitar a pasta de outra sem misturar."""

    def __init__(self, d: Path, sementes, passos: int, cache: int = 64):
        from functools import lru_cache
        self.arqs = sorted(a for a in d.glob("p*_s*.pt")
                           if int(a.stem.rsplit("_s", 1)[1]) in set(sementes))
        self.passos = passos
        self._le = lru_cache(maxsize=cache)(
            lambda a: torch.load(a, map_location="cpu", weights_only=False))

    def __len__(self):
        return len(self.arqs) * self.passos

    def __getitem__(self, i):
        sh = self._le(self.arqs[i // self.passos])
        if len(sh["passos"]) != self.passos:
            raise SystemExit(f"RECUSADO: {self.arqs[i // self.passos].name} tem "
                             f"{len(sh['passos'])} passos, esperava {self.passos}")
        ps = sh["passos"][i % self.passos]
        return sh["comum"], ps["kw"], ps["out"]


# ------------------------------------------------------------------------- condicao cruzada (fase 2)
# O aluno so' via o professor NA TRAJETORIA DELE: x_t do prompt i sempre com a condicao do prompt i.
# Nada ensinava "se a condicao mudar, a saida muda assim" -- e o colapso medido e' exatamente o aluno
# deixar de responder ao prompt (sens 0,54-0,62 contra 1,0 do BF16). A condicao cruzada grava, para o
# MESMO (x_t, t) do shard i, a saida do professor com a condicao do shard vizinho j (mesma semente,
# indice seguinte, circular -- o mesmo par da guarda `sens`). Arquivo separado: os shards originais
# podem estar sendo lidos por outra corrida e nao se reescrevem.

def pares_vizinhos(arqs: list[Path]) -> dict[Path, Path]:
    por_semente: dict[int, list[Path]] = {}
    for arq in arqs:
        por_semente.setdefault(int(arq.stem.rsplit("_s", 1)[1]), []).append(arq)
    return {L[i]: L[(i + 1) % len(L)] for L in por_semente.values() if len(L) >= 2 for i in range(len(L))}


def grava_cruzado(a, raiz: Path, origem: Path, destino: Path, dev) -> None:
    arqs = sorted(x for x in origem.glob("p*_s*.pt") if int(x.stem.rsplit("_s", 1)[1]) in set(a.sementes))
    par = pares_vizinhos(arqs)
    destino.mkdir(parents=True, exist_ok=True)
    faltam = [x for x in arqs if x in par and not (destino / x.name).is_file()]
    if not faltam:
        log(f"cruzado: {destino.name} completo")
        return
    from contextlib import nullcontext
    from ajusta_denso_diffusers import carrega_transformer
    tr = carrega_transformer(Path(a.professor), raiz / "transformer" / "config.json",
                             torch.bfloat16, torch.device("cpu")).to(dev).eval()
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
        self.passos = passos_sel if passos_sel is not None else list(range(base.passos))
        from functools import lru_cache
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


# --------------------------------------------------------------------------------------------- aluno

def junta_lote(itens: list[tuple[dict, dict, torch.Tensor]]) -> tuple[dict, torch.Tensor]:
    """N exemplos -> um lote. Tensor com dimensão 0 == 1 é concatenado no eixo 0 (é o eixo de lote:
    hidden_states, encoder_hidden_states, timestep). Tensor sem eixo de lote (ids posicionais) TEM de
    ser idêntico entre os exemplos -- a 1024 px e com o texto preenchido até 512 tokens ele é; se não
    for, recusa em vez de misturar posições de exemplos diferentes."""
    dicts = [{**c, **k} for c, k, _ in itens]
    lote = {}
    for chave, v0 in dicts[0].items():
        vs = [d[chave] for d in dicts]
        if not torch.is_tensor(v0) or len(vs) == 1:
            lote[chave] = v0
        elif v0.ndim >= 1 and v0.shape[0] == 1:
            lote[chave] = torch.cat(vs, dim=0)
        elif all(v.shape == v0.shape and torch.equal(v, v0) for v in vs[1:]):
            lote[chave] = v0
        else:
            raise SystemExit(f"RECUSADO: '{chave}' {tuple(v0.shape)} nao tem eixo de lote e difere "
                             f"entre exemplos; lote > 1 misturaria posicoes")
    return lote, torch.cat([a for _, _, a in itens], dim=0)


def forward_aluno(aluno, comum, kw, dev, contexto):
    todos = {**comum, **kw}
    todos = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in todos.items()}
    todos["return_dict"] = False
    with contexto:
        out = aluno(**todos)
    return out[0] if isinstance(out, tuple) else out


def cria_otimizador(nome: str, params, lr: float):
    """`params`: lista de tensores OU de grupos {"params", "lr"} (lr por grupo: corpo x denso)."""
    if nome == "adamw8bit-sr":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)
    if nome == "adamw4bit-sr":
        from torchao.optim import AdamW4bit
        return AdamW4bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)
    if nome == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0)
    if nome == "adam-fp32":
        return torch.optim.Adam(params, lr=lr)
    raise SystemExit(f"otimizador desconhecido: {nome}")


def _arredonda_sr_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 com arredondamento ESTOCASTICO (soma ruido uniforme nos 16 bits que o bf16 descarta
    e trunca). Sem isso um passo de L1 de ~1e-6 num peso de ~1e-2 some inteiro no arredondamento."""
    i = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, i.shape, device=x.device, dtype=torch.int32)
    return ((i + r) & -65536).view(torch.float32).to(torch.bfloat16)


@torch.no_grad()
def prox_l1_relativo(corpo_params, passo_l1: float, grupo: int) -> None:
    """w <- sign(w) * max(|w| - passo_l1 * d_g, 0), d_g = media|w| do grupo de `grupo` no eixo K.

    POR QUE L1 E NAO WEIGHT DECAY: o codigo ternario depende de r = |w|/d_g. Weight decay multiplica o
    grupo inteiro pelo mesmo fator e r nao muda -- nenhum codigo troca. L1 subtrai uma constante: um peso
    em r = 0,5 vai para (0,5 - delta)/(1 - delta) < 0,5 e e' rebaixado; o gradiente segura os que importam.
    Relativo a d_g para o mesmo lambda valer em camadas de escala diferente. [TRACADO, nao medido ainda]"""
    for w in corpo_params:
        n, k = w.shape
        g = w.float().reshape(n, k // grupo, grupo)
        d = g.abs().mean(dim=2, keepdim=True)
        novo = torch.sign(g) * (g.abs() - passo_l1 * d).clamp(min=0)
        novo = novo.reshape(n, k)
        w.copy_(_arredonda_sr_bf16(novo) if w.dtype == torch.bfloat16 else novo.to(w.dtype))


def estatistica_codigos(aluno, corpo_mods, cod0: dict, cod_ant: dict, grupo: int) -> dict:
    """Fração de códigos != inicial e != última medição. Guarda o código atual em `cod_ant`."""
    tot = dif0 = difa = 0
    with torch.no_grad():
        for nome, m in corpo_mods:
            c, _ = codigo_e_escala(m.weight, grupo)
            c = c.cpu()
            tot += c.numel()
            dif0 += int((c != cod0[nome]).sum())
            if nome in cod_ant:
                difa += int((c != cod_ant[nome]).sum())
            cod_ant[nome] = c
    return {"codigos_mudados_vs_inicio": dif0 / tot, "codigos_trocados_desde_ultimo": difa / tot,
            "codigos_total": tot}


def cache_professor(a, prompts: list[str], destino: Path, sentido: str) -> None:
    """Reaproveita shards do professor entre VMs via um repo de DADOS no HF (`--professor-hf`).

    O shard depende so' do professor (o BF16), do prompt, da semente, de --passos e de --size -- nao do
    aluno nem do formato. No repo a chave e' o CONTEUDO do prompt, nao o indice (o mesmo prompt tem
    indices diferentes em prompts_treino_124.txt e no Parti): `professor/<sha1[:16]>_s<sem>_n<passos>_<size>.pt`.
    `sentido="baixa"` traz o que existir antes de gravar; `"sobe"` envia o que o repo ainda nao tem."""
    if not a.professor_hf:
        return
    import hashlib
    import shutil
    import tempfile
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    repo = a.professor_hf
    api.create_repo(repo, repo_type="dataset", private=not a.hf_publico, exist_ok=True)
    remotos = {f.split("/", 1)[1] for f in api.list_repo_files(repo, repo_type="dataset")
               if f.startswith("professor/")}
    chave = {(i, s): f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]}_s{s}_n{a.passos}_{a.size}.pt"
             for i, p in enumerate(prompts) for s in a.sementes}
    destino.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if sentido == "baixa":
        falta = {k: c for k, c in chave.items()
                 if c in remotos and not (destino / f"p{k[0]:04d}_s{k[1]}.pt").is_file()}
        if not falta:
            log(f"professor-hf: nada a baixar para {destino.name} ({len(remotos)} no repo)")
            return
        tmp = Path(tempfile.mkdtemp(dir=destino.parent))
        snapshot_download(repo, repo_type="dataset", local_dir=str(tmp), max_workers=16,
                          allow_patterns=[f"professor/{c}" for c in falta.values()])
        for (i, s), c in falta.items():
            os.replace(tmp / "professor" / c, destino / f"p{i:04d}_s{s}.pt")
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"professor-hf: {len(falta)} shards baixados para {destino.name} em {time.perf_counter() - t0:.0f} s")
    else:
        novos = {k: c for k, c in chave.items()
                 if c not in remotos and (destino / f"p{k[0]:04d}_s{k[1]}.pt").is_file()}
        if not novos:
            return
        tmp = Path(tempfile.mkdtemp(dir=destino.parent))
        for (i, s), c in novos.items():
            os.link(destino / f"p{i:04d}_s{s}.pt", tmp / c)
        tam = sum((tmp / c).stat().st_size for c in novos.values())
        try:
            api.upload_folder(folder_path=str(tmp), path_in_repo="professor", repo_id=repo,
                              repo_type="dataset", commit_message=f"{len(novos)} shards do professor")
            log(f"professor-hf: {len(novos)} shards enviados ({tam / 2**30:.2f} GiB, "
                f"{time.perf_counter() - t0:.0f} s) [MEDIDO]")
        except Exception as e:  # noqa: BLE001 -- cache falho nao derruba o treino
            log(f"professor-hf: envio NAO completou ({type(e).__name__}: {str(e)[:500]})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class EmpurraHF:
    """Sobe o checkpoint para um repo PRIVADO do HF, numa thread, sem parar o treino.

    Por que: no Colab o `/content` morre com a VM, `colab drivemount` trava sem TTY (registrado
    pelo lab PT-BR) e `colab download` até a máquina local mede ~5 MB/s (~50 min para 15 GB). Da VM
    para o HF o tráfego fica na rede do Google -- [JULGAMENTO] deve ser muito mais rápido; a
    velocidade real vai IMPRESSA a cada envio, porque ninguém mediu ainda.

    O TOKEN NÃO É DESTE CÓDIGO. `HfApi()` usa o que o ambiente tiver, e nesta bancada já se viu um
    token de LEITURA ser pego no lugar do de escrita e o envio morrer em 403 no meio. Então a
    permissão é provada ANTES de treinar, por prova positiva: sobe 1 byte e apaga (`whoami` responde
    igual para token de leitura).

    Cada envio é um commit com ~14 GB de LFS; sem limpeza o repo acumularia um checkpoint inteiro
    por commit. Depois de cada envio, `super_squash_history` deixa só a revisão atual.

    E ISSO NÃO BASTA, medido 2026-09-23: o squash apaga o histórico mas NÃO devolve a cota na hora.
    Com um único arquivo de 14,55 GB no HEAD, o repo marcava 46,9 GB depois de 3 envios; somado aos
    buckets do dono passou dos 100 GB privados da conta free e todo envio seguinte morreu em
    `BadRequestError` no commit (a causa ficou escondida 2 h porque o erro era truncado em 160
    caracteres). Pendente: apagar e recriar o repo antes de cada envio, ou enviar só o mestre."""

    def __init__(self, repo: str, publico: bool = False):
        import queue
        import threading
        from huggingface_hub import HfApi
        self.api, self.repo, self.t = HfApi(), repo, None
        # Fila de envios AVULSOS (exportado `melhor`, exportado final, json): um worker, em ordem, para
        # dois commits nunca correrem juntos no mesmo repo. O checkpoint segue pelo caminho antigo.
        self.fila: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        # PUBLICO nao conta na cota privada de 100 GB da conta free -- que foi o que derrubou os
        # envios de 2026-09-23 (ver o paragrafo acima). Decisao do dono: checkpoints podem ser publicos.
        self.api.create_repo(repo, private=not publico, exist_ok=True)
        self.api.upload_file(path_or_fileobj=b"x", path_in_repo="_prova_escrita", repo_id=repo)
        self.api.delete_file("_prova_escrita", repo_id=repo)
        log(f"HF: escrita PROVADA em {repo} (subiu e apagou 1 byte)")

    def baixa_se_faltar(self, arq: Path) -> None:
        if arq.is_file():
            return
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo, "ckpt/ultimo.pt", local_dir=str(arq.parent.parent))
            log(f"HF: checkpoint baixado para retomar ({Path(p).stat().st_size / 2**30:.2f} GiB)")
        except Exception as e:  # noqa: BLE001 -- repo novo, sem checkpoint: comeca do zero
            log(f"HF: sem checkpoint remoto ({type(e).__name__}); comecando do zero")

    def envia(self, arq: Path) -> None:
        import threading
        if (self.t is not None and self.t.is_alive()) or self.fila.unfinished_tasks:
            log("HF: envio anterior ainda em curso; este checkpoint fica so local")
            return

        def corre():
            t0 = time.perf_counter()
            try:
                self.api.upload_file(path_or_fileobj=str(arq), path_in_repo="ckpt/ultimo.pt",
                                     repo_id=self.repo)
                self.api.super_squash_history(repo_id=self.repo)
                dt = time.perf_counter() - t0
                log(f"HF: enviado {arq.stat().st_size / 2**30:.2f} GiB em {dt:.0f} s "
                    f"({arq.stat().st_size / 2**20 / dt:.0f} MB/s) [MEDIDO]")
            except Exception as e:  # noqa: BLE001 -- envio falho nao pode derrubar o treino
                log(f"HF: envio NAO completou ({type(e).__name__}: {str(e)[:2000]}); checkpoint local intacto")
        self.t = threading.Thread(target=corre, daemon=False)
        self.t.start()

    def _worker(self) -> None:
        while True:
            fonte, destino, apagar = self.fila.get()
            t0 = time.perf_counter()
            try:
                if self.t is not None and self.t.is_alive():
                    self.t.join()  # nao concorrer com o envio do checkpoint no mesmo repo
                self.api.upload_file(path_or_fileobj=fonte if isinstance(fonte, bytes) else str(fonte),
                                     path_in_repo=destino, repo_id=self.repo)
                tam = len(fonte) if isinstance(fonte, bytes) else Path(fonte).stat().st_size
                log(f"HF: {destino} enviado ({tam / 2**30:.2f} GiB, {time.perf_counter() - t0:.0f} s)")
                if apagar and not isinstance(fonte, bytes):
                    Path(fonte).unlink()
            except Exception as e:  # noqa: BLE001 -- envio falho nao derruba o treino
                log(f"HF: {destino} NAO enviado ({type(e).__name__}: {str(e)[:2000]})")
            finally:
                self.fila.task_done()

    def avulso(self, fonte, destino: str, apagar: bool = False) -> None:
        self.fila.put((fonte, destino, apagar))

    def espera(self) -> None:
        if self.t is not None:
            self.t.join()
        self.fila.join()


def salva_ckpt(pasta: Path, aluno, treinaveis, opt, passo: int, extra: dict) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / "ultimo.pt"
    tmp = pasta / "ultimo.pt.partial"
    estado = {"passo": passo, "extra": extra,
              "mestre": {k: v.detach().to("cpu") for k, v in treinaveis},
              "otim": opt.state_dict(),
              "rng_py": random.getstate(), "rng_torch": torch.get_rng_state()}
    t0 = time.perf_counter()
    torch.save(estado, tmp)
    os.replace(tmp, arq)
    log(f"  checkpoint passo {passo}: {arq.stat().st_size / 2**30:.2f} GiB em "
        f"{time.perf_counter() - t0:.0f} s")


def exporta(aluno, corpo: set[str], grupo: int, saida: Path, meta: dict, escalas: bool = False) -> None:
    """safetensors em nomenclatura diffusers: corpo = ternário desempacotado bf16, resto = mestre bf16.
    Com `escalas` (modo --so-escalas) o corpo é código congelado x escala aprendida."""
    from safetensors.torch import save_file
    sd = {}
    mods = dict(aluno.named_modules()) if escalas else {}
    with torch.no_grad():
        for k, v in aluno.state_dict().items():
            if k.endswith("._esc"):
                continue
            v = v.detach()
            if k in corpo and escalas:
                m = mods[k[:-len(".weight")]]
                nn_, kk = m._cod.shape
                v = (m._cod.float().view(nn_, kk // grupo, grupo) * m._esc).view(nn_, kk)
            elif k in corpo:
                v = ternariza(v.float(), grupo)
            sd[k] = v.to("cpu", torch.bfloat16).contiguous()
    tmp = saida.with_suffix(saida.suffix + ".partial")
    save_file(sd, str(tmp), metadata={k: str(v) for k, v in meta.items()})
    os.replace(tmp, saida)
    log(f"exportado {saida}  {saida.stat().st_size:,} B")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raiz", required=True, help="pasta do pipeline (model_index.json etc)")
    p.add_argument("--professor", required=True, help="transformer BF16, nomes diffusers")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--prompts-holdout", type=Path, required=True)
    p.add_argument("--sementes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--passos", type=int, default=8)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dir", type=Path, required=True, help="pasta PERSISTENTE: shards, ckpt, journal")
    p.add_argument("--otim", choices=["adamw8bit-sr", "adamw4bit-sr", "adamw8bit", "adam-fp32"], required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-passos", type=int, required=True)
    p.add_argument("--lote", type=int, default=1,
                   help="exemplos por passo, concatenados (nao acumulacao): mesmo ruido de gradiente "
                        "de acumular, mas usa a placa melhor se couber")
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--so-escalas", action="store_true",
                   help="congela o codigo do corpo no arredondamento do BF16 e treina SO uma escala por grupo "
                        "(implica --congela-resto)")
    p.add_argument("--professor-hf", default=None, metavar="REPO",
                   help="repo de DADOS no HF que guarda os shards do professor entre VMs (baixa antes, sobe depois)")
    p.add_argument("--formato", choices=["ternario", "int4"], default="ternario",
                   help="codigo do corpo: ternario (absmean, -1..1) ou int4 (absmax/7, -7..7; use --grupo 32)")
    p.add_argument("--ckpt-min", type=float, default=25.0)
    p.add_argument("--log-cada", type=int, default=50)
    p.add_argument("--holdout-cada", type=int, default=500)
    p.add_argument("--sem-grad-ckpt", action="store_true")
    p.add_argument("--so-professor", action="store_true", help="grava os shards e sai")
    p.add_argument("--push-hf", default=None,
                   help="repo PRIVADO do HF para o checkpoint (Colab). O token de escrita tem de "
                        "estar no ambiente, posto pelo dono; a escrita e provada antes de treinar")
    p.add_argument("--hf-publico", action="store_true",
                   help="cria o repo do --push-hf como PUBLICO (nao conta na cota privada)")
    p.add_argument("--inicia-de", type=Path, default=None,
                   help="checkpoint de OUTRA corrida: carrega mestre + estado do otimizador, mas zera o "
                        "contador de passos e a ordem dos exemplos (mesma semente de embaralhamento). "
                        "Serve para REPETIR a mesma sequencia de exemplos a partir de pesos ja treinados.")
    p.add_argument("--congela-resto", action="store_true",
                   help="treina SO o corpo ternario; os tensores fora dele (densos, norms) ficam os do professor")
    p.add_argument("--lr-denso", type=float, default=None,
                   help="lr dos tensores FORA do corpo ternario (modulacao, embedders, normas); padrao = --lr")
    p.add_argument("--l1-corpo", type=float, default=0.0,
                   help="lambda do L1 proximal relativo ao grupo no corpo (esparsifica; ver prox_l1_relativo). "
                        "Passo por iteracao = lr * lambda * d_g")
    p.add_argument("--sens-passos", default="0,3,6",
                   help="passos do sampler usados na guarda de condicionamento (sens)")
    p.add_argument("--sens-min", type=float, default=0.0,
                   help="piso ABSOLUTO de sens para `melhor` (sens = d_aluno/d_ref); 0 = so' o relativo")
    p.add_argument("--sens-tol", type=float, default=0.1,
                   help="piso RELATIVO: sens >= (1 - tol) * maior sens ja' visto no braco")
    p.add_argument("--parada-paciencia", type=int, default=0,
                   help="para o braco se holdout_rel piorar N avaliacoes seguidas (0 = nunca)")
    p.add_argument("--cruzado", action="store_true",
                   help="grava a saida do professor com a condicao do prompt vizinho (fase 2; ver grava_cruzado)")
    p.add_argument("--cruzado-frac", type=float, default=0.0,
                   help="fracao dos passos que treina num exemplo cruzado em vez de um normal")
    p.add_argument("--avalia-ckpt", type=Path, nargs="+", default=None,
                   help="so' mede holdout, holdout_rel e sens destes checkpoints e sai (calibracao)")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    if a.lr_denso is None:
        a.lr_denso = a.lr
    global NIVEIS
    NIVEIS = 7 if a.formato == "int4" else 1

    dev = torch.device(f"cuda:{a.device}")
    raiz = Path(a.raiz)
    hf = EmpurraHF(a.push_hf, a.hf_publico) if a.push_hf else None
    a.dir.mkdir(parents=True, exist_ok=True)
    (a.dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2),
                                     encoding="utf-8")
    tr_p, ho_p = le_prompts(a.prompts), le_prompts(a.prompts_holdout)
    comum_prompts = set(tr_p) & set(ho_p)
    if comum_prompts:
        print(f"RECUSADO: {len(comum_prompts)} prompts no treino E no holdout, p.ex. "
              f"{sorted(comum_prompts)[:2]}", file=sys.stderr)
        return 2
    log(f"prompts: {len(tr_p)} treino, {len(ho_p)} holdout; sementes {a.sementes}; "
        f"{a.passos} passos a {a.size} px")

    for lista, sub in ((tr_p, "professor"), (ho_p, "professor_holdout")):
        cache_professor(a, lista, a.dir / sub, "baixa")
        grava_professor(a, raiz, lista, a.dir / sub, dev)
        cache_professor(a, lista, a.dir / sub, "sobe")
    if a.cruzado:
        # o holdout primeiro: e' pequeno e da' a metrica holdout_cruz_rel mesmo se o treino nao terminar
        grava_cruzado(a, raiz, a.dir / "professor_holdout", a.dir / "professor_holdout_cruzado", dev)
        if not a.avalia_ckpt:  # a calibracao so' precisa do cruzado do holdout
            grava_cruzado(a, raiz, a.dir / "professor", a.dir / "professor_cruzado", dev)
    if a.so_professor:
        log("FIM " + time.strftime("%H:%M:%S"))
        return 0

    treino = Exemplos(a.dir / "professor", a.sementes, a.passos)
    holdout = Exemplos(a.dir / "professor_holdout", a.sementes, a.passos, cache=32)
    sens_sel = [int(x) for x in a.sens_passos.split(",") if x.strip()]
    ho_cruz = ExemplosCruzados(holdout, a.dir / "professor_holdout_cruzado", sens_sel)
    tr_cruz = ExemplosCruzados(treino, a.dir / "professor_cruzado")
    if a.cruzado_frac > 0 and len(tr_cruz) == 0:
        print("RECUSADO: --cruzado-frac > 0 sem shards cruzados de treino (rodar com --cruzado)", file=sys.stderr)
        return 2
    log(f"exemplos: {len(treino)} treino, {len(holdout)} holdout; cruzados {len(tr_cruz)} treino, "
        f"{len(ho_cruz)} holdout")

    # ---------------------------------------------------------------- aluno, do BF16 original
    from ajusta_denso_diffusers import carrega_transformer
    mestre_dt = torch.bfloat16 if a.otim.endswith("-sr") else torch.float32
    aluno = carrega_transformer(Path(a.professor), raiz / "transformer" / "config.json",
                                torch.bfloat16, torch.device("cpu"))
    aluno = aluno.to(dtype=mestre_dt).to(dev)
    nomes = [k for k, v in aluno.named_parameters()]
    pil = pilhas_reais(nomes)
    corpo = {k for k, v in aluno.named_parameters()
             if v.ndim == 2 and (m := BLOCO.match(k)) and m.group("pilha") in pil}
    n_ste = instala_ste(aluno, corpo, a.grupo, torch.bfloat16)
    if n_ste != len(corpo):
        print(f"RECUSADO: {len(corpo)} pesos de corpo e {n_ste} Linear trocados -- algum peso de "
              f"corpo nao e nn.Linear.", file=sys.stderr)
        return 2
    if a.so_escalas:
        instala_escalas(aluno, corpo, a.grupo, torch.bfloat16)
        a.congela_resto = True
    if not a.sem_grad_ckpt:
        aluno.enable_gradient_checkpointing()
    aluno.train()
    treinaveis = ([(k, v) for k, v in aluno.named_parameters() if k.endswith("._esc")] if a.so_escalas
                  else list(aluno.named_parameters()))
    n_par = sum(v.numel() for _, v in treinaveis)
    n_corpo = sum(v.numel() for k, v in treinaveis if k in corpo)
    log(f"aluno: {n_par:,} params treinaveis, corpo ternario {n_corpo:,} em {n_ste} Linear, "
        f"mestre {mestre_dt}, otim {a.otim}")
    corpo_mods = [(n, m) for n, m in aluno.named_modules()
                  if isinstance(m, torch.nn.Linear) and f"{n}.weight" in corpo]
    with torch.no_grad():
        cod0 = {n: codigo_e_escala(m.weight, a.grupo)[0].cpu() for n, m in corpo_mods}
    cod_ant: dict = {}
    contexto = (torch.autocast("cuda", dtype=torch.bfloat16) if mestre_dt == torch.float32
                else torch.autocast("cuda", enabled=False))

    # ------------------------------------------------ guarda de condicionamento (sens): referencia
    # Pares de shards do holdout com a MESMA semente e prompts vizinhos: a saida no mesmo x_t com a
    # condicao do proprio prompt vs a do vizinho. d = ||s(x,c_i) - s(x,c_j)|| / ||s(x,c_i)||. A referencia
    # e' o mesmo d com o mestre AINDA igual ao BF16 e o STE desligado -> sens = d_aluno / d_ref. Colapso
    # (imagem igual para todo prompt, visto na A100 #1 em p9191 com o MSE ainda caindo) = sens caindo.
    global STE_LIGADO
    sens_passos = [int(x) for x in a.sens_passos.split(",") if x.strip()]
    por_semente: dict[int, list[Path]] = {}
    for arq in holdout.arqs:
        por_semente.setdefault(int(arq.stem.rsplit("_s", 1)[1]), []).append(arq)
    pares = [(L[i], L[(i + 1) % len(L)]) for L in por_semente.values() if len(L) >= 2 for i in range(len(L))]

    def mede_d() -> float:
        ds = []
        with torch.no_grad():
            for ai, aj in pares:
                shi, shj = holdout._le(ai), holdout._le(aj)
                for pj in sens_passos:
                    kw = shi["passos"][pj]["kw"]
                    sii = forward_aluno(aluno, shi["comum"], kw, dev, contexto).float()
                    sij = forward_aluno(aluno, shj["comum"], kw, dev, contexto).float()
                    ds.append(float((sii - sij).norm() / sii.norm().clamp(min=1e-12)))
        return sum(ds) / len(ds)

    def erro_rel(conj) -> float:
        """Erro relativo medio do aluno contra o professor num conjunto (usado no cruzado do holdout:
        mede se o aluno responde a TROCA de condicao como o professor responde)."""
        rs = []
        with torch.no_grad():
            for comum, kw, alvo in conj:
                s = forward_aluno(aluno, comum, kw, dev, contexto).float()
                al = alvo.to(dev, torch.float32)
                rs.append(float((s - al).norm() / al.norm().clamp(min=1e-12)))
        return sum(rs) / len(rs)

    ref_arq = a.dir / "ref_sens.json"
    if ref_arq.is_file():
        d_ref = json.loads(ref_arq.read_text(encoding="utf-8"))["d_ref"]
    else:
        # ANTES de carregar qualquer checkpoint: aqui o mestre ainda e' o BF16 do professor, bit a bit.
        STE_LIGADO = False
        d_ref = mede_d()
        STE_LIGADO = True
        ref_arq.write_text(json.dumps({"d_ref": d_ref, "pares": len(pares), "passos": sens_passos}),
                           encoding="utf-8")
    log(f"sens: referencia BF16 d_ref {d_ref:.5f} ({len(pares)} pares x {len(sens_passos)} passos)")

    if a.avalia_ckpt:
        # Calibracao do instrumento em checkpoints cujo desfecho em imagem ja' se conhece; sem treino.
        for cp in a.avalia_ckpt:
            est = torch.load(cp, map_location="cpu", weights_only=False, mmap=True)
            with torch.no_grad():
                for k, v in treinaveis:
                    v.copy_(est["mestre"][k].to(dev, v.dtype))
            vs, rs = [], []
            with torch.no_grad():
                for comum, kw, alvo in holdout:
                    s = forward_aluno(aluno, comum, kw, dev, contexto).float()
                    al = alvo.to(dev, torch.float32)
                    vs.append(float(F.mse_loss(s, al)))
                    rs.append(float((s - al).norm() / al.norm().clamp(min=1e-12)))
            linha = {"ckpt": str(cp), "passo": est["passo"], "holdout": sum(vs) / len(vs),
                     "holdout_rel": sum(rs) / len(rs), "sens": mede_d() / d_ref}
            if len(ho_cruz):
                linha["holdout_cruz_rel"] = erro_rel(ho_cruz)
            log("AVALIA " + json.dumps(linha))
            with (a.dir / "avalia_ckpt.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(linha) + "\n")
            del est
        log("FIM " + time.strftime("%H:%M:%S"))
        return 0

    denso_params = [v for k, v in treinaveis if k not in corpo]
    corpo_params = [v for k, v in treinaveis if k in corpo]
    if a.so_escalas:
        # o "corpo" treinavel sao as escalas; todo o resto (inclusive os pesos originais) fica congelado
        corpo_params, denso_params = [v for _, v in treinaveis], []
        for k, v in aluno.named_parameters():
            if not k.endswith("._esc"):
                v.requires_grad_(False)
    # lr como TENSOR no grupo: o torchao da VM (2026-09-24) nao converte o lr de um grupo passado em
    # dict e o `step()` morre com "lr was changed to a non-Tensor object"; o 0.18 local converte em
    # `add_param_group`, por isso o smoke local passou. Tensor funciona nos dois.
    grupos = [{"params": corpo_params, "lr": torch.tensor(a.lr, dtype=torch.float32)}]
    if a.congela_resto:
        # Transplantes de 24/09: corpo do b4d + resto ORIGINAL faz cena sem o atrator, e o atrator mora no
        # resto treinado. Aqui o resto nem entra no otimizador: fica byte a byte o do professor.
        for v in denso_params:
            v.requires_grad_(False)
    else:
        grupos.append({"params": denso_params, "lr": torch.tensor(a.lr_denso, dtype=torch.float32)})
    opt = cria_otimizador(a.otim, grupos, a.lr)
    log(f"lr corpo {a.lr:g}, lr denso {'CONGELADO' if a.congela_resto else f'{a.lr_denso:g}'}, "
        f"L1 corpo lambda {a.l1_corpo:g}")
    passo = 0
    ck = a.dir / "ckpt" / "ultimo.pt"
    if hf is not None:
        hf.baixa_se_faltar(ck)
    random.seed(20260922)
    if ck.is_file():
        # mmap: o checkpoint tem ~11 GiB e a 3090 local roda com ~15 GiB de RAM livre. Conferido
        # 2026-09-23: o pickle abre em 1,2 s e o estado do AdamW4bit volta como `OptimState4bit`
        # (o aviso "Unable to import torchao Tensor objects" do import NAO impede). O
        # `opt.load_state_dict` na GPU ainda nao foi exercitado numa retomada real.
        est = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        with torch.no_grad():
            for k, v in treinaveis:
                v.copy_(est["mestre"][k].to(dev, v.dtype))
        opt.load_state_dict(est["otim"])
        random.setstate(est["rng_py"])
        torch.set_rng_state(est["rng_torch"])
        passo = est["passo"]
        log(f"RETOMADO do passo {passo}")
        del est
    elif a.inicia_de is not None:
        # Pesos e Adam de outra corrida; passo 0 e RNG da semente fixa -> a MESMA ordem de exemplos
        # que a corrida de origem viu desde o inicio. O contador de passo do Adam continua o dele.
        # Estado de otimizador de UM grupo (corridas antigas) nao casa com os dois grupos de agora.
        est = torch.load(a.inicia_de, map_location="cpu", weights_only=False, mmap=True)
        with torch.no_grad():
            for k, v in treinaveis:
                v.copy_(est["mestre"][k].to(dev, v.dtype))
        opt.load_state_dict(est["otim"])
        log(f"INICIADO de {a.inicia_de} (passo {est['passo']} da origem); ordem de exemplos do zero")
        (a.dir / "origem.json").write_text(json.dumps({"inicia_de": str(a.inicia_de),
                                                       "passo_origem": est["passo"]}), encoding="utf-8")
        del est
    log(f"VRAM alocada antes do laco: {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB")

    def perda_holdout() -> tuple[float, float]:
        """(MSE medio, erro relativo medio ||s-a||/||a||). O MSE e' dominado pelos passos de sigma alto
        e ficou CEGO ao colapso duas vezes; o relativo pesa todo passo igual."""
        vs, rs = [], []
        with torch.no_grad():
            for comum, kw, alvo in holdout:
                s = forward_aluno(aluno, comum, kw, dev, contexto).float()
                al = alvo.to(dev, torch.float32)
                vs.append(float(F.mse_loss(s, al)))
                rs.append(float((s - al).norm() / al.norm().clamp(min=1e-12)))
        return sum(vs) / len(vs), sum(rs) / len(rs)

    melhor_arq = a.dir / "melhor.json"
    melhor = (json.loads(melhor_arq.read_text(encoding="utf-8")) if melhor_arq.is_file()
              else {"holdout_rel": float("inf"), "passo": None, "ruins_seguidas": 0,
                    "melhor_rel_qualquer": float("inf")})

    def avalia(passo_atual: int) -> dict:
        """Holdout + codigos + sens; decide `melhor` e conta avaliacoes ruins para a parada."""
        h, hr = perda_holdout()
        sens = mede_d() / d_ref
        r = {"holdout": h, "holdout_rel": hr, "sens": sens,
             **estatistica_codigos(aluno, corpo_mods, cod0, cod_ant, a.grupo)}
        if len(ho_cruz):
            r["holdout_cruz_rel"] = erro_rel(ho_cruz)
        # Limiar RELATIVO ao maior sens ja' visto: o ternario ingenuo mede sens 0,156 (smoke 2026-09-23),
        # entao um piso absoluto de 0,8 pararia todo braco antes de ele ter chance de subir. O que marca
        # o colapso e' o sens CAIR enquanto o holdout melhora.
        # o passo 0 (ternario ingenuo, imagem destruida) NAO entra: o d dele e' reacao erratica de um
        # modelo quebrado, nao condicionamento -- e no smoke o sens caiu de 0,156 para 0,066 em 2 passos.
        if passo_atual > 0:
            melhor["sens_max"] = max(melhor.get("sens_max", 0.0), sens)
        piso = max(a.sens_min, melhor.get("sens_max", 0.0) * (1 - a.sens_tol))
        r["sens_piso"] = piso
        if passo_atual > 0 and hr < melhor["holdout_rel"] and sens >= piso:
            saida = a.dir / f"melhor_p{passo_atual}.safetensors"
            exporta(aluno, corpo, a.grupo, saida, escalas=a.so_escalas, meta=
                    {"quantizado_por": "tools/qat_ternario_klein.py", "formato": a.formato, "passo": passo_atual,
                     "holdout_rel": hr, "sens": sens, "lr": a.lr, "lr_denso": a.lr_denso,
                     "l1_corpo": a.l1_corpo, "format": "pt"})
            if hf is None:
                for velho in a.dir.glob("melhor_p*.safetensors"):
                    if velho != saida:
                        velho.unlink()
            melhor.update(holdout_rel=hr, passo=passo_atual, sens=sens)
            if hf is not None:
                hf.avulso(saida, "melhor/aluno_ternario_diffusers.safetensors", apagar=True)
                hf.avulso(json.dumps(melhor).encode(), "melhor/melhor.json")
            r["melhor"] = True
        ruim = not (hr < melhor["melhor_rel_qualquer"]) or sens < piso
        melhor["melhor_rel_qualquer"] = min(melhor["melhor_rel_qualquer"], hr)
        melhor["ruins_seguidas"] = melhor["ruins_seguidas"] + 1 if (ruim and passo_atual > 0) else 0
        melhor_arq.write_text(json.dumps(melhor), encoding="utf-8")
        return r

    jr = (a.dir / "journal.jsonl").open("a", encoding="utf-8")
    ordem: list[int] = []
    ult_ck = time.monotonic()
    acum, t_log = [], time.perf_counter()
    if passo == 0:
        linha = {"passo": 0, **avalia(0)}
        jr.write(json.dumps(linha) + "\n")
        jr.flush()
        log(f"passo 0  holdout {linha['holdout']:.5e}  holdout_rel {linha['holdout_rel']:.4f}  "
            f"sens {linha['sens']:.3f}  (ternario ingenuo a partir do BF16, antes de treinar)")
    parou = False
    # Sorteio do cruzado num gerador PROPRIO: com --cruzado-frac 0 a ordem dos exemplos normais e' a
    # mesma do controle (A100 #1); com frac > 0 cada passo troca o exemplo normal por um cruzado com
    # probabilidade frac, e a sequencia normal so' avanca nos passos normais. [retomada: este gerador
    # recomeca do zero; aceito, afeta so' quais passos sao cruzados]
    rng_cruz = random.Random(20260923)
    ordem_cruz: list[int] = []
    while passo < a.max_passos and not parou:
        if a.cruzado_frac > 0 and rng_cruz.random() < a.cruzado_frac:
            if len(ordem_cruz) < a.lote:
                ordem_cruz = list(range(len(tr_cruz)))
                rng_cruz.shuffle(ordem_cruz)
            fonte, ordem_da_vez = tr_cruz, ordem_cruz
        else:
            if len(ordem) < a.lote:
                ordem = list(range(len(treino)))
                random.shuffle(ordem)
            fonte, ordem_da_vez = treino, ordem
        kw, alvo = junta_lote([fonte[ordem_da_vez.pop()] for _ in range(a.lote)])
        s = forward_aluno(aluno, {}, kw, dev, contexto)
        perda = F.mse_loss(s.float(), alvo.to(dev, torch.float32))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        opt.step()
        if a.l1_corpo > 0:
            prox_l1_relativo(corpo_params, a.lr * a.l1_corpo, a.grupo)
        passo += 1
        v = float(perda.detach())
        if not math.isfinite(v):
            log(f"ABORTANDO: perda {v} no passo {passo}")
            return 1
        acum.append(v)
        if passo % a.log_cada == 0 or passo == a.max_passos:
            dt = (time.perf_counter() - t_log) / len(acum)
            linha = {"passo": passo, "perda": sum(acum) / len(acum), "s_por_passo": dt,
                     "vram_pico_gib": torch.cuda.max_memory_allocated(dev) / 2**30}
            if passo % a.holdout_cada == 0 or passo == a.max_passos:
                linha.update(avalia(passo))
                if a.parada_paciencia and melhor["ruins_seguidas"] >= a.parada_paciencia:
                    linha["parada"] = (f"{melhor['ruins_seguidas']} avaliacoes seguidas sem melhorar "
                                       f"holdout_rel ou com sens abaixo do piso")
                    parou = True
            jr.write(json.dumps(linha) + "\n")
            jr.flush()
            log("passo " + "  ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in linha.items() if k != "passo") + f"  [{passo}]")
            acum, t_log = [], time.perf_counter()
        if time.monotonic() - ult_ck > a.ckpt_min * 60:
            salva_ckpt(a.dir / "ckpt", aluno, treinaveis, opt, passo, {"otim": a.otim})
            if hf is not None:
                hf.envia(ck)
            ult_ck = time.monotonic()

    if parou:
        log(f"PARADA ANTECIPADA no passo {passo}; melhor: passo {melhor['passo']} "
            f"holdout_rel {melhor['holdout_rel']:.4f}")
    salva_ckpt(a.dir / "ckpt", aluno, treinaveis, opt, passo, {"otim": a.otim})
    if hf is not None:
        hf.espera()
        hf.envia(ck)
        hf.t.join()
    aluno.eval()
    final = a.dir / "aluno_ternario_diffusers.safetensors"
    exporta(aluno, corpo, a.grupo, final, escalas=a.so_escalas, meta=
            {"quantizado_por": "tools/qat_ternario_klein.py", "formato": a.formato, "otim": a.otim, "passos": passo,
             "grupo": a.grupo, "lr": a.lr, "lr_denso": a.lr_denso, "congela_resto": a.congela_resto, "so_escalas": a.so_escalas, "l1_corpo": a.l1_corpo,
             "format": "pt"})
    if hf is not None:
        hf.avulso(final, "final/aluno_ternario_diffusers.safetensors")
        hf.avulso(json.dumps({"passo": passo, "parou": parou, "melhor": melhor}).encode(),
                  "final/final.json")
        hf.espera()
    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem, nenhum epsilon no protocolo do ComfyUI: medir fora, depois.")
    print("  Nenhum tempo de 1,58 bit: corpo desempacotado em bf16.")
    log("FIM " + time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
