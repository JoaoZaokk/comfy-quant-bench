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


def codigo_e_escala(w: torch.Tensor, grupo: int):
    """(código int8 [n, k], escala fp32 [n, k//grupo, 1]). Mesma receita do braço 0."""
    n, k = w.shape
    g = w.float().reshape(n, k // grupo, grupo)
    d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
    t = (g / d).clamp(-1, 1).round()
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


def instala_ste(modelo: torch.nn.Module, corpo: set[str], grupo: int, dtype_calc) -> int:
    """Troca o forward de cada nn.Linear do corpo por F.linear(x, STE(w)). Devolve quantos."""
    n = 0
    for nome, m in modelo.named_modules():
        if isinstance(m, torch.nn.Linear) and f"{nome}.weight" in corpo:
            def fwd(x, _m=m):
                w = _STE.apply(_m.weight, grupo).to(dtype_calc)
                b = None if _m.bias is None else _m.bias.to(dtype_calc)
                return F.linear(x.to(dtype_calc), w, b)
            m.forward = fwd
            n += 1
    return n


# ----------------------------------------------------------------------------------------- professor

def le_prompts(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def grava_professor(a, raiz: Path, prompts: list[str], destino: Path, dev) -> None:
    """Um shard por (prompt, semente): {"comum": kw que não muda entre passos, "passos": [...]}.

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
            pipe(prompt=p, height=a.size, width=a.size, num_inference_steps=a.passos,
                 guidance_scale=1.0, generator=torch.Generator(device="cpu").manual_seed(s),
                 output_type="latent")
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
        if j == 0 or (j + 1) % 16 == 0 or j == len(faltam) - 1:
            log(f"  professor {j + 1}/{len(faltam)}  {len(atual)} chamadas  "
                f"{(time.perf_counter() - t0) / (j + 1):.1f} s/shard")
    tr.forward = orig
    pipe.to("cpu")
    del pipe, tr
    torch.cuda.empty_cache()


def carrega_exemplos(d: Path) -> list[tuple[dict, dict, torch.Tensor]]:
    """[(comum, kw_do_passo, alvo)] com `comum` COMPARTILHADO entre os passos da mesma chamada."""
    ex = []
    for arq in sorted(d.glob("p*_s*.pt")):
        sh = torch.load(arq, map_location="cpu", weights_only=False)
        for ps in sh["passos"]:
            ex.append((sh["comum"], ps["kw"], ps["out"]))
    return ex


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
    if nome == "adamw8bit-sr":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)
    if nome == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0)
    if nome == "adam-fp32":
        return torch.optim.Adam(params, lr=lr)
    raise SystemExit(f"otimizador desconhecido: {nome}")


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
    por commit. Depois de cada envio, `super_squash_history` deixa só a revisão atual."""

    def __init__(self, repo: str):
        from huggingface_hub import HfApi
        self.api, self.repo, self.t = HfApi(), repo, None
        self.api.create_repo(repo, private=True, exist_ok=True)
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
        if self.t is not None and self.t.is_alive():
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
                log(f"HF: envio NAO completou ({type(e).__name__}: {str(e)[:160]}); checkpoint local intacto")
        self.t = threading.Thread(target=corre, daemon=False)
        self.t.start()


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


def exporta(aluno, corpo: set[str], grupo: int, saida: Path, meta: dict) -> None:
    """safetensors em nomenclatura diffusers: corpo = ternário desempacotado bf16, resto = mestre bf16."""
    from safetensors.torch import save_file
    sd = {}
    with torch.no_grad():
        for k, v in aluno.state_dict().items():
            v = v.detach()
            if k in corpo:
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
    p.add_argument("--otim", choices=["adamw8bit-sr", "adamw8bit", "adam-fp32"], required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-passos", type=int, required=True)
    p.add_argument("--lote", type=int, default=1,
                   help="exemplos por passo, concatenados (nao acumulacao): mesmo ruido de gradiente "
                        "de acumular, mas usa a placa melhor se couber")
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--ckpt-min", type=float, default=25.0)
    p.add_argument("--log-cada", type=int, default=50)
    p.add_argument("--holdout-cada", type=int, default=500)
    p.add_argument("--sem-grad-ckpt", action="store_true")
    p.add_argument("--so-professor", action="store_true", help="grava os shards e sai")
    p.add_argument("--push-hf", default=None,
                   help="repo PRIVADO do HF para o checkpoint (Colab). O token de escrita tem de "
                        "estar no ambiente, posto pelo dono; a escrita e provada antes de treinar")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.device}")
    raiz = Path(a.raiz)
    hf = EmpurraHF(a.push_hf) if a.push_hf else None
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

    grava_professor(a, raiz, tr_p, a.dir / "professor", dev)
    grava_professor(a, raiz, ho_p, a.dir / "professor_holdout", dev)
    if a.so_professor:
        log("FIM " + time.strftime("%H:%M:%S"))
        return 0

    treino = carrega_exemplos(a.dir / "professor")
    holdout = carrega_exemplos(a.dir / "professor_holdout")
    log(f"exemplos: {len(treino)} treino, {len(holdout)} holdout")

    # ---------------------------------------------------------------- aluno, do BF16 original
    from ajusta_denso_diffusers import carrega_transformer
    mestre_dt = torch.bfloat16 if a.otim == "adamw8bit-sr" else torch.float32
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
    if not a.sem_grad_ckpt:
        aluno.enable_gradient_checkpointing()
    aluno.train()
    treinaveis = list(aluno.named_parameters())
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

    opt = cria_otimizador(a.otim, [v for _, v in treinaveis], a.lr)
    passo = 0
    ck = a.dir / "ckpt" / "ultimo.pt"
    if hf is not None:
        hf.baixa_se_faltar(ck)
    random.seed(20260922)
    if ck.is_file():
        est = torch.load(ck, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for k, v in treinaveis:
                v.copy_(est["mestre"][k].to(dev, v.dtype))
        opt.load_state_dict(est["otim"])
        random.setstate(est["rng_py"])
        torch.set_rng_state(est["rng_torch"])
        passo = est["passo"]
        log(f"RETOMADO do passo {passo}")
        del est
    log(f"VRAM alocada antes do laco: {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB")

    def perda_holdout() -> float:
        vs = []
        with torch.no_grad():
            for comum, kw, alvo in holdout:
                s = forward_aluno(aluno, comum, kw, dev, contexto)
                vs.append(float(F.mse_loss(s.float(), alvo.to(dev, torch.float32))))
        return sum(vs) / len(vs)

    jr = (a.dir / "journal.jsonl").open("a", encoding="utf-8")
    ordem: list[int] = []
    ult_ck = time.monotonic()
    acum, t_log = [], time.perf_counter()
    if passo == 0:
        h = perda_holdout()
        cs = estatistica_codigos(aluno, corpo_mods, cod0, cod_ant, a.grupo)
        linha = {"passo": 0, "holdout": h, **cs}
        jr.write(json.dumps(linha) + "\n")
        jr.flush()
        log(f"passo 0  holdout {h:.5e}  (ternario ingenuo a partir do BF16, antes de treinar)")
    while passo < a.max_passos:
        if len(ordem) < a.lote:
            ordem = list(range(len(treino)))
            random.shuffle(ordem)
        kw, alvo = junta_lote([treino[ordem.pop()] for _ in range(a.lote)])
        s = forward_aluno(aluno, {}, kw, dev, contexto)
        perda = F.mse_loss(s.float(), alvo.to(dev, torch.float32))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        opt.step()
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
                linha["holdout"] = perda_holdout()
                linha.update(estatistica_codigos(aluno, corpo_mods, cod0, cod_ant, a.grupo))
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

    salva_ckpt(a.dir / "ckpt", aluno, treinaveis, opt, passo, {"otim": a.otim})
    if hf is not None:
        if hf.t is not None:
            hf.t.join()
        hf.envia(ck)
        hf.t.join()
    aluno.eval()
    exporta(aluno, corpo, a.grupo, a.dir / "aluno_ternario_diffusers.safetensors",
            {"quantizado_por": "tools/qat_ternario_klein.py", "otim": a.otim, "passos": passo,
             "grupo": a.grupo, "lr": a.lr, "format": "pt"})
    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem, nenhum epsilon no protocolo do ComfyUI: medir fora, depois.")
    print("  Nenhum tempo de 1,58 bit: corpo desempacotado em bf16.")
    log("FIM " + time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
