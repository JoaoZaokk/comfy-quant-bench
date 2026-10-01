"""Orquestração do QAT: professor -> referência sens -> (calibração | treino com retomada) -> exportado.

`main(argv, carrega=None)`: `carrega(cfg)` devolve o transformer do aluno (nomes diffusers) em CPU; o
default lê o BF16 do `--professor` pelo `ajusta_denso_diffusers`. Os testes injetam um modelo minúsculo.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lowbit_canon
import torch
import torch.nn.functional as F

from .avalia import Instrumentos, Juiz
from .config import TRAJETORIA, Config, ErroConfig, parser
from .estado import EstadoCorrida, Geradores, Parada, carrega_mestre, retoma, salva_ckpt
from .exporta import caminho_lowbit, exporta
from .modulos import (
    ChaveSTE,
    cria_otimizador,
    estatistica_codigos,
    instala,
    prox_l1_relativo,
)
from .professor import (
    Exemplos,
    ExemplosCruzados,
    cache_professor,
    forward_aluno,
    grava_cruzado,
    grava_professor,
    junta_lote,
    recusa_vazamento,
)
from .status import Status
from .util import grava_json_atomico, le_prompts, log

RC_PARADO = 143  # 128 + SIGTERM


def _carrega_padrao(cfg: Config):
    from ajusta_denso_diffusers import carrega_transformer
    return carrega_transformer(Path(cfg.professor), cfg.raiz / "transformer" / "config.json",
                               torch.bfloat16, torch.device("cpu"))


def main(argv=None, carrega=None, descricao: str = "") -> int:
    a = parser(descricao).parse_args(argv)
    try:
        cfg = Config.de_args(a)
    except ErroConfig as e:
        print(f"RECUSADO: {e.msg}", file=sys.stderr)
        return 2
    cfg.dir.mkdir(parents=True, exist_ok=True)
    status = Status(cfg.dir, "qat_ternario_klein")
    try:
        rc = _corre(cfg, status, carrega or _carrega_padrao)
    except SystemExit as e:
        msg = e.code if isinstance(e.code, str) else ""
        code = 1 if msg else (e.code or 0)
        if code:
            status.fim("RECUSADO" if "RECUSADO" in msg else "FALHOU", code, msg)
        raise
    except BaseException as e:
        status.fim("FALHOU", 1, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        raise
    if rc == 2:
        status.fim("RECUSADO", rc)
    return rc


def _registra_lancamento(cfg: Config) -> None:
    """`args.json` = o da PRIMEIRA execução nesta pasta (antes era sobrescrito a cada lançamento, apagando o
    registro da corrida); cada lançamento vai para `lancamentos.jsonl`."""
    d = cfg.como_dict()
    if not (cfg.dir / "args.json").is_file():
        (cfg.dir / "args.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    with (cfg.dir / "lancamentos.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **d}) + "\n")


def _confere_impressao(cfg: Config, est: dict, imp: dict) -> None:
    antiga = (est.get("estado") or {}).get("impressao") or {}
    if antiga:
        if antiga.get("hash") == imp["hash"]:
            return
        dif = {k: (antiga.get(k), imp.get(k)) for k in imp if k != "hash" and antiga.get(k) != imp.get(k)}
        if not cfg.retoma_config_diferente:
            raise SystemExit(f"RECUSADO: o checkpoint e' de outra configuracao: {dif} "
                             f"(--retoma-config-diferente para aceitar)")
        log(f"AVISO: retomando com configuracao diferente (aceito por flag): {dif}")
        return
    arq = cfg.dir / "args.json"
    if arq.is_file():  # checkpoint antigo: o melhor que há é o args.json (da última execução antiga)
        velho = json.loads(arq.read_text(encoding="utf-8"))
        atual = cfg.como_dict()
        dif = [k for k in TRAJETORIA if k in velho and velho[k] != atual[k]
               and not (k == "sementes" and velho[k].strip("[]()").replace(",", " ").split()
                        == atual[k].strip("[]()").replace(",", " ").split())]
        if dif:
            log(f"AVISO: checkpoint antigo sem impressao digital; args.json difere em {dif}")


def _corre(cfg: Config, status: Status, carrega) -> int:
    from .hf_sync import EmpurraHF
    dev = cfg.dispositivo()
    raiz = cfg.raiz
    hf = EmpurraHF(cfg.push_hf, cfg.hf_publico) if cfg.push_hf else None
    _registra_lancamento(cfg)
    tr_p, ho_p = le_prompts(cfg.prompts), le_prompts(cfg.prompts_holdout)
    comum_prompts = set(tr_p) & set(ho_p)
    if comum_prompts:
        print(f"RECUSADO: {len(comum_prompts)} prompts no treino E no holdout, p.ex. "
              f"{sorted(comum_prompts)[:2]}", file=sys.stderr)
        return 2
    log(f"prompts: {len(tr_p)} treino, {len(ho_p)} holdout; sementes {list(cfg.sementes)}; "
        f"{cfg.passos} passos a {cfg.size} px")

    status.atualiza(fase="professor")
    for lista, sub in ((tr_p, "professor"), (ho_p, "professor_holdout")):
        cache_professor(cfg, lista, cfg.dir / sub, "baixa")
        grava_professor(cfg, raiz, lista, cfg.dir / sub, dev)
        cache_professor(cfg, lista, cfg.dir / sub, "sobe")
    if cfg.cruzado:
        status.atualiza(fase="cruzado")
        # o holdout primeiro: e' pequeno e da' a metrica holdout_cruz_rel mesmo se o treino nao terminar
        grava_cruzado(cfg, raiz, cfg.dir / "professor_holdout", cfg.dir / "professor_holdout_cruzado", dev)
        if not cfg.avalia_ckpt:  # a calibracao so' precisa do cruzado do holdout
            grava_cruzado(cfg, raiz, cfg.dir / "professor", cfg.dir / "professor_cruzado", dev)
    if cfg.so_professor:
        log("FIM " + time.strftime("%H:%M:%S"))
        status.fim("FIM", 0)
        return 0
    recusa_vazamento(cfg.dir / "professor", cfg.dir / "professor_holdout", cfg.sementes)

    treino = Exemplos(cfg.dir / "professor", cfg.sementes, cfg.passos, size=cfg.size)
    holdout = Exemplos(cfg.dir / "professor_holdout", cfg.sementes, cfg.passos, cache=32, size=cfg.size)
    ho_cruz = ExemplosCruzados(holdout, cfg.dir / "professor_holdout_cruzado", cfg.sens_passos)
    tr_cruz = ExemplosCruzados(treino, cfg.dir / "professor_cruzado")
    if cfg.cruzado_frac > 0 and len(tr_cruz) == 0:
        print("RECUSADO: --cruzado-frac > 0 sem shards cruzados de treino (rodar com --cruzado)", file=sys.stderr)
        return 2
    log(f"exemplos: {len(treino)} treino, {len(holdout)} holdout; cruzados {len(tr_cruz)} treino, "
        f"{len(ho_cruz)} holdout")

    # ---------------------------------------------------------------- aluno, do BF16 original
    mestre_dt = torch.bfloat16 if cfg.otim.endswith("-sr") else torch.float32
    aluno = carrega(cfg).to(dtype=mestre_dt).to(dev)
    corpo = lowbit_canon.corpo(aluno.named_parameters())
    chave = ChaveSTE()
    corpo_mods = instala(aluno, corpo, cfg.quantizador, chave, torch.bfloat16,
                         "escalas" if cfg.so_escalas else "ste")
    if not cfg.sem_grad_ckpt:
        aluno.enable_gradient_checkpointing()
    aluno.train()
    treinaveis = ([(k, v) for k, v in aluno.named_parameters() if k.endswith("._esc")] if cfg.so_escalas
                  else list(aluno.named_parameters()))
    n_par = sum(v.numel() for _, v in treinaveis)
    n_corpo = sum(v.numel() for k, v in treinaveis if k in corpo)
    log(f"aluno: {n_par:,} params treinaveis, corpo {cfg.formato} {n_corpo:,} em {len(corpo_mods)} Linear, "
        f"mestre {mestre_dt}, otim {cfg.otim}")
    with torch.no_grad():
        cod0 = {n: m.codigo_e_escala()[0].cpu() for n, m in corpo_mods}
    cod_ant: dict = {}
    contexto = (torch.autocast(dev.type, dtype=torch.bfloat16) if mestre_dt == torch.float32
                else torch.autocast(dev.type, enabled=False))

    # ------------------------------------------------ guarda de condicionamento (sens): referencia
    status.atualiza(fase="referencia")
    inst = Instrumentos(aluno, holdout, ho_cruz, cfg.sens_passos, dev, contexto)
    # ANTES de carregar qualquer checkpoint: aqui o mestre ainda e' o BF16 do professor, bit a bit.
    d_ref = inst.referencia(cfg.dir / "ref_sens.json", chave)
    log(f"sens: referencia BF16 d_ref {d_ref:.5f} ({len(inst.pares)} pares x {len(cfg.sens_passos)} passos)")

    if cfg.avalia_ckpt:
        return _calibra(cfg, inst, d_ref, treinaveis, dev, status)

    denso_params = [v for k, v in treinaveis if k not in corpo]
    corpo_params = [v for k, v in treinaveis if k in corpo]
    if cfg.so_escalas:
        # o "corpo" treinavel sao as escalas; todo o resto (inclusive os pesos originais) fica congelado
        corpo_params, denso_params = [v for _, v in treinaveis], []
        for k, v in aluno.named_parameters():
            if not k.endswith("._esc"):
                v.requires_grad_(False)
    # lr como TENSOR no grupo: o torchao da VM (2026-09-24) nao converte o lr de um grupo passado em
    # dict e o `step()` morre com "lr was changed to a non-Tensor object". Tensor funciona nos dois.
    grupos = [{"params": corpo_params, "lr": torch.tensor(cfg.lr, dtype=torch.float32)}]
    if cfg.congela_resto:
        # Transplantes de 24/09: o atrator mora no resto treinado. Aqui o resto nem entra no otimizador.
        for v in denso_params:
            v.requires_grad_(False)
    else:
        grupos.append({"params": denso_params, "lr": torch.tensor(cfg.lr_denso, dtype=torch.float32)})
    opt = cria_otimizador(cfg.otim, grupos, cfg.lr)
    log(f"lr corpo {cfg.lr:g}, lr denso {'CONGELADO' if cfg.congela_resto else f'{cfg.lr_denso:g}'}, "
        f"L1 corpo lambda {cfg.l1_corpo:g}")

    # ------------------------------------------------ estado: novo, retomado ou iniciado de outra corrida
    imp = cfg.impressao()
    ger = Geradores()
    ck = cfg.dir / "ckpt" / "ultimo.pt"
    melhor_arq = cfg.dir / "melhor.json"
    retomado = False
    if hf is not None:
        hf.baixa_se_faltar(ck)
    if ck.is_file():
        # mmap: o checkpoint tem ~11 GiB e a 3090 local roda com ~15 GiB de RAM livre.
        est = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        _confere_impressao(cfg, est, imp)
        legado = json.loads(melhor_arq.read_text(encoding="utf-8")) if melhor_arq.is_file() else None
        estado = retoma(est, treinaveis, opt, dev, ger, legado)
        retomado = True
        log(f"RETOMADO do passo {estado.passo}")
        del est
    else:
        estado = EstadoCorrida()
        if melhor_arq.is_file():
            log("AVISO: melhor.json sem checkpoint nesta pasta: ignorado (seria de uma corrida morta)")
        if cfg.inicia_de is not None:
            # Pesos e Adam de outra corrida; passo 0 e RNG da semente fixa -> a MESMA ordem de exemplos
            # que a corrida de origem viu desde o inicio. O contador de passo do Adam continua o dele.
            est = torch.load(cfg.inicia_de, map_location="cpu", weights_only=False, mmap=True)
            carrega_mestre(est, treinaveis, dev)
            opt.load_state_dict(est["otim"])
            log(f"INICIADO de {cfg.inicia_de} (passo {est['passo']} da origem); ordem de exemplos do zero")
            grava_json_atomico(cfg.dir / "origem.json", {"inicia_de": str(cfg.inicia_de), "passo_origem": est["passo"]})
            del est
    estado.impressao = imp
    if dev.type == "cuda":
        log(f"VRAM alocada antes do laco: {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB")

    juiz = Juiz(cfg, estado.melhor, melhor_arq)
    base_meta = {"quantizado_por": "tools/qat_ternario_klein.py", "formato": cfg.formato, "lr": cfg.lr,
                 "lr_denso": cfg.lr_denso, "l1_corpo": cfg.l1_corpo, "format": "pt"}

    def avalia(passo_atual: int) -> dict:
        status.atualiza(fase="avaliando", passo=passo_atual)
        h, hr = inst.perda_holdout()
        sens = inst.mede_d() / d_ref
        r = {"holdout": h, "holdout_rel": hr, "sens": sens, **estatistica_codigos(corpo_mods, cod0, cod_ant)}
        if len(ho_cruz):
            r["holdout_cruz_rel"] = inst.erro_rel(ho_cruz)
        novo, piso = juiz.julga(passo_atual, hr, sens)
        r["sens_piso"] = piso
        if novo:
            saida = cfg.dir / f"melhor_p{passo_atual}.safetensors"
            feitos = exporta(aluno, corpo_mods, saida, {**base_meta, "passo": passo_atual, "holdout_rel": hr,
                                                        "sens": sens}, cfg.exporta)
            if hf is None:
                for velho in cfg.dir.glob("melhor_p*.safetensors"):
                    if velho not in feitos:
                        velho.unlink()
            else:
                for f in feitos:
                    alvo = "melhor/aluno_ternario_diffusers.safetensors"
                    hf.avulso(f, caminho_lowbit(Path(alvo)).as_posix() if f.stem.endswith("_lowbit") else alvo,
                              apagar=True)
                hf.avulso(json.dumps(juiz.melhor).encode(), "melhor/melhor.json")
            r["melhor"] = True
        status.atualiza(fase="treino")
        return r

    jr = (cfg.dir / "journal.jsonl").open("a", encoding="utf-8")
    passo = estado.passo
    if retomado:
        jr.write(json.dumps({"retomado_de": passo, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                 + "\n")
        jr.flush()
    ult_ck = time.monotonic()
    acum, t_log = [], time.perf_counter()
    status.atualiza(fase="treino", passo=passo)
    if passo == 0:
        linha = {"passo": 0, **avalia(0)}
        jr.write(json.dumps(linha) + "\n")
        jr.flush()
        log(f"passo 0  holdout {linha['holdout']:.5e}  holdout_rel {linha['holdout_rel']:.4f}  "
            f"sens {linha['sens']:.3f}  ({cfg.formato} ingenuo a partir do BF16, antes de treinar)")
    parou = False
    parada = Parada(cfg.para_no_passo)
    parada.instala()
    # Sorteio do cruzado num gerador PROPRIO: com --cruzado-frac 0 a ordem dos exemplos normais e' a
    # mesma do controle (A100 #1); com frac > 0 cada passo troca o exemplo normal por um cruzado com
    # probabilidade frac, e a sequencia normal so' avanca nos passos normais.
    try:
        while passo < cfg.max_passos and not parou:
            if cfg.cruzado_frac > 0 and ger.cruz.random() < cfg.cruzado_frac:
                if len(estado.ordem_cruz) < cfg.lote:
                    estado.ordem_cruz = list(range(len(tr_cruz)))
                    ger.cruz.shuffle(estado.ordem_cruz)
                fonte, ordem_da_vez = tr_cruz, estado.ordem_cruz
            else:
                if len(estado.ordem) < cfg.lote:
                    estado.ordem = list(range(len(treino)))
                    ger.ordem.shuffle(estado.ordem)
                fonte, ordem_da_vez = treino, estado.ordem
            kw, alvo = junta_lote([fonte[ordem_da_vez.pop()] for _ in range(cfg.lote)])
            s = forward_aluno(aluno, {}, kw, dev, contexto)
            perda = F.mse_loss(s.float(), alvo.to(dev, torch.float32))
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            if cfg.l1_corpo > 0:
                prox_l1_relativo(corpo_params, cfg.lr * cfg.l1_corpo, cfg.grupo)
            passo += 1
            estado.passo = passo
            v = float(perda.detach())
            if not math.isfinite(v):
                log(f"ABORTANDO: perda {v} no passo {passo}")
                status.fim("FALHOU", 1, f"perda {v} no passo {passo}")
                return 1
            acum.append(v)
            if passo % cfg.log_cada == 0 or passo == cfg.max_passos:
                dt = (time.perf_counter() - t_log) / len(acum)
                linha = {"passo": passo, "perda": sum(acum) / len(acum), "s_por_passo": dt,
                         "vram_pico_gib": (torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda"
                                           else 0.0)}
                if passo % cfg.holdout_cada == 0 or passo == cfg.max_passos:
                    linha.update(avalia(passo))
                    if juiz.deve_parar():
                        linha["parada"] = (f"{juiz.melhor['ruins_seguidas']} avaliacoes seguidas sem melhorar "
                                           f"holdout_rel ou com sens abaixo do piso")
                        parou = True
                jr.write(json.dumps(linha) + "\n")
                jr.flush()
                log("passo " + "  ".join(f"{k} {x:.5g}" if isinstance(x, float) else f"{k} {x}"
                                         for k, x in linha.items() if k != "passo") + f"  [{passo}]")
                acum, t_log = [], time.perf_counter()
                status.atualiza(passo=passo)
            if time.monotonic() - ult_ck > cfg.ckpt_min * 60:
                salva_ckpt(cfg.dir / "ckpt", treinaveis, opt, estado, {"otim": cfg.otim}, ger)
                if hf is not None:
                    hf.envia(ck, passo)
                ult_ck = time.monotonic()
            if parada.confere(passo):
                break
    finally:
        parada.restaura()

    if parada.pedida:
        salva_ckpt(cfg.dir / "ckpt", treinaveis, opt, estado, {"otim": cfg.otim}, ger)
        if hf is not None:
            hf.espera()
            hf.envia(ck, passo)
            hf.espera()
        jr.write(json.dumps({"parado_no": passo, "motivo": parada.motivo}) + "\n")
        jr.close()
        log(f"PARADO no passo {passo} ({parada.motivo}); checkpoint gravado, retomavel")
        status.fim("PARADO", RC_PARADO, parada.motivo)
        return RC_PARADO

    if parou:
        log(f"PARADA ANTECIPADA no passo {passo}; melhor: passo {juiz.melhor['passo']} "
            f"holdout_rel {juiz.melhor['holdout_rel']:.4f}")
    salva_ckpt(cfg.dir / "ckpt", treinaveis, opt, estado, {"otim": cfg.otim}, ger)
    if hf is not None:
        hf.espera()
        hf.envia(ck, passo)
        hf.espera()
    aluno.eval()
    status.atualiza(fase="exportando")
    final = cfg.dir / "aluno_ternario_diffusers.safetensors"
    feitos = exporta(aluno, corpo_mods, final, {**base_meta, "otim": cfg.otim, "passos": passo, "grupo": cfg.grupo,
                                                "congela_resto": cfg.congela_resto, "so_escalas": cfg.so_escalas},
                     cfg.exporta)
    if hf is not None:
        for f in feitos:
            hf.avulso(f, f"final/{f.name}")
        hf.avulso(json.dumps({"passo": passo, "parou": parou, "melhor": juiz.melhor}).encode(), "final/final.json")
        hf.espera()
    jr.close()
    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem, nenhum epsilon no protocolo do ComfyUI: medir fora, depois.")
    print("  Nenhum tempo de 1,58 bit: corpo desempacotado em bf16.")
    log("FIM " + time.strftime("%H:%M:%S"))
    status.fim("FIM", 0)
    return 0


def _calibra(cfg: Config, inst: Instrumentos, d_ref: float, treinaveis, dev, status: Status) -> int:
    """Calibração do instrumento em checkpoints cujo desfecho em imagem já se conhece; sem treino."""
    for cp in cfg.avalia_ckpt:
        est = torch.load(cp, map_location="cpu", weights_only=False, mmap=True)
        carrega_mestre(est, treinaveis, dev)
        h, hr = inst.perda_holdout()
        linha = {"ckpt": str(cp), "passo": est["passo"], "holdout": h, "holdout_rel": hr,
                 "sens": inst.mede_d() / d_ref}
        if len(inst.ho_cruz):
            linha["holdout_cruz_rel"] = inst.erro_rel(inst.ho_cruz)
        log("AVALIA " + json.dumps(linha))
        with (cfg.dir / "avalia_ckpt.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(linha) + "\n")
        del est
    log("FIM " + time.strftime("%H:%M:%S"))
    status.fim("FIM", 0)
    return 0
