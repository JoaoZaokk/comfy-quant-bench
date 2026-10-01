"""Camada 2 do avaliador: a matematica quantizada EXECUTA hoje, ou o peso so ocupa menos VRAM?

A CAMADA 1 NAO PODE RESPONDER ISTO, E DIZ QUE NAO PODE
------------------------------------------------------
`tools/avaliar.py` le cabecalho, sidecar e analise -- 163 checkpoints em 0,68 s, sem torch. Ela ve
o campo `backend` do `.quant.json` e sabe que os kernels resolveram para
`comfy_kitchen.backends.cuda`, mas o cego da propria checagem ja avisa: *"e o registro de uma
conversao passada, nao uma execucao agora"*. Entre aquela conversao e hoje mudaram comfy-kitchen
(0.2.23 -> 0.2.31), ComfyUI (0.29 -> 0.33) e torch (2.12.1 -> 2.13.0), e nenhum deles avisa quando
um formato deixa de resolver.

Esta camada carrega o modelo pelo caminho normal do ComfyUI, roda um forward e CONTA.

O QUE ELA NAO REIMPLEMENTA, DE PROPOSITO
-----------------------------------------
A contagem e do `tools/probe_quant_dispatch.py`, chamado por subprocesso. Ele ja resolveu os dois
erros dificeis, e refaze-los aqui seria criar uma segunda verdade que diverge:

  * le o kwarg que o proprio `comfy/ops.py` calcula (`weight_only_quant`) em vez de re-derivar
    `_use_quantized`, que foi como uma versao anterior mediu o backend errado;
  * instrumenta DEPOIS do load, porque o load dequantiza de forma legitima.

DUAS ARMADILHAS QUE DECIDEM SE ESTA CAMADA PRESTA
--------------------------------------------------
**Text encoder e travado de fabrica.** `comfy/sd.py:269` chama `set_model_compute_dtype(float32)`
para TODO objeto CLIP, o que liga `comfy_force_cast_weights` em cada modulo e derruba a matematica
para dequantizada. Medido em 2026-08-31: um encoder ConvRot INT4 publico de 13,2 GiB faz 350
dequantize e ZERO chamadas ao caminho de 4 bits. Isso nao e defeito do checkpoint. Uma camada 2 que
dissesse `NAO_DESPACHA` para todo text encoder estaria certa nos numeros e errada na conclusao, e
mandaria alguem reconverter arquivos que estao bons. Dai `TRAVADO_PELO_COMFY`, com a causa nomeada.

**Zero significa duas coisas opostas.** Contador zerado por nao ter executado e contador zerado por
ter executado dequantizado dao o mesmo numero e significam o contrario. O probe separa pelo campo
`rodou`; esta camada propaga a separacao em vez de somar. Forward que morreu vira `SEM VEREDITO`,
nunca `NAO_DESPACHA`.

`DESPACHA` NAO E APROVACAO
--------------------------
Ele responde uma pergunta binaria e decidivel -- o kernel foi chamado? -- e nada mais. Medido nesta
bancada: o HunyuanVideo 1.5 W4A4 despacha nativamente e o render e destruido; o Wan 2.1 VACE misto
despacha e sai uma mancha. Por isso o veredito tem o nome do FATO e nao do julgamento, e `APROVADO`
continua ausente do avaliador inteiro.

Criterio, previsoes e as tres condicoes de refutacao: `bench/criterio_camada2_despacho.md`,
escrito antes de rodar.

USO
---
    python.exe -s tools/avaliar_despacho.py --controles      # so os dois controles, ~2 min
    python.exe -s tools/avaliar_despacho.py                  # tudo que a camada 1 marcou
    python.exe -s tools/avaliar_despacho.py --so ltx --limite 3
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PY = RAIZ / "python_embeded" / "python.exe"
PROBE = RAIZ / "tools" / "probe_quant_dispatch.py"

sys.path.insert(0, str(RAIZ / "tools"))
import gpu_lock  # noqa: E402

# Variavel de modulo para o teste apontar para um arquivo temporario.
LOCK_PATH = gpu_lock.LOCK_PATH

DESPACHA = "DESPACHA"
MISTO = "MISTO"
NAO_DESPACHA = "NAO_DESPACHA"
TRAVADO = "TRAVADO_PELO_COMFY"
NAO_QUANTIZADO = "NAO_CHEGOU_QUANTIZADO"
SEM_VEREDITO = "SEM VEREDITO"
NAO_PROBAVEL = "NAO_PROBAVEL"

# O probe resolve o nome por `folder_paths`, que so conhece estas duas pastas para os dois modos.
# Um arquivo em `checkpoints/` ou `unet/` nao e reprovado: e declarado nao-probavel, porque a
# limitacao e da ferramenta e nao do arquivo, e somar as duas produziria a mesma contagem zero.
PASTA_PARA_MODO = {"diffusion_models": "diffusion", "text_encoders": "te"}

# Verdade conhecida, medida em 2026-08-31. Existem para poder DERRUBAR esta ferramenta: se o
# positivo nao despachar ou o negativo despachar, nenhuma outra linha da tabela vale nada.
CONTROLES = {
    "zimage-v2-w4a4.safetensors": DESPACHA,
    "gemma_3_12B_it_heretic_w4a8.safetensors": TRAVADO,
}


@dataclass
class Alvo:
    caminho: Path
    modo: str | None
    camadas: int
    formatos: dict = field(default_factory=dict)


def alvos_da_camada1(dir_laudos: Path) -> list[Alvo]:
    """Le os laudos da camada 1 e devolve so quem tem camada quantizada."""
    saida: list[Alvo] = []
    for p in sorted(dir_laudos.glob("*.json")):
        try:
            laudo = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        fatos = laudo.get("fatos")
        if not isinstance(fatos, dict) or not fatos.get("camadas_quantizadas"):
            continue
        caminho = Path(laudo.get("arquivo", ""))
        saida.append(Alvo(caminho, PASTA_PARA_MODO.get(caminho.parent.name),
                          int(fatos["camadas_quantizadas"]),
                          fatos.get("formatos") or {}))
    return saida


def lock_vivo(dono: str | None = None) -> tuple[bool, str, dict]:
    """(ok, motivo, estado). Este lote carrega um checkpoint por vez na GPU, dezenas em sequencia,
    e ate a revisao de 2026-09-29 nao olhava lock nenhum. Ele nao TOMA o lock (a regra: fora do
    `_timing.compare()`/BenchGuard, quem roda GPU segura o lock com `Assert-GpuLock` antes); ele
    EXIGE que haja um lock vivo -- pid vivo e heartbeat dentro do limite do protocolo -- e, com
    `dono`, que seja aquele. Sem isso recusa, em vez de disputar a placa com quem esta medindo."""
    estado = gpu_lock.read_state(LOCK_PATH)
    if estado is None:
        return False, (f"sem lock em {LOCK_PATH}. Tome antes: . tools/gpu_lock.ps1; "
                       "Assert-GpuLock -Owner 'comfy:avaliar_despacho'"), {}
    if not estado:
        return False, f"{LOCK_PATH} existe mas nao pode ser lido", {}
    descricao = gpu_lock.describe(estado)
    try:
        vivo = gpu_lock._pid_alive(int(estado.get("pid", "")))
        idade = time.time() - int(estado.get("hb", ""))
    except ValueError:
        return False, f"lock sem pid/hb legiveis: {descricao}", estado
    if not vivo or idade > gpu_lock.STALE_LIMIT_SEC:
        return False, f"lock nao esta vivo (pid morto ou heartbeat velho): {descricao}", estado
    if dono and estado.get("dono") != dono:
        return False, f"lock e de outro dono, nao de {dono!r}: {descricao}", estado
    return True, descricao, estado


def controles_da_execucao(device: int, descricao_lock: str) -> dict:
    """O que um laudo precisa dizer para ser comparavel: placa, lock, versoes, arvore do ComfyUI.
    A camada 1 ja avisa que o backend do sidecar e 'o registro de uma conversao passada'; um laudo
    da camada 2 sem as versoes de AGORA tem o mesmo defeito."""
    versoes = {}
    for dist in ("torch", "comfy-kitchen", "triton-windows"):
        try:
            versoes[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versoes[dist] = None
    try:
        cabeca = subprocess.run(["git", "-C", str(RAIZ / "ComfyUI"), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20, check=False).stdout.strip() or None
        sujo = bool(subprocess.run(["git", "-C", str(RAIZ / "ComfyUI"), "status", "--porcelain",
                                    "--untracked-files=no"], capture_output=True, text=True,
                                   timeout=60, check=False).stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        cabeca, sujo = None, None
    return {"quando": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": device,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "lock": descricao_lock, "versoes": versoes, "comfyui_head": cabeca,
            "comfyui_com_alteracoes_locais": sujo, "python": sys.version.split()[0]}


def rodar_probe(alvo: Alvo, device: int, segundos: int) -> tuple[dict | None, str]:
    cmd = [str(PY), "-s", str(PROBE), "--mode", alvo.modo, "--forward-only", "--json",
           "--device", str(device), alvo.caminho.name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=segundos, cwd=str(RAIZ))
    except subprocess.TimeoutExpired:
        return None, f"estourou {segundos}s carregando ou rodando"
    for linha in proc.stdout.splitlines():
        if linha.startswith("@@JSON@@"):
            try:
                return json.loads(linha[len("@@JSON@@"):]), ""
            except json.JSONDecodeError:
                return None, "o probe emitiu @@JSON@@ ilegivel"
    cauda = (proc.stderr or proc.stdout).strip().splitlines()
    return None, (cauda[-1][:180] if cauda else f"o probe nao emitiu JSON (rc={proc.returncode})")


def julgar(alvo: Alvo, rep: dict | None, motivo: str) -> tuple[str, str, dict]:
    """Um relatorio do probe vira veredito. Toda a logica de zero-significa-duas-coisas mora aqui."""
    if alvo.modo is None:
        return (NAO_PROBAVEL,
                f"vive em `{alvo.caminho.parent.name}/`, que o probe nao resolve -- "
                f"limitacao da ferramenta, nao do arquivo", {})
    if rep is None:
        return SEM_VEREDITO, motivo, {}

    contas = rep.get("counts") or {}
    q = contas.get("quantizado", 0) + contas.get("quantizado_com_entrada_qt", 0)
    nq = contas.get("nao_quantizado", 0)
    dq = contas.get("dequantize", 0)
    impls = {k[len("impl:"):]: v for k, v in contas.items() if k.startswith("impl:")}
    numeros = {"modulos_quantizados": rep.get("n_modulos_quantizados", 0),
               "forwards_quantizados": q, "forwards_sem": nq, "dequantize": dq,
               "impl": impls, "quant_format": rep.get("quant_format") or {},
               # O veredito TRAVADO_PELO_COMFY depende INTEIRAMENTE deste campo, e a primeira
               # versao nao o gravava: dava para ler `NAO_DESPACHA` no laudo e nao ter como
               # conferir por que nao foi `TRAVADO`. Um veredito cuja evidencia nao esta no
               # relatorio e uma opiniao. Achado ao ver o `qwen3vl_32b_minimax_h3-int4_convrot`
               # -- o MESMO arquivo com que esta bancada estabeleceu a trava dos text encoders --
               # sair NAO_DESPACHA em vez de TRAVADO, sem nada no laudo que explicasse a diferenca.
               "comfy_force_cast_weights": rep.get("comfy_force_cast_weights") or {},
               "full_precision_mm": rep.get("full_precision_mm") or {}}

    if rep.get("erro"):
        # `n_modulos_quantizados == 0` com a camada 1 dizendo que ha camadas quantizadas no
        # ARQUIVO e um achado sobre o arquivo (o loader nao reconheceu o formato), nao uma falha
        # da medicao -- por isso nao e SEM VEREDITO.
        return NAO_QUANTIZADO, f"o loader nao viu nenhuma camada quantizada: {rep['erro']}", numeros
    if not rep.get("rodou"):
        return (SEM_VEREDITO,
                "o forward nao completou, entao os contadores nao dizem nada sobre o dispatch",
                numeros)
    if q > 0 and nq == 0:
        return DESPACHA, f"{q} forwards quantizados, 0 sem, {dq} dequantize", numeros
    if q > 0:
        return MISTO, f"{q} quantizados contra {nq} nao -- ler camada a camada", numeros

    # SAO DUAS TRAVAS, e a primeira versao desta funcao so conhecia uma.
    #
    # O `CLAUDE.md` documenta as duas: `comfy_force_cast_weights`, que vem de
    # `set_model_compute_dtype(float32)` em `comfy/sd.py:269`, e `full_precision_mm`, hardcodado
    # em `comfy/sd1_clip.py:114` para TODO text encoder. A regra aqui pedia so a primeira, e o
    # arquivo que a derrubou foi justamente `qwen3vl_32b_minimax_h3-int4_convrot` -- o mesmo com
    # que esta bancada ESTABELECEU a trava dos encoders. Medido 2026-09-01:
    #
    #     force_cast   {'False': 351}      <- a trava que a regra procurava: ausente
    #     fpmm         {'True': 350}       <- a outra: presente
    #
    # Ele saiu `NAO_DESPACHA`, que le como defeito do checkpoint e mandaria alguem reconverter um
    # arquivo publico que esta bom. Basta UMA das duas para a matematica cair, entao a regra pede
    # uma das duas e o veredito NOMEIA qual, porque as duas tem origem e conserto diferentes.
    forca = rep.get("comfy_force_cast_weights") or {}
    fpmm = rep.get("full_precision_mm") or {}
    presas = [nome for nome, contagem, origem in
              (("comfy_force_cast_weights", forca, "comfy/sd.py:269"),
               ("full_precision_mm", fpmm, "comfy/sd1_clip.py:114"))
              if contagem and set(contagem) - {"None"} == {"True"}]
    if alvo.modo == "te" and presas:
        onde = {"comfy_force_cast_weights": "comfy/sd.py:269",
                "full_precision_mm": "comfy/sd1_clip.py:114"}
        return (TRAVADO,
                "zero forwards quantizados, mas o ComfyUI trava todo text encoder por "
                + " e ".join(f"`{n}` ({onde[n]})" for n in presas)
                + ". A causa e o loader, nao o checkpoint. Memoria economizada, tempo nao.",
                numeros)
    # `forca`/`fpmm`, nao `travas`: a variavel foi renomeada quando a regra passou a conhecer as
    # DUAS travas, e este f-string ficou com o nome antigo -- `NameError` em todo NAO_DESPACHA
    # calculado do zero. Nao estourou na tabela porque a unica linha desse veredito veio do cache,
    # medida antes da reescrita. Achado pelo `ruff` (F821) dez minutos depois de ele ser instalado.
    return (NAO_DESPACHA,
            f"zero forwards quantizados com {dq} dequantize, force_cast em {forca or 'nada'} e "
            f"full_precision_mm em {fpmm or 'nada'} -- economia de memoria, nao de tempo", numeros)


NAO_COBERTO = [
    "Nada sobre QUALIDADE. Despachar nativamente e ortogonal a gerar imagem boa: nesta bancada",
    "  o HunyuanVideo 1.5 W4A4 despacha e o render e destruido, e o Wan 2.1 VACE misto despacha",
    "  e sai uma mancha. `DESPACHA` nao e aprovacao.",
    "Nada sobre FIDELIDADE: nenhuma referencia BF16 casada e carregada, entao nenhum numero aqui",
    "  compara contra coisa nenhuma.",
    "Poucos passos, lado pequeno -- o bastante para o dispatch acontecer, nao para julgar imagem.",
    "Uma placa, um build: comfy-kitchen 0.2.31, ComfyUI 0.33.0, torch 2.13.0+cu130, sm86. O",
    "  veredito e sobre ESTA maquina hoje -- que e o ponto da camada, e tambem o limite dela.",
    "Sem SASS: `chamou o kernel` vem de contador em Python, nao de instrucao de maquina.",
    "Checkpoint que a camada 1 nao marcou como quantizado nao e visitado. Ausencia de linha nao",
    "  e aprovacao nem reprova.",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laudos", type=Path, default=RAIZ / ".scratch" / "avaliacao",
                   help="pasta com a saida da camada 1")
    p.add_argument("--saida", type=Path, default=RAIZ / ".scratch" / "despacho")
    p.add_argument("--so", default=None, help="substring do nome do arquivo")
    p.add_argument("--limite", type=int, default=None)
    p.add_argument("--controles", action="store_true",
                   help="roda so os dois controles de verdade conhecida")
    p.add_argument("--refazer", action="store_true", help="ignora resultados ja gravados")
    p.add_argument("--segundos", type=int, default=900, help="teto por checkpoint")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dono", default=None,
                   help="exige que o lock vivo seja deste dono (o -Owner do Assert-GpuLock)")
    a = p.parse_args()

    if not a.laudos.is_dir():
        print(f"sem laudos da camada 1 em {a.laudos} -- rode `tools/avaliar.py --saida` antes",
              file=sys.stderr)
        return 2

    alvos = alvos_da_camada1(a.laudos)
    if a.controles:
        alvos = [x for x in alvos if x.caminho.name in CONTROLES]
    if a.so:
        alvos = [x for x in alvos if a.so.lower() in x.caminho.name.lower()]
    # Controles primeiro, sempre: se eles cairem, nao vale gastar placa no resto.
    alvos.sort(key=lambda x: (x.caminho.name not in CONTROLES, x.camadas))
    if a.limite:
        alvos = alvos[:a.limite]
    if not alvos:
        print("nenhum alvo", file=sys.stderr)
        return 2

    a.saida.mkdir(parents=True, exist_ok=True)
    vai_rodar = [x for x in alvos if x.modo is not None
                 and (a.refazer or not (a.saida / f"{x.caminho.stem}.json").is_file())]
    controles_exec = None
    if vai_rodar:
        ok, motivo, _estado = lock_vivo(a.dono)
        if not ok:
            print(f"RECUSADO: {len(vai_rodar)} checkpoint(s) iriam para a GPU e {motivo}",
                  file=sys.stderr)
            return 3
        controles_exec = controles_da_execucao(a.device, motivo)
        print(f"lock vivo: {motivo}")
    linhas, contagem, controles_vistos = [], {}, {}
    for i, alvo in enumerate(alvos, 1):
        destino = a.saida / f"{alvo.caminho.stem}.json"
        if destino.is_file() and not a.refazer:
            laudo = json.loads(destino.read_text(encoding="utf-8"))
            veredito, mensagem = laudo["veredito"], laudo["mensagem"] + "  (cache)"
            numeros, segundos = laudo.get("numeros", {}), laudo.get("segundos", 0.0)
        else:
            inicio = time.perf_counter()
            rep, motivo = (None, "") if alvo.modo is None else rodar_probe(alvo, a.device,
                                                                          a.segundos)
            veredito, mensagem, numeros = julgar(alvo, rep, motivo)
            segundos = time.perf_counter() - inicio
            destino.write_text(json.dumps(
                {"arquivo": str(alvo.caminho), "modo": alvo.modo, "camadas_no_arquivo": alvo.camadas,
                 "formatos": alvo.formatos, "veredito": veredito, "mensagem": mensagem,
                 "numeros": numeros, "segundos": round(segundos, 1),
                 "controles": controles_exec},
                indent=2, ensure_ascii=False), encoding="utf-8")

        contagem[veredito] = contagem.get(veredito, 0) + 1
        if alvo.caminho.name in CONTROLES:
            controles_vistos[alvo.caminho.name] = veredito
        print(f"[{i}/{len(alvos)}] {veredito:22s} {alvo.caminho.name}  ({segundos:.0f}s)")
        print(f"          {mensagem}")
        linhas.append((alvo, veredito, mensagem, numeros))

    print("\n" + "=" * 78)
    print("  ".join(f"{v} {k}" for k, v in sorted(contagem.items(), key=lambda kv: -kv[1])))

    # As condicoes de refutacao do criterio, cobradas aqui e nao na leitura de quem passar depois.
    falhou_controle = False
    for nome, esperado in CONTROLES.items():
        visto = controles_vistos.get(nome)
        if visto is None:
            print(f"CONTROLE NAO RODOU  {nome}: sem ele a tabela nao se distingue de uma "
                  f"ferramenta quebrada")
            falhou_controle = True
        elif visto != esperado:
            print(f"CONTROLE FALHOU  {nome}: esperado {esperado}, veio {visto}")
            falhou_controle = True
        else:
            print(f"controle OK  {nome}: {visto}")
    if falhou_controle:
        print("\nNAO PUBLICAR ESTA TABELA. Ver as condicoes de refutacao em")
        print("  bench/criterio_camada2_despacho.md")

    print("\nNAO COBERTO POR ESTA EXECUCAO:")
    for linha in NAO_COBERTO:
        print(f"  - {linha}")
    return 1 if falhou_controle else 0


if __name__ == "__main__":
    raise SystemExit(main())
