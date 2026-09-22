"""Um workflow salvo esta pronto para enfileirar nesta maquina? (EXECUTADO, CPU, sem abrir porta)

Duas perguntas, as duas contra o registro REAL do ComfyUI e nao contra o disco:

  1. todo `type` de no do workflow esta em `NODE_CLASS_MAPPINGS`?
  2. todo valor de widget que parece nome de modelo esta entre as opcoes que o PROPRIO no oferece?

A pergunta 2 e feita assim de proposito. A versao anterior desta checagem tinha um mapa escrito a mao
de tipo-de-no -> pasta do `folder_paths`, e ele errou DUAS vezes no mesmo workflow: `LTXVAudioVAELoader`
le `checkpoints`, nao `vae` (`comfy_extras/nodes_lt_audio.py:19`), e o segundo widget do
`LTXAVTextEncoderLoader` tambem e `ckpt_name`. As duas "faltas" eram do instrumento. Perguntar ao no
quais opcoes ele oferece nao pode errar de pasta, porque nao usa o conceito de pasta.

DOIS pontos cegos que este teste tem POR CONSTRUCAO, e que vao impressos em toda execucao:

  - Nos que so existem no navegador (LiteGraph) nunca entram em `NODE_CLASS_MAPPINGS` e aparecem como
    faltando. `SetNode`/`GetNode` do KJNodes sao assim -- `web/js/setgetnodes.js:901` e `:1213`
    chamam `LiteGraph.registerNodeType` e nao ha classe Python nenhuma. Eles estao na lista FRONTEND
    abaixo; o teste os separa em vez de mentir sobre eles.
  - Se o `PromptServer` nao existir antes de `init_extra_nodes`, todo pack com rota web morre com
    `type object 'PromptServer' has no attribute 'instance'`. Medido: 2349 tipos registrados sem ele
    contra 3350 com ele -- 1001 nos a menos, e KJNodes, rgthree e VideoHelperSuite entre os mortos.
    Este script segue a ordem do `main.py:525` -> `:531`.

Nao cobre: registro e nome, nunca execucao. Nao carrega peso, nao amostra, nao toca a GPU. Um no que
registra pode falhar no forward; um arquivo que aparece na lista pode estar corrompido.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# nos virtuais do frontend: registrados em JS, invisiveis ao Python por construcao
FRONTEND = {"SetNode", "GetNode", "Reroute", "PrimitiveNode", "Note", "MarkdownNote"}
EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")


def carrega_comfy(raiz: Path) -> tuple[dict, list[str]]:
    """Registra os nos na mesma ordem do main.py. Devolve (mappings, avisos)."""
    avisos: list[str] = []
    sys.argv = ["main.py", "--cpu"]
    sys.path.insert(0, str(raiz / "ComfyUI"))

    import utils.extra_config

    yaml = raiz / "ComfyUI/extra_model_paths.yaml"
    if yaml.is_file():
        utils.extra_config.load_extra_path_config(str(yaml))
    else:
        avisos.append(f"{yaml} nao existe: todo root montado fica invisivel")

    import hook_breaker_ac10a0
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop)  # main.py:525 -- sem isto, 1001 nos nao registram
    if server.PromptServer.instance is None:
        avisos.append("PromptServer.instance ficou None: a contagem abaixo esta subestimada")

    hook_breaker_ac10a0.save_functions()
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True))
    hook_breaker_ac10a0.restore_functions()
    return nodes.NODE_CLASS_MAPPINGS, avisos


def opcoes_do_no(cls) -> set[str]:
    """Uniao de todas as listas de opcao que a classe declara, nos dois formatos de schema."""
    op: set[str] = set()
    try:
        if hasattr(cls, "define_schema"):
            for e in cls.define_schema().inputs:
                for v in (getattr(e, "options", None) or []):
                    if isinstance(v, str):
                        op.add(v)
        elif hasattr(cls, "INPUT_TYPES"):
            it = cls.INPUT_TYPES()
            for grupo in it.values():
                if not isinstance(grupo, dict):
                    continue
                for spec in grupo.values():
                    alvo = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
                    if isinstance(alvo, (list, tuple)):
                        op.update(v for v in alvo if isinstance(v, str))
    except Exception as e:  # noqa: BLE001 -- schema de terceiro pode explodir de varias formas
        op.add(f"__ERRO_NO_SCHEMA__{type(e).__name__}")
    return op


def valores_de_arquivo(no: dict) -> list[str]:
    """Todo valor de widget que termina em extensao de modelo, inclusive dentro de dict/lista."""
    achados: list[str] = []

    def anda(v):
        if isinstance(v, str):
            if v.lower().endswith(EXT):
                achados.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                anda(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                anda(x)

    anda(no.get("widgets_values"))
    return achados


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("workflow", help="caminho do .json salvo")
    p.add_argument("--raiz", default="F:/COMFY_PORTABLE")
    a = p.parse_args()

    raiz = Path(a.raiz)
    wf = json.loads(Path(a.workflow).read_text(encoding="utf-8"))
    reg, avisos = carrega_comfy(raiz)
    for m in avisos:
        print(f"AVISO: {m}")
    print(f"{len(reg)} tipos de no registrados\n")

    faltam_no: list[str] = []
    frontend: list[str] = []
    faltam_arq: list[tuple[str, int, str]] = []
    desligados: list[tuple[str, int, str]] = []
    n_arq = 0

    for no in wf.get("nodes", []):
        t, nid = no.get("type"), no.get("id")
        if t not in reg:
            (frontend if t in FRONTEND else faltam_no).append(t)
            continue
        op = opcoes_do_no(reg[t])
        for nome in valores_de_arquivo(no):
            n_arq += 1
            if nome in op:
                continue
            # rgthree guarda lora desligada com "on": false -- nao carrega, entao nao bloqueia
            wv = no.get("widgets_values")
            off = any(isinstance(d, dict) and d.get("lora") == nome and d.get("on") is False
                      for d in (wv if isinstance(wv, list) else []))
            (desligados if off else faltam_arq).append((t, nid, nome))

    print(f"=== nos: {len(wf.get('nodes', []))} no total, {n_arq} valores de arquivo conferidos ===")
    for t in sorted(set(faltam_no)):
        print(f"  FALTA no       {t}")
    for t, nid, nome in faltam_arq:
        print(f"  FALTA arquivo  #{nid} {t}: {nome}")
    for t, nid, nome in desligados:
        print(f"  ausente mas DESLIGADO  #{nid} {t}: {nome}  (nao carrega, nao bloqueia)")
    for t in sorted(set(frontend)):
        print(f"  FRONTEND       {t}  (registrado em JS; este teste nao o ve, e isso e esperado)")

    bloqueia = len(faltam_no) + len(faltam_arq)
    print(f"\n=== VEREDITO: {bloqueia} item(ns) bloqueiam a fila ===")
    if not bloqueia:
        print("  Todo no Python registra e todo arquivo esta entre as opcoes do proprio no.")

    print("\n=== NAO COBERTO ===")
    print("  Registro e nome de arquivo. NAO houve render, carga de peso, amostragem nem uso de GPU.")
    print("  Nos de frontend (JS) e loras desligadas ficam fora da conta de bloqueio, de proposito.")
    print("  Imagem de entrada, prompt, semente e resolucao nao sao conferidos.")
    return 1 if bloqueia else 0


if __name__ == "__main__":
    raise SystemExit(main())
