"""Tela e MCP na mesma porta, para dirigir esta bancada sem ler tudo antes.

    python_embeded\\python.exe -s tools/bench_server.py
    -> http://127.0.0.1:8199/      a pagina
    -> http://127.0.0.1:8199/mcp   o endpoint MCP (JSON-RPC 2.0, HTTP streamable)

POR QUE ASSIM. Nesta instalacao NAO existe `tkinter`, `PySide6`, `PyQt5`, `gradio`, `fastapi`,
`flask` nem o pacote `mcp` -- e instalar pacote e decisao do dono. Existe `aiohttp` 3.14.3, que e o
que o proprio ComfyUI usa. Entao a tela e uma pagina servida por aiohttp e aberta no navegador, e o
MCP e JSON-RPC falado na mao. Zero instalacao.

EXECUTADO, 2026-08-31, contra o cliente MCP de verdade desta maquina:

    claude mcp list  ->  bancada: http://127.0.0.1:8199/mcp (HTTP) - OK Connected

O handshake que um cliente real exige e `initialize` -> `notifications/initialized` ->
`tools/list` -> `tools/call`. stdio NAO e exigido; HTTP streamable serve. O cliente manda
`Accept: application/json, text/event-stream` e honra o `Mcp-Session-Id` devolvido no initialize.

PARA REGISTRAR NO CLAUDE CODE:

    claude mcp add --transport http bancada http://127.0.0.1:8199/mcp

SEGURANCA, e nao e detalhe: liga SO em 127.0.0.1. As ferramentas de escrita ficam atras de
`--permitir-escrita`, desligado por padrao, porque um conversor consome horas de GPU e escreve
gigabytes -- nada disso deve comecar porque um modelo achou que devia. Sem a flag o servidor e
somente-leitura: ele conta o que existe e o que ja foi medido.

NAO COBERTO: sem autenticacao (o escopo e a maquina local). Nao valida que quem chama e o dono.
Nao implementa `resources` nem `prompts` do MCP, so `tools`. As respostas vem do disco na hora da
chamada -- nada e cacheado, e nada aqui verifica que os documentos que ele le estao corretos.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from aiohttp import web

RAIZ = Path(__file__).resolve().parent.parent
TOOLS = RAIZ / "tools"
sys.path.insert(0, str(TOOLS))

PROTOCOLO = "2025-06-18"
SESSOES: dict[str, dict] = {}
PERMITIR_ESCRITA = False


# ------------------------------------------------------------------ o que as ferramentas fazem

def _conversores() -> list[dict]:
    import convert
    return [{"subcomando": k, "modulo": f"{m}.py", "o_que_faz": d}
            for k, (m, d) in convert.COMANDOS.items()]


def _flags(subcomando: str) -> str:
    import convert
    if subcomando not in convert.COMANDOS:
        return f"subcomando desconhecido: {subcomando!r}. Conhecidos: {', '.join(convert.COMANDOS)}"
    r = subprocess.run([sys.executable, "-s", str(TOOLS / "convert.py"), subcomando, "--help"],
                       capture_output=True, text=True, cwd=str(RAIZ), timeout=120)
    return (r.stdout or r.stderr).strip()


def _checkpoints() -> list[dict]:
    """Todo .safetensors de diffusion_models e text_encoders, com o que o header diz.

    Le SO o cabecalho -- nunca mmap, nunca carrega tensor. Um modelo de 48 GiB custa milissegundos.
    """
    import struct
    out = []
    for sub in ("diffusion_models", "text_encoders"):
        d = RAIZ / "ComfyUI/models" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.safetensors")):
            reg = {"arquivo": f.name, "pasta": sub, "gib": round(f.stat().st_size / 1024**3, 2),
                   "quantizado": False, "formatos": {}, "camadas_quantizadas": 0}
            try:
                with open(f, "rb") as h:
                    n = struct.unpack("<Q", h.read(8))[0]
                    header = json.loads(h.read(n))
                meta = header.pop("__metadata__", {}) or {}
                qm = meta.get("_quantization_metadata")
                if qm:
                    camadas = json.loads(qm).get("layers", {})
                    reg["quantizado"] = True
                    reg["camadas_quantizadas"] = len(camadas)
                    for c in camadas.values():
                        k = c.get("format", "?")
                        reg["formatos"][k] = reg["formatos"].get(k, 0) + 1
                else:
                    inline = sum(1 for k in header if k.endswith(".comfy_quant"))
                    if inline:
                        reg["quantizado"] = True
                        reg["camadas_quantizadas"] = inline
                        reg["formatos"] = {"(por-camada, sem manifesto)": inline}
            except Exception as exc:  # noqa: BLE001
                reg["erro"] = f"{type(exc).__name__}: {exc}"
            out.append(reg)
    return out


def _arquivos(a: dict) -> dict:
    """Lista .safetensors sob as raizes permitidas, para a pagina oferecer escolha em vez de texto.

    POR QUE NAO E O DIALOGO DO WINDOWS. Um `<input type=file>` do navegador entrega o CONTEUDO do
    arquivo, nunca o caminho -- por politica de seguranca, e nao ha como contornar. E o servidor
    precisa de um caminho, porque quem le o arquivo e o conversor, do outro lado. Listar do lado do
    servidor resolve isso e ainda tem uma propriedade que um dialogo nativo nao teria: nao existe
    caminho fora das raizes para escolher, entao a validacao deixa de depender de o usuario se
    comportar.
    """
    import struct
    filtro = (a.get("filtro") or "").lower()
    saida = []
    for raiz in RAIZES:
        if not raiz.is_dir():
            saida.append({"raiz": str(raiz), "indisponivel": True, "arquivos": []})
            continue
        itens = []
        for f in sorted(raiz.rglob("*.safetensors")):
            rel = f.relative_to(raiz).as_posix()
            if filtro and filtro not in rel.lower():
                continue
            reg = {"caminho": str(f), "rel": rel,
                   "gib": round(f.stat().st_size / 1024**3, 2), "quantizado": None}
            try:
                with open(f, "rb") as h:
                    n = struct.unpack("<Q", h.read(8))[0]
                    cab = json.loads(h.read(n))
                meta = cab.pop("__metadata__", {}) or {}
                reg["quantizado"] = bool(meta.get("_quantization_metadata")) or any(
                    k.endswith(".comfy_quant") for k in cab)
            except Exception:  # noqa: BLE001
                pass                      # cabecalho ilegivel nao impede escolher o arquivo
            itens.append(reg)
            if len(itens) >= 400:
                break
        saida.append({"raiz": str(raiz), "indisponivel": False, "arquivos": itens})
    return {"raizes": saida,
            "nota": "Um checkpoint marcado como ja quantizado sera recusado pelo conversor: "
                    "requantizar destroi o que ja foi perdido uma vez."}


def _achados(consulta: str, limite: int = 12) -> list[dict]:
    """Busca nos documentos de medicao. E isto que responde 'isso a gente ja tinha visto?'."""
    docs = [RAIZ / n for n in (
        "CLAUDE.md", "W4A4_PROGRESS.md", "W4A4_HANDOFF.md", "AUDITORIA_2026-08-18.md",
        "RESUMO_MODELOS_2026-08-31.md", "CORTIQ_LTX25_HANDOFF.md")]
    docs += sorted(RAIZ.glob("HANDOFF_*.md"))
    docs += sorted((RAIZ / ".scratch/varredura-2026-08-22/issues").glob("*.md"))
    termos = [t for t in re.split(r"\s+", consulta.strip()) if t]
    if not termos:
        return []
    padrao = re.compile("|".join(re.escape(t) for t in termos), re.I)
    achados = []
    for d in docs:
        if not d.is_file():
            continue
        linhas = d.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, linha in enumerate(linhas):
            if not padrao.search(linha):
                continue
            pontos = sum(1 for t in termos if re.search(re.escape(t), linha, re.I))
            achados.append({"documento": d.relative_to(RAIZ).as_posix(), "linha": i + 1,
                            "texto": linha.strip()[:400], "termos_casados": pontos})
    achados.sort(key=lambda a: -a["termos_casados"])
    return achados[:limite]


def _como_medir(a: dict | None = None) -> str:
    """O texto na lingua pedida. Cai no ingles quando a lingua nao existe.

    As versoes moram em `bench_como_medir.json`, ao lado deste arquivo, e nao aqui dentro: o
    texto e traduzido por gente (tradutor mais revisor nativo) e nao deve exigir editar codigo
    Python para corrigir uma virgula em chines.
    """
    lingua = (a or {}).get("lingua") or "en"
    arq = TOOLS / "bench_como_medir.json"
    if arq.is_file():
        try:
            versoes = json.loads(arq.read_text(encoding="utf-8"))
            if versoes.get(lingua):
                return versoes[lingua]
            if versoes.get("en"):
                return versoes["en"]
        except Exception:  # noqa: BLE001
            pass
    return _como_medir_pt()


def _como_medir_pt() -> str:
    return (
        "COMO ESTA BANCADA MEDE, e o que cada instrumento NAO responde.\n"
        "\n"
        "1. Erro por camada (tools/quant_mixed.py, probe_*_real_acts). Escolhe FORMATO por camada.\n"
        "   Preve o erro de predicao do modelo. NAO preve a imagem final.\n"
        "2. Epsilon por passo com entrada casada (tools/probe_epsilon_per_step.py). Compara dois\n"
        "   CHECKPOINTS. O braco de referencia grava cada (x, timestep) que recebeu via\n"
        "   model_options['model_function_wrapper'] e os quantizados repetem exatamente aqueles,\n"
        "   entao divergencia de trajetoria nao pode existir por construcao.\n"
        "3. Imagem final em trajetoria livre. NAO CARREGA SINAL para comparar duas quantizacoes do\n"
        "   MESMO modelo: com poucos passos uma perturbacao minima desvia o amostrador e o destino\n"
        "   continua sendo uma boa imagem. Serve para 'quebrou ou nao quebrou', nunca para 'qual e\n"
        "   melhor'.\n"
        "\n"
        "REGRAS QUE CUSTARAM CARO AQUI:\n"
        "- Uma execucao nao e uma medicao. Tres corridas na mesma placa ociosa discordaram em ate\n"
        "  1,4x. Repita e publique o espalhamento ao lado da razao.\n"
        "- A/B so vale se os dois bracos tomaram o mesmo caminho de dispatch. A diferenca costuma\n"
        "  ser por qual ramo cada braco passou, nao pelo que voce mudou.\n"
        "- Diga se o resultado foi TRACADO (li o codigo) ou EXECUTADO (rodei). Uma leitura cuidadosa\n"
        "  parece evidencia e le como medicao, e nada na escrita separa as duas.\n"
        "- Ausencia num grep nunca e ausencia no sistema. Confira que o padrao enxerga o que voce\n"
        "  esta negando.\n"
        "- Divergencia de latente NAO tem limiar: 0,8255 destruido contra 0,7173 otimo, medido.\n"
        "- Uma conversao limpa que resolve o backend CUDA prova DESPACHO, nunca qualidade.\n"
    )


def _lock() -> dict:
    f = RAIZ / "GPU_BENCH.lock"
    if not f.is_file():
        return {"livre": True}
    txt = f.read_text(encoding="utf-8", errors="replace")
    d = dict(re.findall(r"^(\w+)=(.*)$", txt, re.M))
    return {"livre": False, "dono": d.get("owner", "?"), "pid": d.get("pid", "?"), "cru": txt[:400]}


# ------------------------------------------------------------------ conversao de verdade

# Raizes onde um caminho pode entrar ou sair. Qualquer coisa fora daqui e recusada -- este
# servidor recebe caminho de quem abriu a pagina e de qualquer IA que fale com o MCP, e um
# conversor escreve gigabytes onde mandarem.
RAIZES = [RAIZ / "ComfyUI/models", RAIZ / "bench", Path("D:/ComfyUI-Models")]

# Toda flag que os sete conversores aceitam, com se ela carrega caminho. Lista fechada: uma flag
# que nao esteja aqui e recusada, em vez de repassada. `--device` fica de fora de proposito --
# quant_int8 roda em CPU por decisao, e nao e a pagina que muda isso.
FLAGS = {
    "--input": "caminho", "--source": "caminho", "--output": "caminho",
    "--calibration": "caminho", "--analysis": "caminho", "--save-analysis": "caminho",
    "--calibrate-with": "nome", "--profile": "valor", "--arch": "valor",
    "--convrot-groupsize": "valor", "--group-size": "valor", "--promote-error": "valor",
    "--keep-bf16-error": "valor", "--budget": "valor", "--uncalibrated": "valor",
    "--sigma-weight": "valor", "--alpha": "valor", "--calibration-tokens": "valor",
    "--limit": "valor", "--verify": "valor", "--reference": "valor",
    "--dry-run": "sozinha", "--auto-detect": "sozinha", "--convrot": "sozinha",
    "--no-convrot": "sozinha", "--no-codebook": "sozinha", "--foreign-analysis": "sozinha",
    "--keep-fused": "sozinha",
}

TRABALHOS: dict[str, dict] = {}


def _dentro_das_raizes(bruto: str) -> Path:
    p = Path(bruto)
    if not p.is_absolute():
        p = (RAIZ / p)
    p = p.resolve()
    for r in RAIZES:
        try:
            p.relative_to(r.resolve())
            return p
        except (ValueError, OSError):
            continue
    raise ValueError(f"caminho fora das raizes permitidas: {p}\npermitidas: "
                     + ", ".join(str(r) for r in RAIZES))


def _linha_de_comando(subcomando: str, flags: dict) -> list[str]:
    """Monta o argv, recusando o que nao reconhece. Nunca shell, nunca string."""
    import convert
    if subcomando not in convert.COMANDOS:
        raise ValueError(f"subcomando desconhecido: {subcomando!r}. "
                         f"Conhecidos: {', '.join(convert.COMANDOS)}")
    argv = [sys.executable, "-s", str(TOOLS / "convert.py"), subcomando]
    for chave, valor in flags.items():
        chave = chave if chave.startswith("--") else f"--{chave}"
        tipo = FLAGS.get(chave)
        if tipo is None:
            raise ValueError(f"flag nao permitida por este servidor: {chave}. "
                             "Se ela existe no conversor, adicione-a a FLAGS em bench_server.py "
                             "conscientemente -- a lista e fechada de proposito.")
        if tipo == "sozinha":
            if valor in (True, "true", "1", "on", "sim", ""):
                argv.append(chave)
            continue
        if valor in (None, ""):
            continue
        if tipo == "caminho":
            argv += [chave, str(_dentro_das_raizes(str(valor)))]
        else:
            v = str(valor)
            if v.startswith("-"):
                raise ValueError(f"valor suspeito para {chave}: {v!r}")
            argv += [chave, v]
    return argv


def _converter(a: dict) -> dict:
    """Roda um conversor de verdade, em subprocesso, e devolve o id do trabalho.

    Subprocesso e nao chamada direta por tres motivos, nesta ordem: uma conversao leva horas e nao
    pode segurar o laco do servidor; ela pode morrer de OOM e nao pode levar o servidor junto; e e
    literalmente a mesma linha de comando que uma pessoa digitaria, entao o que a pagina faz e o
    que o terminal faz.
    """
    subcomando = a.get("subcomando", "")
    flags = a.get("flags") or {}
    if isinstance(flags, str):
        flags = json.loads(flags)
    seco = str(flags.get("--dry-run", flags.get("dry_run", ""))).lower() in ("1", "true", "sim", "on")

    argv = _linha_de_comando(subcomando, flags)

    # Um ensaio nao escreve nada, entao passa mesmo em modo somente-leitura. Uma conversao de
    # verdade exige a flag do dono.
    if not seco and not PERMITIR_ESCRITA:
        return {"recusado": True,
                "porque": "Este servidor esta em modo somente-leitura. Um ensaio (--dry-run) "
                          "roda; uma conversao de verdade exige que o dono reinicie com "
                          "--permitir-escrita. Uma conversao consome horas de GPU e escreve "
                          "gigabytes, e isso nao deve comecar porque alguem clicou.",
                "linha_de_comando": " ".join(argv[2:])}

    lock = _lock()
    if not seco and not lock["livre"]:
        return {"recusado": True,
                "porque": f"O lock da GPU esta com {lock['dono']!r} (pid {lock['pid']}). "
                          "Uma conversao disputaria a placa com quem esta medindo. Espere ou "
                          "fale com o dono do lock.",
                "linha_de_comando": " ".join(argv[2:])}

    tid = uuid.uuid4().hex[:12]
    TRABALHOS[tid] = {"id": tid, "subcomando": subcomando, "seco": seco,
                      "linha_de_comando": " ".join(argv[2:]), "estado": "rodando",
                      "saida": [], "codigo": None}

    async def roda() -> None:
        t = TRABALHOS[tid]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(RAIZ), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            t["pid"] = proc.pid
            assert proc.stdout is not None
            async for linha in proc.stdout:
                texto = linha.decode("utf-8", "replace").rstrip()
                t["saida"].append(texto)
                del t["saida"][:-400]      # so a cauda; uma conversao imprime uma linha por camada
            t["codigo"] = await proc.wait()
            t["estado"] = "terminou" if t["codigo"] == 0 else "falhou"
        except Exception as exc:  # noqa: BLE001
            t["estado"] = "falhou"
            t["saida"].append(f"{type(exc).__name__}: {exc}")
            t["codigo"] = -1

    asyncio.get_running_loop().create_task(roda())
    return {"id": tid, "estado": "rodando", "seco": seco,
            "linha_de_comando": " ".join(argv[2:]),
            "como_acompanhar": f"estado_do_trabalho com id={tid}"}


def _estado_trabalho(a: dict) -> dict:
    tid = a.get("id", "")
    if not tid:
        return {"trabalhos": [{k: t[k] for k in ("id", "subcomando", "estado", "codigo", "seco")}
                              for t in TRABALHOS.values()]}
    t = TRABALHOS.get(tid)
    if t is None:
        return {"erro": f"trabalho desconhecido: {tid}"}
    linhas = int(a.get("linhas", 40))
    return {**{k: t[k] for k in ("id", "subcomando", "estado", "codigo", "seco",
                                 "linha_de_comando")},
            "saida": t["saida"][-linhas:], "linhas_guardadas": len(t["saida"])}


FERRAMENTAS = [
    {"name": "converter", "escreve": True,
     "description": "Roda um conversor de verdade, em subprocesso, e devolve o id do trabalho. "
                    "Um ensaio (flag --dry-run) roda sempre; uma conversao real exige que o dono "
                    "tenha iniciado o servidor com --permitir-escrita, e o lock da GPU livre. "
                    "Chame flags_do_conversor antes para saber as flags daquele subcomando.",
     "inputSchema": {"type": "object", "required": ["subcomando"], "properties": {
         "subcomando": {"type": "string",
                        "description": "w4a4, w4a8, int8, mixed, smooth, native ou from-svdq"},
         "flags": {"type": "object",
                   "description": "as flags, por exemplo {\"--input\": \"...\", "
                                  "\"--dry-run\": true}. Caminhos precisam estar sob "
                                  "ComfyUI/models, bench ou D:/ComfyUI-Models."}}},
     "fn": _converter},
    {"name": "estado_do_trabalho", "escreve": False,
     "description": "Estado e cauda da saida de uma conversao. Sem id, lista todos.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "linhas": {"type": "integer", "default": 40}}},
     "fn": _estado_trabalho},
    {"name": "listar_conversores", "escreve": False,
     "description": "Lista os conversores de checkpoint desta bancada e o que cada um faz. "
                    "Comece por aqui.",
     "inputSchema": {"type": "object", "properties": {}},
     "fn": lambda a: _conversores()},
    {"name": "flags_do_conversor", "escreve": False,
     "description": "As flags exatas de um conversor, tiradas do --help dele agora (nao de "
                    "documentacao que pode estar velha).",
     "inputSchema": {"type": "object", "required": ["subcomando"],
                     "properties": {"subcomando": {"type": "string",
                                                   "description": "w4a4, w4a8, int8, mixed, "
                                                                  "smooth, native ou from-svdq"}}},
     "fn": lambda a: _flags(a.get("subcomando", ""))},
    {"name": "listar_arquivos", "escreve": False,
     "description": "Todo .safetensors sob as raizes que este servidor aceita, com tamanho e se "
                    "ja esta quantizado. Use para descobrir o que existe antes de converter.",
     "inputSchema": {"type": "object", "properties": {
         "filtro": {"type": "string", "description": "substring do caminho relativo"}}},
     "fn": _arquivos},
    {"name": "listar_checkpoints", "escreve": False,
     "description": "Todo checkpoint em disco, com se e quantizado, quantas camadas e em que "
                    "formato. Le so o cabecalho.",
     "inputSchema": {"type": "object", "properties": {}},
     "fn": lambda a: _checkpoints()},
    {"name": "buscar_achados", "escreve": False,
     "description": "Procura nos documentos de medicao desta bancada. Use ANTES de medir algo: "
                    "provavelmente ja foi medido, e a ressalva importa tanto quanto o numero.",
     "inputSchema": {"type": "object", "required": ["consulta"],
                     "properties": {"consulta": {"type": "string"},
                                    "limite": {"type": "integer", "default": 12}}},
     "fn": lambda a: _achados(a.get("consulta", ""), int(a.get("limite", 12)))},
    {"name": "como_medir", "escreve": False,
     "description": "As tres formas de medir desta bancada, qual pergunta cada uma responde, e "
                    "as regras que ja custaram trabalho perdido aqui. Leia antes de concluir algo.",
     "inputSchema": {"type": "object", "properties": {
         "lingua": {"type": "string", "description": "en, pt, es ou zh. Default en."}}},
     "fn": _como_medir},
    {"name": "estado_do_servidor", "escreve": False,
     "description": "Se este servidor esta em modo somente-leitura ou com escrita liberada, e "
                    "quais raizes de caminho ele aceita.",
     "inputSchema": {"type": "object", "properties": {}},
     "fn": lambda a: {"permitir_escrita": PERMITIR_ESCRITA,
                      "raizes": [str(r) for r in RAIZES],
                      "conversores": len(_conversores())}},
    {"name": "estado_da_gpu", "escreve": False,
     "description": "Se o lock da GPU compartilhada esta livre, e de quem esta se nao.",
     "inputSchema": {"type": "object", "properties": {}},
     "fn": lambda a: _lock()},
]


# ------------------------------------------------------------------ MCP

def _resultado(valor) -> dict:
    txt = valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False, indent=2)
    return {"content": [{"type": "text", "text": txt}]}


async def mcp(request: web.Request) -> web.Response:
    try:
        corpo = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"jsonrpc": "2.0", "id": None,
                                  "error": {"code": -32700, "message": "Parse error"}})

    def responde(resultado, id_):
        return web.json_response({"jsonrpc": "2.0", "id": id_, "result": resultado},
                                 headers=cabecalhos)

    metodo, id_ = corpo.get("method"), corpo.get("id")
    params = corpo.get("params") or {}
    cabecalhos = {}

    if metodo == "initialize":
        sid = uuid.uuid4().hex
        SESSOES[sid] = {"iniciado": True}
        cabecalhos["Mcp-Session-Id"] = sid
        return responde({
            "protocolVersion": PROTOCOLO,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "bancada", "version": "1.0.0"},
            "instructions": (
                "Bancada de quantizacao W4A4/W4A8 do ComfyUI. Chame `como_medir` antes de tirar "
                "conclusao de qualquer numero, e `buscar_achados` antes de medir algo novo -- "
                "quase tudo ja foi medido uma vez, e a ressalva importa tanto quanto o numero."),
        }, id_)

    if metodo in ("notifications/initialized", "notifications/cancelled"):
        return web.Response(status=202)

    if metodo == "ping":
        return responde({}, id_)

    if metodo == "tools/list":
        return responde({"tools": [{k: t[k] for k in ("name", "description", "inputSchema")}
                                   for t in FERRAMENTAS]}, id_)

    if metodo == "tools/call":
        nome = params.get("name")
        args = params.get("arguments") or {}
        alvo = next((t for t in FERRAMENTAS if t["name"] == nome), None)
        if alvo is None:
            return web.json_response({"jsonrpc": "2.0", "id": id_,
                                      "error": {"code": -32602,
                                                "message": f"ferramenta desconhecida: {nome}"}})
        if alvo["escreve"] and not PERMITIR_ESCRITA:
            return responde(_resultado(
                "RECUSADO: esta ferramenta escreve, e o servidor esta em modo somente-leitura. "
                "O dono precisa reiniciar com --permitir-escrita. Isso e de proposito: um "
                "conversor consome horas de GPU e escreve gigabytes."), id_)
        try:
            return responde(_resultado(alvo["fn"](args)), id_)
        except Exception as exc:  # noqa: BLE001
            return responde({"content": [{"type": "text",
                                          "text": f"{type(exc).__name__}: {exc}"}],
                             "isError": True}, id_)

    if metodo in ("resources/list", "prompts/list"):
        return responde({"resources": [], "prompts": []}, id_)

    return web.json_response({"jsonrpc": "2.0", "id": id_,
                              "error": {"code": -32601, "message": f"metodo: {metodo}"}})


# ------------------------------------------------------------------ a tela

async def pagina(request: web.Request) -> web.Response:
    return web.Response(text=(TOOLS / "bench_server.html").read_text(encoding="utf-8"),
                        content_type="text/html")


async def api(request: web.Request) -> web.Response:
    """A pagina fala por aqui. GET para leitura, POST quando ha corpo (flags e um objeto).

    O portao de escrita e o MESMO que o MCP usa -- checado aqui tambem, e nao so na camada MCP,
    porque a pagina e o MCP sao duas portas para as mesmas funcoes e uma delas nao pode ser a
    frouxa.
    """
    nome = request.match_info["nome"]
    alvo = next((t for t in FERRAMENTAS if t["name"] == nome), None)
    if alvo is None:
        return web.json_response({"ok": False, "erro": f"ferramenta desconhecida: {nome}"},
                                 status=404)
    if request.method == "POST":
        try:
            args = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"ok": False, "erro": "corpo nao e JSON"}, status=400)
    else:
        args = dict(request.query)
    if not isinstance(args, dict):
        return web.json_response({"ok": False, "erro": "argumentos precisam ser um objeto"},
                                 status=400)
    try:
        valor = alvo["fn"](args)
        if asyncio.iscoroutine(valor):
            valor = await valor
        return web.json_response({"ok": True, "valor": valor})
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"ok": False, "erro": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    global PERMITIR_ESCRITA
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--porta", type=int, default=8199)
    p.add_argument("--host", default="127.0.0.1",
                   help="so mude com muito motivo: nao ha autenticacao nenhuma aqui")
    p.add_argument("--permitir-escrita", action="store_true",
                   help="libera ferramentas que escrevem. Desligado por padrao de proposito.")
    a = p.parse_args()
    PERMITIR_ESCRITA = a.permitir_escrita

    app = web.Application()
    app.router.add_get("/", pagina)
    app.router.add_post("/mcp", mcp)
    app.router.add_get("/mcp", lambda r: web.Response(status=405, text="POST"))
    app.router.add_get("/api/{nome}", api)
    app.router.add_post("/api/{nome}", api)

    print(f"tela  http://{a.host}:{a.porta}/")
    print(f"mcp   http://{a.host}:{a.porta}/mcp")
    print(f"      claude mcp add --transport http bancada http://{a.host}:{a.porta}/mcp")
    print(f"escrita: {'LIBERADA' if PERMITIR_ESCRITA else 'bloqueada (somente-leitura)'}")
    print(f"{len(FERRAMENTAS)} ferramentas. Ctrl-C para parar.")
    try:
        web.run_app(app, host=a.host, port=a.porta, print=None)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
