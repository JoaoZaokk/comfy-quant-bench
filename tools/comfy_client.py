r"""The one HTTP client for a running ComfyUI server: submit an API-format prompt, poll /history
with a deadline, read the server's own timing, and write one JSONL record per prompt.

STDLIB ONLY, and that is a contract, not a style: this file is copied as-is to a Ubuntu VM that
runs it with the system `python3`. `psutil` is imported lazily and only for MultiGPU worker
discovery (`bases_irmas`); without it that discovery degrades to "the submitted server only".

WHY THIS FILE EXISTS (revisao 2026-09-29, achado 4). `comfy_run_workflow.py` had the careful client
-- deadline, cache-hit detection from the server's own timestamps, MultiGPU workers, a readable 400
-- and no file imported it. Eight scripts wrote their own `POST /prompt` + `while True: GET
/history` instead, each losing a different piece: no deadline (a hung job hangs the battery), exit
code 0 on a failed render, `node_errors` in an HTTP 200 ignored, cache hits recorded as 0.01 s
renders. The client now lives here so the scripts can import it instead of copying it.

    python3 comfy_client.py --server 127.0.0.1:8190 --lista ordem.txt --saida resultados.jsonl
    python3 comfy_client.py --server 127.0.0.1:8188 --api-prompt grafo.json --saida r.jsonl

WHAT THIS DOES NOT CHECK:
  - that a record's timing is a fair comparison with anything: it times one server, one prompt,
    storage and cache state included. `cache_hit` flags the one case known to time nothing.
  - that the outputs are correct. `files` says what the server wrote, not what it shows.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ticket 04. The server's own execution_start -> execution_success span, in seconds, below which
# "outputs came back" means "the per-node result cache answered", not "this rendered". Compared
# against the SERVER's duration (`Entry.server_side_s`), never this client's wall clock, which also
# carries the poll interval and a network round trip.
CACHE_HIT_THRESHOLD_S = 5.0

# The exceptions that mean "no answer this time" rather than "the server said no".
REDE = (urllib.error.URLError, OSError, TimeoutError)


def normaliza_base(base: str) -> str:
    """`127.0.0.1:8190`, `http://127.0.0.1:8190/` -> `http://127.0.0.1:8190`.

    MEDIDO na revisao de 2026-09-29, por leitura: `--server` chegava sem esquema, `_request`
    prefixava `http://`, e `bases_irmas` devolvia os workers JA com `http://`. O cliente do worker
    montava `http://http://127.0.0.1:27716/history/...`, a excecao era engolida pelo poll como
    "ainda nao", e a varredura de workers do MultiGPU nunca consultava worker nenhum. Uma forma so
    de base, decidida aqui, fecha isso."""
    base = base.strip().rstrip("/")
    if "://" not in base:
        base = "http://" + base
    return base


def _request(base: str, path: str, payload: dict | None = None, timeout: float = 30.0) -> Any:
    """The one place an HTTP call to a ComfyUI server is actually made, GET or POST alike.

    A 400 from ComfyUI carries a JSON body naming the node and input it refused; urllib raises
    before that body is read, so this decodes and prints it before re-raising."""
    url = normaliza_base(base) + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(body)
            print(f"  server refused ({e.code}): {err.get('error')}")
            _imprime_node_errors(err.get("node_errors"))
        except json.JSONDecodeError:
            print(f"  server refused ({e.code}): {body[:4000]}")
        raise


def _imprime_node_errors(node_errors: dict | None) -> None:
    for nid, ne in (node_errors or {}).items():
        print(f"    node {nid} ({ne.get('class_type')}):")
        for d in ne.get("errors") or []:
            print(f"      {d.get('type')}: {d.get('message')} -- {d.get('details')}")


def http_get(base: str, path: str, timeout: float = 30.0) -> Any:
    return _request(base, path, payload=None, timeout=timeout)


def http_post(base: str, path: str, payload: dict, timeout: float = 60.0) -> Any:
    return _request(base, path, payload=payload, timeout=timeout)


class PromptRefused(ValueError):
    """The server answered /prompt without error status but did not accept the whole graph."""


class Comfy:
    """The HTTP boundary to one ComfyUI server. No method here calls `sys.exit`; callers decide
    what a failure means for their exit code."""

    def __init__(self, base: str) -> None:
        self.base = normaliza_base(base)

    def wait_up(self, limit_s: float) -> float:
        """Block until /system_stats answers. Returns seconds waited; `TimeoutError` past
        `limit_s`."""
        t0 = time.time()
        while time.time() - t0 < limit_s:
            try:
                http_get(self.base, "/system_stats", timeout=5.0)
                return time.time() - t0
            except REDE:
                time.sleep(2.0)
        raise TimeoutError(f"server {self.base} did not answer /system_stats in {limit_s:.0f}s")

    def object_info(self, timeout: float = 120.0) -> dict:
        return http_get(self.base, "/object_info", timeout=timeout)

    def system_stats(self, timeout: float = 10.0) -> dict:
        return http_get(self.base, "/system_stats", timeout=timeout)

    def submit(self, api_prompt: dict, client_id: str) -> str:
        """POST an API-format prompt. Returns the `prompt_id`.

        Raises `PromptRefused` (a `ValueError`) when the response has no `prompt_id`, AND when it
        has one but also carries a non-empty `node_errors`. `ComfyUI/server.py` answers HTTP 200
        with both when only SOME output nodes validated: the invalid branches are dropped, the rest
        runs, and /history later says "success". Found by `.scratch/roda_eros_2gpu.py` on 26/09
        and never carried back here until the review of 2026-09-29."""
        resp = http_post(self.base, "/prompt", {"prompt": api_prompt, "client_id": client_id})
        pid = resp.get("prompt_id")
        if resp.get("node_errors"):
            print(f"  server accepted prompt {pid} PARTIALLY -- node_errors:")
            _imprime_node_errors(resp["node_errors"])
            raise PromptRefused(
                f"server dropped part of the graph (node_errors on HTTP 200, prompt_id={pid}): "
                f"{json.dumps(resp['node_errors'])[:2000]}")
        if not pid:
            raise PromptRefused(f"server refused the prompt: {json.dumps(resp)[:2000]}")
        return pid

    def history(self, pid: str) -> dict | None:
        """/history/{pid}'s entry, or None if absent -- including a failed request (a network
        hiccup mid-poll is not fatal; the poll tries again next tick)."""
        try:
            hist = http_get(self.base, f"/history/{pid}", timeout=30.0)
        except REDE:
            return None
        return hist.get(pid)

    def queue(self) -> tuple[int, int]:
        """GET /queue -> (running, pending)."""
        q = http_get(self.base, "/queue", timeout=10.0)
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


def bases_irmas(base: str) -> list[str]:
    """Todo servidor ComfyUI deste host, com `base` -- a quem se submeteu -- sempre na frente.

    MEDIDO 2026-09-12, nao deduzido. `ComfyUI-MultiGPU` sobe UM PROCESSO POR DEVICE
    (`main.py --port N --cuda-device K`, porta alta) e o resultado de um prompt aparece no
    `/history` DO WORKER QUE O EXECUTOU, nunca no do pai a quem se submeteu. Um cliente que so
    consulta o pai fica pendurado ate o timeout enquanto a imagem ja esta gravada em disco -- e
    reporta TIMEOUT, que le como "o modelo nao gerou".

    Le a LINHA DE COMANDO dos processos em vez de assumir portas (a porta do worker e escolhida em
    tempo de execucao), e e chamada DENTRO do laco de poll: o worker nasce sob demanda.

    Sem `psutil` devolve so `[base]` -- degrada em vez de morrer (este arquivo e so stdlib).
    Todas as bases saem normalizadas (`normaliza_base`), para que o pai achado na varredura nao
    vire um segundo cliente, e o worker nao vire `http://http://...`.
    """
    base = normaliza_base(base)
    achadas: list[str] = [base]
    try:
        import psutil
    except ImportError:
        return achadas
    try:
        procs = list(psutil.process_iter(["cmdline"]))
    except Exception:  # noqa: BLE001 -- varredura de processos e best-effort, nunca fatal
        return achadas
    lidos = ilegiveis = 0
    for proc in procs:
        try:
            argv = proc.info.get("cmdline") or []
        # A falha individual e normal (o processo morre entre listar e ler). O que importa e a
        # CEGUEIRA TOTAL, reportada uma vez depois do laco.
        except Exception:  # noqa: BLE001
            ilegiveis += 1
            continue
        lidos += 1
        if not any(a.endswith("main.py") for a in argv):
            continue
        if "--port" not in argv or argv.index("--port") + 1 >= len(argv):
            continue
        porta = argv[argv.index("--port") + 1]
        if not porta.isdigit():
            continue
        # `--listen` do worker e sempre 127.0.0.1 nesta bancada.
        alvo = f"http://127.0.0.1:{porta}"
        if alvo not in achadas:
            achadas.append(alvo)
    if lidos == 0 and ilegiveis:
        print(f"  aviso: nenhum dos {ilegiveis} processos teve a linha de comando lida -- a "
              "descoberta de workers esta CEGA, nao vazia; so o servidor submetido sera "
              "consultado, e um resultado que caia num worker vai parecer TIMEOUT")
    return achadas


class PollTimeout(Exception):
    """`run_and_wait` did not see `pid` in /history within `timeout_s` seconds."""

    def __init__(self, pid: str, timeout_s: float) -> None:
        self.pid = pid
        self.timeout_s = timeout_s
        super().__init__(f"TIMEOUT after {timeout_s:.0f}s -- prompt {pid} still not in /history")


@dataclass
class Entry:
    """One /history entry for a submitted prompt, plus this client's wall clock for the same span
    (`wall`: POST -> /history first showing `pid`, padded by the poll interval). Everything else is
    derived from the raw `hist` on read, so each quantity has exactly one definition."""

    pid: str
    hist: dict
    wall: float
    onde: str = ""   # base whose /history had the entry (a MultiGPU worker, possibly)

    @property
    def status(self) -> str:
        return (self.hist.get("status") or {}).get("status_str", "?")

    @property
    def files(self) -> list[str]:
        outputs = self.hist.get("outputs") or {}
        return [
            f"{v.get('subfolder', '')}/{v.get('filename', '')}"
            for out in outputs.values()
            for key in ("images", "gifs", "audio", "video")
            for v in (out.get(key) or [])
        ]

    def _mensagens(self) -> list:
        return (self.hist.get("status") or {}).get("messages") or []

    @property
    def server_side_s(self) -> float | None:
        """The server's own execution_start -> execution_success span, in seconds (ticket 04)."""
        starts: dict[str, float] = {}
        server_side_s: float | None = None
        for m in self._mensagens():
            if not (isinstance(m, (list, tuple)) and len(m) >= 2):
                continue
            kind, data = m[0], m[1]
            if kind == "execution_start":
                starts["_t0"] = data.get("timestamp")
            elif kind == "execution_success" and starts.get("_t0") and data.get("timestamp"):
                server_side_s = (data["timestamp"] - starts["_t0"]) / 1000.0
        return server_side_s

    @property
    def cache_hit(self) -> bool:
        """True when the server executed nothing: every output node of the prompt is in its own
        `execution_cached` list. A 4-step render can finish in under 3 s, so time alone flags fast
        renders as hits; the time threshold (on `server_side_s`, never `wall`) is only the fallback
        for a history entry without the cached-node list or the prompt's outputs."""
        cached = next((set((m[1] or {}).get("nodes") or []) for m in self._mensagens()
                       if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_cached"), None)
        prompt = self.hist.get("prompt")
        saidas = prompt[4] if isinstance(prompt, (list, tuple)) and len(prompt) > 4 else None
        if cached is not None and saidas:
            return bool(self.files) and all(str(s) in cached for s in saidas)
        return (bool(self.files) and self.server_side_s is not None
                and self.server_side_s < CACHE_HIT_THRESHOLD_S)

    @property
    def erro(self) -> str | None:
        """The server's execution_error, condensed, or None."""
        for m in self._mensagens():
            if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
                d = m[1] or {}
                return (f"{d.get('node_type', '?')} (node {d.get('node_id', '?')}): "
                        f"{d.get('exception_type', '')} {d.get('exception_message', '')}"
                        ).strip()[:1500]
        if self.status != "success":
            return json.dumps(self.hist.get("status"))[:1500]
        return None


def run_and_wait(comfy: Comfy, api_prompt: dict, timeout: float, *,
                 tick: Callable[[float], None] | None = None,
                 poll_s: float = 2.0) -> Entry:
    """Submit `api_prompt` and poll /history -- the submitted server and any MultiGPU worker --
    until the entry appears. There is no unbounded form: `timeout` is required and `PollTimeout`
    is raised past it. `tick(elapsed_s)` runs once per poll (the battery samples GPU/RAM there).

    Raises `PromptRefused` from `submit`, `urllib.error.HTTPError` on a 400, `PollTimeout`."""
    if not timeout or timeout <= 0:
        raise ValueError("run_and_wait needs a positive timeout; there is no unbounded poll")
    t0 = time.time()
    pid = comfy.submit(api_prompt, str(uuid.uuid4()))
    print(f"queued prompt_id={pid}", flush=True)

    last_note = 0.0
    ultima_varredura = -1e9
    clientes: dict[str, Comfy] = {comfy.base: comfy}
    while True:
        agora = time.time()
        if agora - t0 > timeout:
            raise PollTimeout(pid, timeout)
        if tick is not None:
            tick(agora - t0)
        if agora - ultima_varredura >= 10.0:
            ultima_varredura = agora
            for b in bases_irmas(comfy.base):
                clientes.setdefault(b, Comfy(b))
        for cliente in list(clientes.values()):
            hist_entry = cliente.history(pid)
            if hist_entry is not None:
                if cliente.base != comfy.base:
                    print(f"  resultado veio de {cliente.base}, nao de {comfy.base} "
                          f"(worker do ComfyUI-MultiGPU)")
                return Entry(pid=pid, hist=hist_entry, wall=time.time() - t0, onde=cliente.base)
        if agora - last_note >= 30.0:
            last_note = agora
            try:
                running, pending = comfy.queue()
                print(f"  [{agora - t0:6.1f}s] running={running} pending={pending}", flush=True)
            except REDE:
                print(f"  [{agora - t0:6.1f}s] waiting", flush=True)
        time.sleep(poll_s)


def controles(comfy: Comfy) -> dict:
    """What the server says about itself, for each record: the card(s), its argv, versions.

    The 3080 Ti incident turned on a record that did not say which card ran the work; a timing
    without this block is not comparable with anything. Failure to read is recorded, not raised."""
    try:
        st = comfy.system_stats()
    except (*REDE, ValueError) as e:
        return {"placa": None, "args": None, "controles_erro": f"{type(e).__name__}: {e}"[:300]}
    sistema = st.get("system") or {}
    placas = [f"{d.get('type')}:{d.get('index')} {d.get('name')}" for d in st.get("devices") or []]
    return {"placa": placas[0] if placas else None, "placas": placas,
            "args": sistema.get("argv"), "comfyui_version": sistema.get("comfyui_version"),
            "pytorch_version": sistema.get("pytorch_version")}


def grava_jsonl(caminho: Path, registro: dict) -> None:
    """Append one record and flush: a battery that dies halfway keeps what it measured."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        f.flush()


def roda_um(comfy: Comfy, api_prompt: dict, timeout: float, *, rotulo: str,
            extra: dict | None = None, tick: Callable[[float], None] | None = None) -> dict:
    """Submit, wait, and turn every outcome -- success, execution error, refusal, timeout -- into
    one record. Never raises for those outcomes; `registro["status"]` says which it was."""
    reg: dict = {"grafo": rotulo, "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "servidor": comfy.base}
    try:
        e = run_and_wait(comfy, api_prompt, timeout, tick=tick)
    except PollTimeout as exc:
        reg.update(status="timeout", prompt_id=exc.pid, erro=str(exc))
    except urllib.error.HTTPError as exc:
        reg.update(status="recusado", erro=f"HTTP {exc.code}")
    except PromptRefused as exc:
        reg.update(status="recusado", erro=str(exc)[:1500])
    except REDE as exc:
        reg.update(status="sem_servidor", erro=f"{type(exc).__name__}: {exc}"[:500])
    else:
        reg.update(status=e.status, prompt_id=e.pid, erro=e.erro, server_side_s=e.server_side_s,
                   wall=round(e.wall, 3), cache_hit=e.cache_hit, files=e.files, onde=e.onde)
    reg.update(extra or {})
    return reg


def codigo_de_saida(registros: list[dict]) -> int:
    """0 all rendered; 1 any failure/refusal/timeout; 5 all ran but some were cache hits."""
    if any(r.get("status") != "success" for r in registros):
        return 1
    if any(r.get("cache_hit") for r in registros):
        return 5
    return 0


def roda_lista(comfy: Comfy, grafos: list[Path], saida: Path | None, timeout: float) -> int:
    """Run API-format graph files in order, one JSONL record each. Returns the exit code."""
    ctl = controles(comfy)
    registros = []
    for g in grafos:
        print(f"=== {g} {time.strftime('%H:%M:%S')}", flush=True)
        api = json.loads(Path(g).read_text(encoding="utf-8-sig"))
        reg = roda_um(comfy, api, timeout, rotulo=str(g), extra=ctl)
        print(f"  {reg['status']} server_side_s={reg.get('server_side_s')} "
              f"cache_hit={reg.get('cache_hit')} {reg.get('erro') or ''}"[:600], flush=True)
        if saida is not None:
            grava_jsonl(saida, reg)
        registros.append(reg)
    rc = codigo_de_saida(registros)
    falhas = sum(r["status"] != "success" for r in registros)
    print(f"{len(registros)} grafo(s), {falhas} falha(s), exit {rc}")
    return rc


def le_lista(ordem: Path) -> list[Path]:
    """Lines of an `ordem.txt`, resolved against the file's own directory."""
    ordem = Path(ordem)
    return [(ordem.parent / linha.strip()) if not Path(linha.strip()).is_absolute()
            else Path(linha.strip())
            for linha in ordem.read_text(encoding="utf-8-sig").splitlines() if linha.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run API-format ComfyUI graphs with a deadline.")
    ap.add_argument("--server", default="127.0.0.1:8190")
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--api-prompt", type=Path, help="one API-format graph JSON")
    grupo.add_argument("--lista", type=Path,
                       help="text file, one graph path per line (relative to the list's folder)")
    ap.add_argument("--saida", type=Path, help="append one JSONL record per graph here")
    ap.add_argument("--timeout", type=float, default=5400.0, help="per graph, seconds")
    ap.add_argument("--wait-server", type=float, default=900.0)
    a = ap.parse_args(argv)
    comfy = Comfy(a.server)
    try:
        comfy.wait_up(a.wait_server)
    except TimeoutError as e:
        print(str(e))
        return 2
    grafos = [a.api_prompt] if a.api_prompt else le_lista(a.lista)
    return roda_lista(comfy, grafos, a.saida, a.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
