"""Tests for `comfy_client.py` against a FAKE ComfyUI (stdlib `http.server` on a high loopback
port). No ComfyUI, no GPU, no lock.

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s tools/test_comfy_client.py

Same runner convention as the other `test_*.py` here (no pytest in the embedded interpreter).

WHAT THIS PROVES: the HTTP contract this client relies on -- node_errors on a 200 refuses, a
/history that never answers times out, a worker's /history is actually polled (the
`http://http://` defect), records and exit codes per outcome, and that the module imports with
the standard library alone.
WHAT IT DOES NOT: that real ComfyUI still answers in these shapes, or anything about MultiGPU
beyond reading a command line and polling a second port.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import comfy_client as cc  # noqa: E402


class Falso:
    """One fake ComfyUI. `historico` is what /history/<pid> answers once `apos` polls passed."""

    def __init__(self, *, resposta_prompt=None, codigo_prompt=200, historico=None, apos=0):
        self.resposta_prompt = resposta_prompt if resposta_prompt is not None else {
            "prompt_id": "pid-1", "number": 1, "node_errors": {}}
        self.codigo_prompt = codigo_prompt
        self.historico = historico
        self.apos = apos
        self.polls = 0
        self.prompts: list[dict] = []
        falso = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                corpo = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)

            def do_GET(self):
                if self.path == "/system_stats":
                    return self._json(200, {"system": {"argv": ["main.py", "--port", "x"],
                                                       "comfyui_version": "0.0-falso",
                                                       "pytorch_version": "falso"},
                                            "devices": [{"type": "cuda", "index": 0,
                                                         "name": "Placa Falsa", "vram_total": 2**34, "vram_free": 2**33}]})
                if self.path == "/queue":
                    return self._json(200, {"queue_running": [], "queue_pending": []})
                if self.path.startswith("/history/"):
                    pid = self.path.rsplit("/", 1)[1]
                    falso.polls += 1
                    if falso.historico is not None and falso.polls > falso.apos:
                        return self._json(200, {pid: falso.historico})
                    return self._json(200, {})
                self._json(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                falso.prompts.append(json.loads(self.rfile.read(n) or b"{}"))
                self._json(falso.codigo_prompt, falso.resposta_prompt)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.porta = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"127.0.0.1:{self.porta}"     # sem esquema, de proposito (o caso do --server)

    def fecha(self):
        self.srv.shutdown()
        self.srv.server_close()


def hist(status="success", segundos=12.0, arquivos=("a_00001_.png",)):
    msgs = [["execution_start", {"timestamp": 1_000_000}]]
    if status == "success":
        msgs.append(["execution_success", {"timestamp": 1_000_000 + int(segundos * 1000)}])
    else:
        msgs.append(["execution_error", {"node_id": "6", "node_type": "KSampler",
                                         "exception_type": "RuntimeError",
                                         "exception_message": "boom"}])
    return {"status": {"status_str": status, "completed": status == "success", "messages": msgs},
            "outputs": {"9": {"images": [{"filename": f, "subfolder": "sub", "type": "output"}
                                         for f in arquivos]}}}


def _sem_workers(fn):
    salvo = cc.bases_irmas
    cc.bases_irmas = lambda base: [cc.normaliza_base(base)]
    try:
        return fn()
    finally:
        cc.bases_irmas = salvo


def test_normaliza_base():
    assert cc.normaliza_base("127.0.0.1:8190") == "http://127.0.0.1:8190"
    assert cc.normaliza_base("http://127.0.0.1:8190/") == "http://127.0.0.1:8190"
    assert cc.Comfy("127.0.0.1:1").base == cc.Comfy("http://127.0.0.1:1").base


def test_submit_levanta_em_node_errors_com_http_200():
    f = Falso(resposta_prompt={"prompt_id": "pid-x", "number": 1,
                               "node_errors": {"8": {"class_type": "SaveImage", "errors": []}}})
    try:
        try:
            cc.Comfy(f.base).submit({"1": {}}, "c")
        except cc.PromptRefused as e:
            assert "node_errors" in str(e)
        else:
            raise AssertionError("node_errors num 200 passou calado")
    finally:
        f.fecha()


def test_submit_ok_devolve_pid():
    f = Falso()
    try:
        assert cc.Comfy(f.base).submit({"1": {}}, "c") == "pid-1"
    finally:
        f.fecha()


def test_run_and_wait_estoura_prazo():
    f = Falso(historico=None)
    try:
        try:
            _sem_workers(lambda: cc.run_and_wait(cc.Comfy(f.base), {}, 1.0, poll_s=0.1))
        except cc.PollTimeout as e:
            assert e.pid == "pid-1"
        else:
            raise AssertionError("sem PollTimeout")
        assert f.polls >= 3, f.polls
    finally:
        f.fecha()


def test_run_and_wait_recusa_prazo_nulo():
    for t in (0, -1, None):
        try:
            cc.run_and_wait(cc.Comfy("127.0.0.1:1"), {}, t)
        except ValueError:
            continue
        raise AssertionError(f"timeout={t} aceito")


def test_run_and_wait_sucesso_e_tick():
    f = Falso(historico=hist(segundos=12.0), apos=2)
    ticks = []
    try:
        e = _sem_workers(lambda: cc.run_and_wait(cc.Comfy(f.base), {}, 10.0, poll_s=0.05,
                                                 tick=ticks.append))
        assert e.status == "success" and e.server_side_s == 12.0 and not e.cache_hit
        assert e.files == ["sub/a_00001_.png"] and e.erro is None
        assert len(ticks) >= 3, ticks
    finally:
        f.fecha()


def test_cache_hit_pelo_tempo_do_servidor():
    e = cc.Entry(pid="p", hist=hist(segundos=0.02), wall=9.0)
    assert e.cache_hit
    e = cc.Entry(pid="p", hist=hist(segundos=30.0), wall=0.5)
    assert not e.cache_hit


def test_cache_hit_pela_lista_de_nos_em_cache():
    """Com `execution_cached` e as saídas do prompt, o tempo não decide: um render rápido de
    verdade (2,8 s, KSampler fora do cache) não é hit; tudo em cache é hit mesmo se lento."""
    def com_cache(segundos, nos):
        h = hist(segundos=segundos)
        h["status"]["messages"].insert(1, ["execution_cached", {"nodes": nos, "timestamp": 0}])
        h["prompt"] = [0, "p", {}, {}, ["9"]]
        return h
    assert not cc.Entry(pid="p", hist=com_cache(2.8, ["1", "2", "3"]), wall=3.0).cache_hit
    assert cc.Entry(pid="p", hist=com_cache(6.0, ["1", "2", "3", "9"]), wall=6.5).cache_hit


def test_worker_multigpu_e_consultado_de_fato():
    """O pai nunca tem o /history; o worker tem. Antes da normalizacao o cliente do worker
    montava `http://http://...` e o prompt estourava o prazo com a imagem pronta."""
    import psutil
    pai = Falso(historico=None)
    worker = Falso(historico=hist())

    class P:
        def __init__(self, argv):
            self.info = {"cmdline": argv}
    procs = [P(["python", "main.py", "--port", str(pai.porta)]),
             P(["python", "main.py", "--port", str(worker.porta), "--cuda-device", "1"])]
    original = psutil.process_iter
    psutil.process_iter = lambda attrs=None: iter(procs)
    try:
        e = cc.run_and_wait(cc.Comfy(pai.base), {}, 5.0, poll_s=0.05)
        assert e.onde == f"http://127.0.0.1:{worker.porta}", e.onde
    finally:
        psutil.process_iter = original
        pai.fecha()
        worker.fecha()


def test_roda_um_registra_cada_desfecho():
    ok = Falso(historico=hist())
    erro = Falso(historico=hist(status="error"))
    recusa = Falso(resposta_prompt={"error": {"type": "x"}, "node_errors": {}}, codigo_prompt=400)
    parado = Falso(historico=None)
    try:
        r = _sem_workers(lambda: cc.roda_um(cc.Comfy(ok.base), {}, 5, rotulo="g"))
        assert r["status"] == "success" and r["prompt_id"] == "pid-1" and r["server_side_s"] == 12.0
        r = _sem_workers(lambda: cc.roda_um(cc.Comfy(erro.base), {}, 5, rotulo="g"))
        assert r["status"] == "error" and "boom" in r["erro"], r
        r = _sem_workers(lambda: cc.roda_um(cc.Comfy(recusa.base), {}, 5, rotulo="g"))
        assert r["status"] == "recusado", r
        salvo = cc.run_and_wait
        cc.run_and_wait = lambda c, a, t, tick=None: salvo(c, a, 0.5, tick=tick, poll_s=0.05)
        try:
            r = _sem_workers(lambda: cc.roda_um(cc.Comfy(parado.base), {}, 5, rotulo="g"))
        finally:
            cc.run_and_wait = salvo
        assert r["status"] == "timeout", r
        r = cc.roda_um(cc.Comfy("127.0.0.1:1"), {}, 5, rotulo="g")
        assert r["status"] in ("sem_servidor", "recusado"), r
    finally:
        for f in (ok, erro, recusa, parado):
            f.fecha()


def test_codigo_de_saida():
    assert cc.codigo_de_saida([{"status": "success", "cache_hit": False}]) == 0
    assert cc.codigo_de_saida([{"status": "success", "cache_hit": True}]) == 5
    assert cc.codigo_de_saida([{"status": "success"}, {"status": "timeout"}]) == 1


def test_roda_lista_grava_jsonl_com_controles():
    f = Falso(historico=hist())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "g1.json").write_text("{}", encoding="utf-8")
        (td / "g2.json").write_text("{}", encoding="utf-8")
        (td / "ordem.txt").write_text("g1.json\n\ng2.json\n", encoding="utf-8")
        saida = td / "r.jsonl"
        try:
            rc = _sem_workers(lambda: cc.roda_lista(cc.Comfy(f.base), cc.le_lista(td / "ordem.txt"),
                                                    saida, 5))
        finally:
            f.fecha()
        linhas = [json.loads(x) for x in saida.read_text(encoding="utf-8").splitlines()]
    assert rc == 0 and len(linhas) == 2
    for r in linhas:
        assert r["placa"] == "cuda:0 Placa Falsa" and r["args"][0] == "main.py"
        assert r["files"] == ["sub/a_00001_.png"] and r["prompt_id"] == "pid-1"


def test_comfy_run_workflow_api_prompt_ponta_a_ponta():
    f = Falso(historico=hist())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "g.json").write_text('{"1": {"class_type": "X", "inputs": {}}}', encoding="utf-8")
        cmd = [sys.executable, "-s", str(TOOLS / "comfy_run_workflow.py"), "--server", f.base,
               "--api-prompt", str(td / "g.json"), "--saida", str(td / "r.jsonl"),
               "--timeout", "10", "--wait-server", "5"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        finally:
            f.fecha()
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        r = json.loads((td / "r.jsonl").read_text(encoding="utf-8"))
    assert r["status"] == "success" and r["server_side_s"] == 12.0
    assert f.prompts and f.prompts[0]["prompt"] == {"1": {"class_type": "X", "inputs": {}}}


def _roda_consumidor(argv_fn, historico):
    f = Falso(historico=historico)
    try:
        p = subprocess.run([sys.executable, "-s", *argv_fn(f)], capture_output=True, text=True,
                           timeout=120, check=False)
    finally:
        f.fecha()
    return p


CONSUMIDORES = {
    "ltx25_queue": lambda f: [str(TOOLS / "ltx25_queue.py"), "--server", f.base, "--unet", "x",
                              "--timeout", "20"],
    "ltx_video": lambda f: [str(TOOLS / "ltx_video.py"), "--servidor", f.base, "--transformer", "x",
                            "--limite", "20"],
    "qwen_edit_test": lambda f: [str(TOOLS / "qwen_edit_test.py"), "--servidor", f.base,
                                 "--transformer", "x", "--imagem", "x.png", "--instrucao", "y",
                                 "--limite", "20"],
}


def test_consumidores_saem_0_no_sucesso_e_diferente_de_0_na_falha():
    """ltx25_queue / ltx_video / qwen_edit_test passaram a usar este cliente. ltx_video saia 0
    mesmo com status != success; os tres agora saem != 0 quando o servidor diz erro."""
    for nome, argv in CONSUMIDORES.items():
        ok = _roda_consumidor(argv, hist())
        assert ok.returncode == 0, (nome, ok.stdout[-1500:], ok.stderr[-1500:])
        ruim = _roda_consumidor(argv, hist(status="error"))
        assert ruim.returncode not in (0, None), (nome, ruim.returncode, ruim.stdout[-800:])


def test_so_stdlib():
    """O que `import comfy_client` puxa para `sys.modules`, num processo limpo, e todo stdlib.

    (Um `-S -I` nao serve neste interpretador: o `._pth` do python embutido carrega
    sitecustomize/pywin32 de qualquer jeito, antes do import em teste.)"""
    codigo = (f"import sys; sys.path.insert(0, {str(TOOLS)!r}); antes = set(sys.modules); "
              "import comfy_client; novos = set(sys.modules) - antes; "
              "fora = sorted({m.split('.')[0] for m in novos} - set(sys.stdlib_module_names) "
              "- {'comfy_client'}); print(fora)")
    p = subprocess.run([sys.executable, "-s", "-c", codigo], capture_output=True, text=True,
                       timeout=60, check=False)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "[]", p.stdout


if __name__ == "__main__":
    falhas = 0
    for nome, fn in sorted(globals().items()):
        if not nome.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {nome}")
        except AssertionError as exc:
            falhas += 1
            print(f"FAIL  {nome}: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"FAIL  {nome}: unexpected {exc!r}")
    print("\nNAO COBERTO: ComfyUI real, MultiGPU real, GPU. Servidor falso prova o contrato HTTP.")
    raise SystemExit(1 if falhas else 0)
