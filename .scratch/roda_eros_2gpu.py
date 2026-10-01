"""Enfileira um grafo de API no ComfyUI 8190 e amostra as duas placas e a RAM a cada 5 s ate terminar.
Nao abre video nem audio. Uso:
    python -s .scratch/roda_eros_2gpu.py <grafo_api.json> <saida.csv> [<resultados.jsonl>]

Revisao 2026-09-29: submissao e espera pelo cliente canonico (`tools/comfy_client.py`). Antes o laco
do render era `while True` sem prazo, e uma execucao com status != success saia com codigo 0 -- o
`roda_bateria.ps1` so via erro em validacao. Agora: prazo por grafo (`PRAZO_GRAFO_S`, padrao 5400 s),
node_errors num 200 recusa, exit 0 renderizou / 1 falhou, recusou ou estourou / 5 cache hit, e um
registro por prompt_id no JSONL (grafo, status, server_side_s, wall, cache_hit, files, placa, args)."""
import csv, ctypes, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import comfy_client as cc  # noqa: E402

BASE = f"http://127.0.0.1:{os.environ.get('COMFY_PORT', '8190')}"  # porta configuravel: um processo zumbi prendeu a 8190 (27/09)


class MS(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("totpf", ctypes.c_ulonglong), ("availpf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]


def ram():
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.tot - m.avail) / 2**30, (m.totpf - m.availpf) / 2**30


def gpus():
    # com o commit do Windows no limite o CreateProcess falha (WinError 1455, 26/09); amostra perdida, run segue
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return [x.split(", ") for x in o] + [["", ""], ["", ""]]
    except (OSError, subprocess.TimeoutExpired):  # nvidia-smi pendurado nao pode pendurar a bateria
        return [["", ""], ["", ""]]


_proc = [None]


def comfy_mem():
    """RSS e memoria privada (commit) do processo que escuta a 8190, em GiB. Vazio se nao achar."""
    try:
        import psutil
        p = _proc[0]
        if p is None or not p.is_running():
            pid = next(c.pid for c in psutil.net_connections("tcp")
                       if c.laddr and c.laddr.port == int(os.environ.get("COMFY_PORT", "8190")) and c.status == psutil.CONN_LISTEN)
            p = _proc[0] = psutil.Process(pid)
        m = p.memory_info()
        return f"{m.rss / 2**30:.1f}", f"{m.private / 2**30:.1f}"
    except Exception:
        return "", ""


def main():
    g = json.load(open(sys.argv[1]))
    # servidor que morreu no meio da bateria (aimdo abortou, 26/09) deixava este laco esperando para sempre:
    # prazo de 10 min para o servidor subir, e `PULAR_GRAFOS` (um prefixo por linha) pula grafos de um servidor
    # que se sabe morto
    pular = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PULAR_GRAFOS")
    if os.path.exists(pular) and any(p and p in sys.argv[1] for p in open(pular).read().split()):
        print("PULADO (servidor morto):", sys.argv[1], flush=True)
        sys.exit(3)
    comfy = cc.Comfy(BASE)
    try:
        comfy.wait_up(600)
    except TimeoutError:
        print("ERRO: servidor nao respondeu em 10 min", flush=True)
        sys.exit(2)
    prazo = float(os.environ.get("PRAZO_GRAFO_S", "5400"))
    t0 = time.time()
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "g0_mib", "g0_util", "g1_mib", "g1_util", "ram_gib", "commit_gib", "comfy_rss_gib", "comfy_priv_gib"])
        ultima = [-1e9]

        def amostra(_decorrido):
            if time.time() - ultima[0] < 5:
                return
            ultima[0] = time.time()
            (a, b), (c, d) = gpus()[:2]
            r, cm = ram()
            w.writerow([round(time.time() - t0), a, b, c, d, f"{r:.1f}", f"{cm:.1f}", *comfy_mem()]); f.flush()

        reg = cc.roda_um(comfy, g, prazo, rotulo=sys.argv[1], tick=amostra,
                         extra={**cc.controles(comfy), "amostras_csv": sys.argv[2]})
    print(f"{reg['status']} em {time.time() - t0:.0f} s (server_side_s={reg.get('server_side_s')}, "
          f"cache_hit={reg.get('cache_hit')})", flush=True)
    if reg["status"] != "success":
        print(str(reg.get("erro"))[-3000:], flush=True)
    if len(sys.argv) > 3:
        cc.grava_jsonl(sys.argv[3], reg)
    print("FIM", flush=True)
    sys.exit(cc.codigo_de_saida([reg]))


if __name__ == "__main__":
    main()
