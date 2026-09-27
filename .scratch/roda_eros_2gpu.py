"""Enfileira um grafo de API no ComfyUI 8190 e amostra as duas placas e a RAM a cada 5 s ate terminar.
Nao abre video nem audio. Uso: python -s .scratch/roda_eros_2gpu.py <grafo_api.json> <saida.csv>"""
import csv, ctypes, json, os, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:8190"


def req(path, data=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


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
                           capture_output=True, text=True).stdout.strip().splitlines()
        return [x.split(", ") for x in o]
    except OSError:
        return [["", ""], ["", ""]]


_proc = [None]


def comfy_mem():
    """RSS e memoria privada (commit) do processo que escuta a 8190, em GiB. Vazio se nao achar."""
    try:
        import psutil
        p = _proc[0]
        if p is None or not p.is_running():
            pid = next(c.pid for c in psutil.net_connections("tcp")
                       if c.laddr and c.laddr.port == 8190 and c.status == psutil.CONN_LISTEN)
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
    limite = time.time() + 600
    while True:
        try:
            req("/queue"); break
        except Exception:
            if time.time() > limite:
                print("ERRO: servidor nao respondeu em 10 min", flush=True)
                sys.exit(2)
            time.sleep(5)
    t0 = time.time()
    resp = req("/prompt", {"prompt": g})
    if resp.get("node_errors"):  # saida com erro de validacao e ignorada e o prompt "passa" sem rodar (26/09)
        print("ERRO de validacao:", json.dumps(resp["node_errors"])[:3000], flush=True)
        sys.exit(1)
    pid = resp["prompt_id"]
    print(f"enviado {pid}", flush=True)
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "g0_mib", "g0_util", "g1_mib", "g1_util", "ram_gib", "commit_gib", "comfy_rss_gib", "comfy_priv_gib"])
        while True:
            (a, b), (c, d) = gpus()[:2]
            r, cm = ram()
            w.writerow([round(time.time() - t0), a, b, c, d, f"{r:.1f}", f"{cm:.1f}", *comfy_mem()]); f.flush()
            h = req(f"/history/{pid}")
            if pid in h:
                st = h[pid]["status"]
                print(f"{st.get('status_str')} em {time.time() - t0:.0f} s", flush=True)
                if st.get("status_str") != "success":
                    print(json.dumps(st.get("messages", []))[-3000:], flush=True)
                break
            time.sleep(5)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
