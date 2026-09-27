"""Sobe o ComfyUI com o `HostBuffer.read_file_slice` do comfy-aimdo instrumentado, para achar por que o DiT grande
lido do NAS (fast_disk=False) falha com `device copy failed result=2` e derruba o processo.

Registra em JSONL, a cada ~1 GiB lido e em toda falha: RAM disponivel, pinned do ComfyUI (TOTAL/MAX), VRAM livre,
tamanho do host buffer, se o endereco de origem esta registrado (cudaPointerGetAttributes) e o erro CUDA pendente
(cudaPeekAtLastError). Com DIAG_FIX=1 aplica o conserto candidato: na falha da copia, limpa o erro pendente, relê
so para o host e copia para a placa com cudaMemcpy sincrono.

    python_embeded\\python.exe -s .scratch/diag_aimdo_2026-09-27/lanca_diag_aimdo.py <saida.jsonl> -- <args do main.py>
"""
import ctypes
import json
import os
import runpy
import sys
import threading
import time

RAIZ = r"F:\COMFY_PORTABLE\ComfyUI"
SAIDA = os.path.abspath(sys.argv[1])
ARGS = sys.argv[sys.argv.index("--") + 1:]
LANCADOR = os.path.abspath(__file__)
CONSERTA = os.environ.get("DIAG_FIX") == "1"
GIB = 2 ** 30


def grava(reg):
    reg["t"] = round(time.time(), 2)
    with open(SAIDA, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")


class PtrAttr(ctypes.Structure):  # cudaPointerAttributes (CUDA 11+)
    _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int),
                ("devicePointer", ctypes.c_void_p), ("hostPointer", ctypes.c_void_p)]


def instala():
    # espera o ComfyUI terminar os imports (importar torch antes do main.py mudaria o alocador que ele configura)
    while "server" not in sys.modules or "comfy.model_management" not in sys.modules:
        time.sleep(0.2)
    time.sleep(2)
    import psutil
    import torch
    import comfy_aimdo.host_buffer as hb

    rt = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_13.dll"))
    rt.cudaPeekAtLastError.restype = ctypes.c_int
    rt.cudaGetLastError.restype = ctypes.c_int
    rt.cudaPointerGetAttributes.restype = ctypes.c_int
    rt.cudaMemcpy.restype = ctypes.c_int
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def ptr_tipo(ptr):
        return "nao_medido"

    def estado():
        try:
            return _estado()
        except Exception as e:  # noqa: BLE001
            return {"estado_erro": repr(e)[:200]}

    def _estado():
        mm = sys.modules.get("comfy.model_management")
        vm = psutil.virtual_memory()
        livre, total = torch.cuda.mem_get_info(0)
        return {"ram_disp_gib": round(vm.available / GIB, 2), "ram_usada_pct": vm.percent,
                "pinned_gib": round(getattr(mm, "TOTAL_PINNED_MEMORY", 0) / GIB, 2) if mm else None,
                "pinned_max_gib": round(getattr(mm, "MAX_PINNED_MEMORY", 0) / GIB, 2) if mm else None,
                "vram_livre_gib": round(livre / GIB, 2), "cuda_pendente": rt.cudaPeekAtLastError()}

    lido, lido_erro = [0, 0], []
    orig = hb.HostBuffer.read_file_slice

    injeta = [int(os.environ.get("DIAG_INJETA", "0") or 0), 0]

    def read_file_slice(self, file_obj, file_offset, size, offset=0, stream=0, device_ptr=0, device=-1):
        # DIAG_INJETA=N: a N-esima leitura com destino na placa falha como o aimdo falha (RuntimeError), para exercitar
        # o fallback do ComfyUI sem depender de reproduzir o OOM da copia
        if injeta[0] and device_ptr:
            injeta[1] += 1
            if injeta[1] == injeta[0]:
                grava({"ev": "INJETADO", "n": injeta[1], "size": int(size), **estado()})
                raise RuntimeError("HostBuffer.read_file_slice failed (injetado pelo diagnostico)")
        try:
            r = orig(self, file_obj, file_offset, size, offset=offset, stream=stream, device_ptr=device_ptr, device=device)
        except RuntimeError as e:
            host = self.get_raw_address() + int(offset)
            reg = {"ev": "FALHA", "erro": str(e), "size": int(size), "offset": int(offset), "hostbuf_gib": round(self.size / GIB, 2),
                   "device": device, "device_ptr": hex(int(device_ptr)), "stream": hex(int(stream or 0)),
                   "origem": ptr_tipo(host), "destino": ptr_tipo(int(device_ptr)) if device_ptr else None,
                   "lido_gib": round(lido[0] / GIB, 2), **estado()}
            grava(reg)
            if not (CONSERTA and device_ptr):
                raise
            # conserto candidato: descarta o erro pendente, le so para o host e copia sincrono
            rt.cudaGetLastError()
            torch.cuda.synchronize()
            orig(self, file_obj, file_offset, size, offset=offset, stream=0, device_ptr=0, device=None)
            rc = rt.cudaMemcpy(ctypes.c_void_p(int(device_ptr)), ctypes.c_void_p(host), int(size), 1)
            grava({"ev": "FALLBACK", "cudaMemcpy": rc, **estado()})
            if rc:
                rt.cudaGetLastError()
                raise RuntimeError(f"fallback cudaMemcpy falhou ({rc})") from e
            r = None
        pend = rt.cudaPeekAtLastError() if device_ptr else 0
        if pend and not lido_erro:
            lido_erro.append(1)
            host = self.get_raw_address() + int(offset)
            grava({"ev": "ERRO_PENDENTE_APOS_LEITURA", "cuda": pend, "size": int(size), "offset": int(offset),
                   "hostbuf_gib": round(self.size / GIB, 2), "device_ptr": hex(int(device_ptr)),
                   "origem": ptr_tipo(host), "destino": ptr_tipo(int(device_ptr)), "lido_gib": round(lido[0] / GIB, 2),
                   **estado()})
        lido[0] += int(size)
        if lido[0] - lido[1] >= GIB:
            lido[1] = lido[0]
            grava({"ev": "leitura", "lido_gib": round(lido[0] / GIB, 1), "hostbuf_gib": round(self.size / GIB, 2),
                   "com_device": bool(device_ptr), **estado()})
        return r

    hb.HostBuffer.read_file_slice = read_file_slice

    # a excecao que interrompe a amostragem (o cleanup do finally e onde o processo aborta)
    import traceback
    import comfy.samplers as smp
    orig_inner = smp.CFGGuider.inner_sample

    def inner_sample(self, *a, **k):
        try:
            return orig_inner(self, *a, **k)
        except BaseException as e:
            grava({"ev": "EXCECAO_NA_AMOSTRAGEM", "tipo": type(e).__name__, "msg": str(e)[:600],
                   "tb": traceback.format_exc()[-3000:], **estado()})
            raise

    smp.CFGGuider.inner_sample = inner_sample
    grava({"ev": "instalado", "conserta": CONSERTA, "args": ARGS})


# o ComfyUI-Manager relanca o main.py com os.execv; volta pelo lancador para manter a instrumentacao
_execv = os.execv


def execv(path, argv):
    q = (lambda s: f'"{s}"' if " " in s else s)
    _execv(sys.executable, [q(sys.executable), "-s", q(LANCADOR), q(SAIDA), "--", *ARGS])


os.execv = execv
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import faulthandler
_fh = open(SAIDA + ".faulthandler.txt", "a")
faulthandler.enable(file=_fh, all_threads=True)
threading.Thread(target=instala, daemon=True).start()
sys.argv = [os.path.join(RAIZ, "main.py"), *ARGS]
runpy.run_path(sys.argv[0], run_name="__main__")
