"""Fase 7: roda ubench_imma (IMMA m16n8k64 s4 + F FFMA por IMMA) e converte em IMMA por ciclo por SMSP.

Taxa de IMMA = blocos * warps * L * 8 / tempo; FFMA por IMMA = F. Clock do SM lido do nvidia-smi durante a rodada nao e
confiavel a 280 W; o resultado relevante e a RAZAO entre F = 0 e F > 0 (quantas FFMA cabem sem perder IMMA).
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402


def main():
    torch.cuda.set_device(0)
    pyd = open(os.path.join(HERE, "build_g64", "ubench_imma", "OK"), encoding="utf-8").read().strip()
    spec = importlib.util.spec_from_file_location("ubench_imma", pyd)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    Fs = [0, 2, 4, 6, 8, 10, 12, 16, 24, 32]
    L = 2000
    for warps in (4, 8, 16):
        for mode in (0, 1):
            ms = mod.bench(sms, warps * 32, L, mode)
            n_imma = sms * warps * L * 8
            base = ms[0]
            txt = " ".join(f"F{f}={m:.2f}ms({(m / base - 1) * 100:+.0f}%)" for f, m in zip(Fs, ms))
            print(f"warps/SM={warps} modo={mode}: IMMA/s(F0)={n_imma / base / 1e6:.0f}/ms  {txt}", flush=True)


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:fase7_ubench_imma") as gd:
        if gd.refused:
            print(gd.refused); raise SystemExit(1)
        main()
