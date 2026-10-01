"""Sobe o main.py com dump de todas as threads a cada 120 s em pilha.txt (diagnóstico de travamento na importação)."""
import faulthandler, runpy, sys
f = open(r"F:\COMFY_PORTABLE\.scratch\pip_2026-09-29\pilha.txt", "w")
faulthandler.dump_traceback_later(15, repeat=True, file=f)
sys.argv = [r"F:\COMFY_PORTABLE\ComfyUI\main.py"] + sys.argv[1:]
sys.path.insert(0, r"F:\COMFY_PORTABLE\ComfyUI")
runpy.run_path(sys.argv[0], run_name="__main__")
