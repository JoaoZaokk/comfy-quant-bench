"""Sobe o ComfyUI com transformers 5.17 de uma pasta isolada na frente do site-packages (o python_embeded usa ._pth,
que ignora PYTHONPATH). O site-packages principal continua com o 4.57.6."""
import runpy
import sys

sys.path.insert(0, "C:/ComfyBench/zen_test/pydeps")
sys.argv = ["main.py", *sys.argv[1:]]
runpy.run_path("F:/COMFY_PORTABLE/ComfyUI/main.py", run_name="__main__")
