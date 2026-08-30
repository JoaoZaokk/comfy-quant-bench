@echo off
cd /d F:\COMFY_PORTABLE
set CUDA_VISIBLE_DEVICES=0,1
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --listen 0.0.0.0 --port 8190
pause
