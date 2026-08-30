@echo off
setlocal
title ComfyUI 8190 - ULTRA Video / Low Cache Dual GPU

cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "COMFYUI_MGPU_DISABLED=1"

REM Recommended for LTX/VOID/large video graphs:
REM cache-none prevents old intermediate tensors from exhausting 64 GB system RAM.
REM CUDA graphs remain enabled globally; compatible workflow nodes decide when to capture.
set CUDA_VISIBLE_DEVICES=0,1
".\python_embeded\python.exe" -s ".\ComfyUI\main.py" ^
  --windows-standalone-build ^
  --use-sage-attention ^
  --disable-dynamic-vram ^
  --cache-none ^
  --preview-method none ^
  --listen 127.0.0.1 ^
  --port 8190

set "COMFY_EXIT_CODE=%ERRORLEVEL%"
echo.
echo ComfyUI encerrado com codigo %COMFY_EXIT_CODE%.
pause
exit /b %COMFY_EXIT_CODE%
