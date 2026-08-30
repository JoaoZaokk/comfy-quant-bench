@echo off
setlocal
title ComfyUI 8190 - ULTRA Image / Nunchaku Dual GPU

cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Keep one ComfyUI process that can assign components to both GPUs.
REM The job-level orchestrator launches isolated workers and cannot split one graph.
set "COMFYUI_MGPU_DISABLED=1"

REM Recommended for Qwen/Z-Image/Nunchaku and repeated image edits:
REM - SageAttention on Ampere
REM - estimate-based loading required by the Nunchaku lazy Linear loaders
REM - RAM-pressure cache keeps unchanged loader/conditioning results while protecting RAM
REM - no sampler previews competing for GPU/PCIe/UI time
set CUDA_VISIBLE_DEVICES=0,1
".\python_embeded\python.exe" -s ".\ComfyUI\main.py" ^
  --windows-standalone-build ^
  --use-sage-attention ^
  --disable-dynamic-vram ^
  --cache-ram ^
  --preview-method none ^
  --listen 127.0.0.1 ^
  --port 8190

set "COMFY_EXIT_CODE=%ERRORLEVEL%"
echo.
echo ComfyUI encerrado com codigo %COMFY_EXIT_CODE%.
pause
exit /b %COMFY_EXIT_CODE%
