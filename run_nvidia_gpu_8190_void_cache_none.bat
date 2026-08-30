@echo off
setlocal
title ComfyUI 8190 - VOID cache-none

cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Disable the job-level MGPU orchestrator. Native Select * Device nodes remain active.
set "COMFYUI_MGPU_DISABLED=1"

REM VOID/Nunchaku-safe baseline: SageAttention, no DynamicVRAM, no execution-output cache.
set CUDA_VISIBLE_DEVICES=0,1
".\python_embeded\python.exe" -s ".\ComfyUI\main.py" ^
  --windows-standalone-build ^
  --use-sage-attention ^
  --disable-dynamic-vram ^
  --cache-none ^
  --listen 127.0.0.1 ^
  --port 8190

set "COMFY_EXIT_CODE=%ERRORLEVEL%"
echo.
echo ComfyUI encerrado com codigo %COMFY_EXIT_CODE%.
pause
exit /b %COMFY_EXIT_CODE%
