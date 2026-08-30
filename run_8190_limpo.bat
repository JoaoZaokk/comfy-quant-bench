@echo off
REM Launcher do ComfyUI na 8190 — sem pause (nao trava a janela) e sem QuickEdit.
REM Log fica em F:\COMFY_PORTABLE\comfy_8190.log para poder ler o crash depois.

title ComfyUI-8190

REM desliga QuickEdit: e o que congela o console quando voce clica dentro dele
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

cd /d F:\COMFY_PORTABLE

echo ===== INICIO %date% %time% ===== > comfy_8190.log

set CUDA_VISIBLE_DEVICES=0,1
.\python_embeded\python.exe -s .\ComfyUI\main.py ^
  --windows-standalone-build ^
  --use-sage-attention ^
  --reserve-vram 1.5 ^
  --listen 0.0.0.0 ^
  --port 8190 2>&1 | .\python_embeded\python.exe -c "import sys;[ (sys.stdout.write(l), sys.stdout.flush(), open(r'F:\COMFY_PORTABLE\comfy_8190.log','a',encoding='utf-8',errors='replace').write(l)) for l in sys.stdin ]"

echo ===== FIM %date% %time% ===== >> comfy_8190.log
