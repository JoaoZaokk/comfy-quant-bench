@echo off
setlocal
title ComfyUI 8190 - Dual GPU Components
REM Qwen/Nunchaku dual-component mode:
REM   RTX 3090   = DiT principal (cuda:0)
REM   RTX 3080 Ti = text encoder e VAE (cuda:1)
REM
REM O multigpu-orchestrator cria um worker por GPU e envia cada prompt ao
REM worker menos ocupado. Isso serve para executar jobs independentes em
REM paralelo, mas nao serve para um unico workflow que precisa enxergar as
REM duas GPUs com papeis fixos. Desabilitamos apenas esse orquestrador; os
REM nodes do ComfyUI-MultiGPU continuam carregados e funcionais.

REM Executa a partir da pasta deste launcher, mesmo se o portable mudar de unidade.
cd /d "%~dp0"

REM Preserva acentos e prompts nao latinos no console e nos logs dos nodes.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "COMFYUI_MGPU_DISABLED=1"

REM SageAttention e o padrao validado nesta instalacao. Dynamic VRAM fica
REM desativado porque os loaders Nunchaku SVDQuant nao sao compativeis com
REM a inicializacao lazy de Linear usada pelo AIMDO.
set CUDA_VISIBLE_DEVICES=0,1
".\python_embeded\python.exe" -s ".\ComfyUI\main.py" --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190

set "COMFY_EXIT_CODE=%ERRORLEVEL%"
echo.
echo ComfyUI encerrado com codigo %COMFY_EXIT_CODE%.
pause
exit /b %COMFY_EXIT_CODE%
