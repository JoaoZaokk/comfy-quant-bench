$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:diag_travamento_ui'
try {
    $env:CUDA_VISIBLE_DEVICES = '0,1'; $env:COMFYUI_MGPU_DISABLED = '1'; $env:PYTHONUTF8 = '1'; $env:PYTHONNOUSERSITE = '1'
    & .\python_embeded\python.exe -s .scratch\pip_2026-09-29\pilha.py --windows-standalone-build --enable-triton-backend --use-sage-attention --enable-manager --disable-dynamic-vram --disable-auto-launch --listen 127.0.0.1 --port 8198 *> .scratch\pip_2026-09-29\pilha_ui.log
    "rc=$LASTEXITCODE"
} finally { Release-GpuLock }
