# Carrega todos os custom nodes e sai (--quick-test-for-ci). Só a 3090, orquestrador desligado, lock próprio.
# Uso: importa.ps1 <rotulo>   -> importa_<rotulo>.log
param([Parameter(Mandatory)][string]$Rotulo)
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:pip_upgrade_importa'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:COMFYUI_MGPU_DISABLED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
$D = '.scratch\pip_2026-09-29'
try {
    & .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-triton-backend --use-sage-attention --disable-dynamic-vram --enable-manager --disable-auto-launch --quick-test-for-ci --port 8199 *> "$D\importa_$Rotulo.log"
    "rc=$LASTEXITCODE"
} finally {
    Release-GpuLock
}
