# Fase 5: mixed conservador (promote 0,10: promove tambem o gate_up) e as escalas dele. Um lock.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
$PY = '.\python_embeded\python.exe'
$DM = 'P:\ComfyBench\diffusion_models'
$SRC = "$DM\qwen_image_2.1_bf16.safetensors"
$CAL = 'calib\qwen21_bf16_2026-09-26.calib.pt'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qwen21_fase5_mixed010'
function Passo($nome, [string[]]$argv) {
    Write-Host "=== $nome inicio $(Get-Date -Format T)"
    & $PY -s @argv *> "$D\fase2\$nome.log"
    Write-Host "=== $nome fim $(Get-Date -Format T) rc=$LASTEXITCODE"
}
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    Passo 'mixed_p010' @('tools\quant_mixed.py', '--input', $SRC, '--calibration', $CAL, '--promote-error', '0.10',
                         '--output', "$DM\qwen_image_2.1_bf16_mixed_p010.safetensors")
} finally {
    Release-GpuLock
}
Write-Host "=== FIM fase5 $(Get-Date -Format T)"
