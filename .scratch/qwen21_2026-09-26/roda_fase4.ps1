# Fase 4: refino de escalas por linha (minimos quadrados, forma fechada) e erro por camada dos refinados. Um lock.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
$PY = '.\python_embeded\python.exe'
$DM = 'P:\ComfyBench\diffusion_models'
$SRC = "$DM\qwen_image_2.1_bf16.safetensors"
$CAL = 'calib\qwen21_bf16_2026-09-26.calib.pt'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qwen21_fase4_escalas'
function Passo($nome, [string[]]$argv) {
    Write-Host "=== $nome inicio $(Get-Date -Format T)"
    & $PY -s @argv *> "$D\fase2\$nome.log"
    Write-Host "=== $nome fim $(Get-Date -Format T) rc=$LASTEXITCODE"
}
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    foreach ($b in @('qwen_image_2.1_bf16_mixed', 'qwen_image_2.1_bf16_w4a4_convrot_f32', 'qwen_image_2.1_bf16_w4a8_f32')) {
        Passo "escalas_$b" @('tools\refina_escalas.py', '--model', "$DM\$b.safetensors", '--source', $SRC, '--calib', $CAL)
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM fase4 $(Get-Date -Format T)"
