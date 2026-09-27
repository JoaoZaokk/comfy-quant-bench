# Fase 3: nossas quants refeitas do FP32 (correcao de 26/09) e erro por camada delas. Um lock.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
$PY = '.\python_embeded\python.exe'
$DM = 'P:\ComfyBench\diffusion_models'
$SRC = "$DM\qwen_image_2.1_bf16.safetensors"
$CAL = 'calib\qwen21_bf16_2026-09-26.calib.pt'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qwen21_fase3_quant_f32'
function Passo($nome, [string[]]$argv) {
    Write-Host "=== $nome inicio $(Get-Date -Format T)"
    & $PY -s @argv *> "$D\fase2\$nome.log"
    Write-Host "=== $nome fim $(Get-Date -Format T) rc=$LASTEXITCODE"
}
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    Passo 'int8_f32' @('tools\quant_int8.py', '--input', $SRC, '--convrot', '--device', 'cuda', '--output', "$DM\qwen_image_2.1_bf16_int8_convrot_f32.safetensors")
    Passo 'w4a8_f32' @('tools\quant_w4a8.py', '--input', $SRC, '--output', "$DM\qwen_image_2.1_bf16_w4a8_f32.safetensors")
    Passo 'w4a4_f32' @('tools\quant_w4a4.py', '--input', $SRC, '--output', "$DM\qwen_image_2.1_bf16_w4a4_convrot_f32.safetensors")
    Passo 'camadas_f32' @('tools\erro_por_camada.py', '--source', $SRC, '--calib', $CAL,
                          '--build', "comfyorg_int8=$DM\qwen_image_2.1_int8_convrot.safetensors",
                          '--build', 'nidall_mixed=D:\ComfyUI-Models\diffusion_models\qwen_image_2.1_mixed_balanced.safetensors',
                          '--build', "nosso_int8_f32=$DM\qwen_image_2.1_bf16_int8_convrot_f32.safetensors",
                          '--build', "nosso_w4a8_f32=$DM\qwen_image_2.1_bf16_w4a8_f32.safetensors",
                          '--build', "nosso_w4a4_f32=$DM\qwen_image_2.1_bf16_w4a4_convrot_f32.safetensors",
                          '--out', "$D\fase2\erro_por_camada_f32.json")
} finally {
    Release-GpuLock
}
Write-Host "=== FIM fase3 $(Get-Date -Format T)"
