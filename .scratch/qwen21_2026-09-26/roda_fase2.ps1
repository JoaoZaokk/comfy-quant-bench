# Fase 2 do Qwen-Image-2.1: nossas quantizacoes, calibracao real, mixed, erro por camada. Um lock do inicio ao fim.
# Cada passo grava log proprio em fase2\; um passo que falha nao impede os independentes.
param([string[]]$Passos = @('int8', 'w4a8', 'w4a4', 'calib', 'mixed', 'camadas'))
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
New-Item -ItemType Directory -Force "$D\fase2" | Out-Null
$PY = '.\python_embeded\python.exe'
$DM = 'P:\ComfyBench\diffusion_models'
$SRC = "$DM\qwen_image_2.1_bf16.safetensors"
$CAL = 'calib\qwen21_bf16_2026-09-26.calib.pt'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qwen21_fase2_quant_calib'
function Passo($nome, [string[]]$argv) {
    Write-Host "=== $nome inicio $(Get-Date -Format T)"
    & $PY -s @argv *> "$D\fase2\$nome.log"
    Write-Host "=== $nome fim $(Get-Date -Format T) rc=$LASTEXITCODE"
}
try {
    $env:CUDA_VISIBLE_DEVICES = '0'   # so a 3090; a 3080 Ti fica livre
    if ($Passos -contains 'int8') { Passo 'int8' @('tools\quant_int8.py', '--input', $SRC, '--convrot', '--device', 'cuda') }
    if ($Passos -contains 'w4a8') { Passo 'w4a8' @('tools\quant_w4a8.py', '--input', $SRC) }
    if ($Passos -contains 'w4a4') { Passo 'w4a4' @('tools\quant_w4a4.py', '--input', $SRC) }
    if ($Passos -contains 'calib') {
        Passo 'calib' @('tools\calibrate_activations.py', '--model', $SRC, '--profile', 'qwen_image21',
                        '--clip', 'qwen3vl_8b_w4a8.safetensors', '--clip-type', 'qwen_image',
                        '--prompt-file', "$D\prompts_calib.txt", '--steps', '25', '--seeds', '42',
                        '--size', '1024', '--rows', '256', '--attention', 'sage', '--out', $CAL)
    }
    if ($Passos -contains 'mixed') {
        Passo 'mixed' @('tools\quant_mixed.py', '--input', $SRC, '--calibration', $CAL,
                        '--save-analysis', 'calib\qwen21_bf16_2026-09-26.analysis.json')
    }
    if ($Passos -contains 'camadas') {
        Passo 'camadas' @('tools\erro_por_camada.py', '--source', $SRC, '--calib', $CAL,
                          '--build', "comfyorg_int8=$DM\qwen_image_2.1_int8_convrot.safetensors",
                          '--build', 'nidall_mixed=D:\ComfyUI-Models\diffusion_models\qwen_image_2.1_mixed_balanced.safetensors',
                          '--build', 'mesmer_int4=D:\ComfyUI-Models\diffusion_models\qwen-image-2.1-int4-r128.safetensors',
                          '--build', "nosso_int8=$DM\qwen_image_2.1_bf16_int8_convrot.safetensors",
                          '--build', "nosso_w4a8=$DM\qwen_image_2.1_bf16_w4a8.safetensors",
                          '--build', "nosso_w4a4=$DM\qwen_image_2.1_bf16_w4a4_convrot.safetensors",
                          '--build', "nosso_mixed=$DM\qwen_image_2.1_bf16_mixed.safetensors",
                          '--out', "$D\fase2\erro_por_camada.json")
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM fase2 $(Get-Date -Format T)"
