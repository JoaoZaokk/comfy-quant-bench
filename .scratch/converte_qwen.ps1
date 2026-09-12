$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$env:CUDA_VISIBLE_DEVICES = '0'

# As duas receitas sao as mesmas usadas no Krea2 e no Z-Image, para a comparacao entre familias
# valer: `--promote-error 10.0` nao promove nada (840/840 em W4A4) e `0.15` e o corte misto.
$src = 'ComfyUI\models\diffusion_models\qwen_image_edit_2511_bf16.safetensors'
$calib = 'calib\qwen_image_edit_2511.calib.pt'
$ana = 'calib\qwen_image_edit_2511.analysis.json'
$receitas = @(
    @{ sufixo = 'w4a4';  promote = '10.0' },
    @{ sufixo = 'mixed'; promote = '0.15' }
)

Assert-GpuLock -Owner 'w4a4:converte_qwen'
try {
    foreach ($r in $receitas) {
        $out = "ComfyUI\models\diffusion_models\qwen_image_edit_2511_$($r.sufixo).safetensors"
        Write-Host "=============== $($r.sufixo) ==============="
        if (Test-Path $out) { Write-Host "ja existe, pulando"; continue }
        $extra = if (Test-Path $ana) { @('--analysis', $ana) } else { @('--save-analysis', $ana) }
        & .\python_embeded\python.exe -s .\tools\quant_mixed.py `
            --input $src --profile qwen_image --calibration $calib `
            --promote-error $r.promote --output $out @extra
        if ($LASTEXITCODE -ne 0) { throw "conversao $($r.sufixo) falhou" }
    }
}
finally { Release-GpuLock }
Write-Host "CONVERTE_QWEN_OK"
