$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$env:CUDA_VISIBLE_DEVICES = '0'

$bf16 = 'ComfyUI\models\diffusion_models\krea2_turbo_bf16.safetensors'
$calib = 'calib\krea2_turbo.calib.pt'

Assert-GpuLock -Owner 'w4a4:teto_krea2'
try {
    foreach ($cg in 64, 16) {
        $out = "ComfyUI\models\diffusion_models\krea2_turbo_w4a4_cg$cg.safetensors"
        $ana = "calib\krea2_turbo_cg$cg.analysis.json"
        Write-Host "=============== convrot_groupsize $cg ==============="
        if (Test-Path $out) { Write-Host "ja existe, pulando: $out"; continue }
        & .\python_embeded\python.exe -s .\tools\quant_mixed.py `
            --input $bf16 --profile krea2 --calibration $calib `
            --somente-w4a4 --uncalibrated fail --convrot-groupsize $cg `
            --save-analysis $ana --output $out
        Write-Host "EXIT_CG$cg=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw "conversao cg$cg falhou" }
    }
}
finally {
    Release-GpuLock
}
Write-Host "TETO_CONVERSOES_OK"
