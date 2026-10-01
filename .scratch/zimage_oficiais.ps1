$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$env:CUDA_VISIBLE_DEVICES = '0'

# As analises `xfer_*` que existiam no disco foram RECUSADAS pelo conversor: nao carregam
# `source_identity_sha256`, que so passou a ser escrito em 2026-08-22. O guard esta certo --
# "nada a conferir" nao e "conferido" -- entao recalibra-se do zero. Custa ~83 s por checkpoint.
#
# Parametros copiados da calibragem VALIDA do Z-Image (`zimage_v2_sigma`), nao inventados:
# 8 passos, 1024, cfg 1.0, euler/simple, duas sementes, 128 linhas por camada.
$fontes = @(
    @{ nome = 'zimage_turbo';   arq = 'z_image_turbo_bf16' },
    @{ nome = 'zimage_deturbo'; arq = 'z_image_de_turbo_v1_bf16' }
)
$receitas = @(
    @{ sufixo = 'w4a4';  promote = '10.0' },
    @{ sufixo = 'mixed'; promote = '0.15' }
)

Assert-GpuLock -Owner 'w4a4:zimage_oficiais'
try {
    foreach ($f in $fontes) {
        $calib = "calib\$($f.nome).calib.pt"
        if (-not (Test-Path $calib)) {
            Write-Host "=============== calibrando $($f.nome) ==============="
            & .\python_embeded\python.exe -s .\tools\calibrate_activations.py `
                --model "ComfyUI\models\diffusion_models\$($f.arq).safetensors" `
                --profile zimage --out $calib `
                --clip qwen_3_4b.safetensors --clip-type lumina2 `
                --prompt-file bench\prompts_zimage_runtime.txt `
                --seeds 1234 5678 --steps 8 --size 1024 --cfg 1.0 `
                --sampler euler --scheduler simple --rows 128
            if ($LASTEXITCODE -ne 0) { throw "calibragem $($f.nome) falhou" }
        }
        foreach ($r in $receitas) {
            $out = "ComfyUI\models\diffusion_models\$($f.nome)_$($r.sufixo).safetensors"
            $ana = "calib\$($f.nome).analysis.json"
            Write-Host "=============== $($f.nome) $($r.sufixo) ==============="
            if (Test-Path $out) { Write-Host "ja existe, pulando"; continue }
            $extra = @()
            if (Test-Path $ana) { $extra = @('--analysis', $ana) } else { $extra = @('--save-analysis', $ana) }
            & .\python_embeded\python.exe -s .\tools\quant_mixed.py `
                --input "ComfyUI\models\diffusion_models\$($f.arq).safetensors" `
                --profile zimage --calibration $calib `
                --promote-error $r.promote --output $out @extra
            if ($LASTEXITCODE -ne 0) { throw "conversao $($f.nome) $($r.sufixo) falhou" }
        }
    }
}
finally { Release-GpuLock }
Write-Host "ZIMAGE_OFICIAIS_OK"
