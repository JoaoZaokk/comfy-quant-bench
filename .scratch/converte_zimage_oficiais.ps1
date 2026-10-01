$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$env:CUDA_VISIBLE_DEVICES = '0'

# Os dois checkpoints OFICIAIS do Z-Image nunca foram convertidos nesta bancada -- so o
# `beyond-reality-zimage-v2` (de terceiros) foi. Ambos ja estao na nomenclatura do ComfyUI
# (453 tensores, `attention.qkv`), entao NAO passam pelo `to_native`, e ambos ja tem analise
# medida em cg 256 (`calib/xfer_*.analysis.json`), entao nao recalibram.
#
# As duas receitas sao as mesmas dos builds publicados, para a comparacao valer:
#   --promote-error 10.0  -> nada promove, 170/170 em W4A4
#   --promote-error 0.15  -> misto, o mesmo corte do `zimage-v2-mixed`
$fontes = @(
    @{ nome = 'zimage_turbo';    arq = 'z_image_turbo_bf16';       ana = 'xfer_z_image_turbo_bf16' },
    @{ nome = 'zimage_deturbo';  arq = 'z_image_de_turbo_v1_bf16'; ana = 'xfer_z_image_de_turbo_v1_bf16' }
)
$receitas = @(
    @{ sufixo = 'w4a4';  promote = '10.0' },
    @{ sufixo = 'mixed'; promote = '0.15' }
)

Assert-GpuLock -Owner 'w4a4:zimage_oficiais'
try {
    foreach ($f in $fontes) {
        foreach ($r in $receitas) {
            $out = "ComfyUI\models\diffusion_models\$($f.nome)_$($r.sufixo).safetensors"
            Write-Host "=============== $($f.nome) $($r.sufixo) ==============="
            if (Test-Path $out) { Write-Host "ja existe, pulando"; continue }
            & .\python_embeded\python.exe -s .\tools\quant_mixed.py `
                --input "ComfyUI\models\diffusion_models\$($f.arq).safetensors" `
                --profile zimage `
                --analysis "calib\$($f.ana).analysis.json" `
                --promote-error $r.promote `
                --output $out
            Write-Host "EXIT=$LASTEXITCODE"
            if ($LASTEXITCODE -ne 0) { throw "conversao $($f.nome) $($r.sufixo) falhou" }
        }
    }
}
finally { Release-GpuLock }
Write-Host "ZIMAGE_OFICIAIS_OK"
