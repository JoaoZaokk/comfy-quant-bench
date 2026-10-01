$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES = '0'

# Um ladder por FONTE. Misturar os builds do `beyond-reality` com os do `z_image_turbo` oficial
# num ladder so compararia contra a referencia errada: a divergencia de latente e medida contra
# o braco `--reference`, entao cada fonte precisa da sua.
$d = 'ComfyUI\models\diffusion_models\'
$grupos = @(
    @{ ref = 'z_image_turbo_bf16';        bracos = @('zimage_turbo_w4a4', 'zimage_turbo_mixed');       saida = 'quality_ladder_zimage_turbo' },
    @{ ref = 'z_image_de_turbo_v1_bf16';  bracos = @('zimage_deturbo_w4a4', 'zimage_deturbo_mixed');   saida = 'quality_ladder_zimage_deturbo' }
)

foreach ($g in $grupos) {
    Write-Host "=============== $($g.ref) ==============="
    $modelos = $g.bracos | ForEach-Object { "$($d)$_.safetensors" }
    $faltando = $modelos | Where-Object { -not (Test-Path $_) }
    if ($faltando) { Write-Host "PULANDO, faltam: $faltando"; continue }
    & .\python_embeded\python.exe -s .\tools\quality_ladder.py `
        --reference "$($d)$($g.ref).safetensors" `
        --models @modelos `
        --clip qwen_3_4b.safetensors --clip-type lumina2 `
        --vae ae.safetensors `
        --prompt-file bench\prompts_zimage_runtime.txt `
        --seeds 1 2 --steps 8 --size 1024 --cfg 1.0 `
        --sampler euler --scheduler simple `
        --out "bench\$($g.saida)"
    Write-Host "EXIT_$($g.ref)=$LASTEXITCODE"
}
Write-Host "LADDER_ZIMAGE_FIM"
