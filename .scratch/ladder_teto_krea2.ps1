$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
# CUDA_VISIBLE_DEVICES=0: a 3080 Ti tem o embedding do cortex residente (~2,6 GiB) e o BenchGuard
# recusa acima de 2,00 GiB. Escondendo a placa, a recusa passa a olhar so a que esta em uso -- e
# mantem as condicoes identicas as da corrida de qualidade anterior, que e o que torna os s/passo
# comparaveis.
$env:CUDA_VISIBLE_DEVICES = '0'

$d = 'ComfyUI\models\diffusion_models\'
& .\python_embeded\python.exe -s .\tools\quality_ladder.py `
    --reference "$($d)krea2_turbo_bf16.safetensors" `
    --models "$($d)krea2_turbo_w4a4.safetensors" `
             "$($d)krea2_turbo_w4a4_cg64.safetensors" `
             "$($d)krea2_turbo_w4a4_cg16.safetensors" `
    --clip qwen3vl_4b_bf16.safetensors `
    --clip-type krea2 `
    --vae qwen_image_vae.safetensors `
    --prompt-file bench\prompts_krea2.txt `
    --seeds 1 2 --steps 10 --size 1024 --cfg 1.0 `
    --sampler euler --scheduler simple `
    --out bench\quality_ladder_krea2_teto
Write-Host "LADDER_TETO_EXIT=$LASTEXITCODE"
