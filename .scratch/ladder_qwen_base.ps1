# Ladder do Qwen-Image 2512 BASE. Mesmo desenho do Edit: 6 prompts x 2 sementes, 20 passos,
# 1024px, cfg 2.5. Um eixo -- o transformer. Encoder, VAE, prompts e sementes fixos.
$ErrorActionPreference='Continue'; Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'; $env:COMFY_BYPASS_WMI_UNAME='1'
$P='P:\ComfyBench\diffusion_models\'
& .\python_embeded\python.exe -s .\tools\quality_ladder.py `
    --reference "$($P)qwen_image_2512_bf16.safetensors" `
    --models "$($P)qwen_image_2512_w4a8.safetensors" `
             "$($P)qwen_image_2512_w4a4.safetensors" `
    --clip qwen_2.5_vl_7b.safetensors --clip-type qwen_image `
    --vae qwen_image_vae.safetensors `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 2 --steps 20 --size 1024 --cfg 2.5 `
    --sampler euler --scheduler simple `
    --out bench\quality_ladder_qwen_base
Write-Host "LADDER_BASE_EXIT=$LASTEXITCODE"
Write-Host "LADDER_BASE_FIM"
