# FLUX.1 (Flux_Realistic_NSFW_02). A CALIBRACAO E A VERIFICACAO do perfil `flux_1`:
# `calibrate_activations` imprime "hooking N Linear layers matching profile" e grita se casar
# ZERO. O perfil preve 304 de 314; se sair outro numero, o perfil esta errado e para aqui.
#
# DualCLIP (t5xxl + clip_l) e o `ae.safetensors`, que e o VAE do Flux.
$ErrorActionPreference='Continue'; Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'; $env:COMFY_BYPASS_WMI_UNAME='1'

Write-Host "== calibracao =="
& .\python_embeded\python.exe -s .\tools\calibrate_activations.py `
    --model Flux_Realistic_NSFW_02.safetensors `
    --profile flux_1 `
    --clip t5xxl_fp8_e4m3fn_scaled.safetensors clip_l.safetensors --clip-type flux `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 --steps 20 --size 1024 `
    --out calib\flux_realistic.calib.pt
Write-Host "CALIB_EXIT=$LASTEXITCODE"

Write-Host "`n== W4A8 =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input ComfyUI\models\diffusion_models\Flux_Realistic_NSFW_02.safetensors `
    --calibration calib\flux_realistic.calib.pt `
    --promote-error 0.0 --budget 1.0 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\Flux_Realistic_NSFW_02_w4a8.safetensors
Write-Host "W4A8_EXIT=$LASTEXITCODE"

Write-Host "`n== W4A4 =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input ComfyUI\models\diffusion_models\Flux_Realistic_NSFW_02.safetensors `
    --calibration calib\flux_realistic.calib.pt `
    --somente-w4a4 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\Flux_Realistic_NSFW_02_w4a4.safetensors
Write-Host "W4A4_EXIT=$LASTEXITCODE"
Write-Host "FLUX_CONVERSAO_FIM"
