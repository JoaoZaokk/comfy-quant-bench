$ErrorActionPreference='Continue'; Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'; $env:COMFY_BYPASS_WMI_UNAME='1'
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input ComfyUI\models\diffusion_models\wan2.2_ti2v_5B_fp16.safetensors `
    --calibration calib\wan22_ti2v_5b.calib.pt `
    --somente-w4a4 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\wan2.2_ti2v_5B_w4a4.safetensors
Write-Host "W4A4_EXIT=$LASTEXITCODE"
Write-Host "WAN_W4A4_FIM"
