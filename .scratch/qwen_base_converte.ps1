# Qwen-Image 2512 BASE (nao-edit). Fonte em P: (NAS), saida em P: -- F: tem 15 GB.
#
# Calibra o BASE, nao reusa a analise do EDIT: os dois tem tamanho quase igual (72 bytes de
# diferenca) e a mesma arquitetura MMDiT, mas `quant_mixed` recusa analise de outro checkpoint
# por padrao e esta certo em recusar -- pesos diferentes dao erros por camada diferentes.
#
# Dois bracos, o mesmo par que decidiu no Qwen Edit e no Wan:
#   W4A8  -- o formato que sobreviveu nas duas familias
#   W4A4  -- o que virou estatica no Edit e borrao no Wan. Medir, nao prever.
$ErrorActionPreference='Continue'; Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'; $env:COMFY_BYPASS_WMI_UNAME='1'

Write-Host "== calibracao do BASE =="
& .\python_embeded\python.exe -s .\tools\calibrate_activations.py `
    --model P:\ComfyBench\diffusion_models\qwen_image_2512_bf16.safetensors `
    --profile qwen_image `
    --clip qwen_2.5_vl_7b.safetensors --clip-type qwen_image `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 --steps 20 --size 1024 `
    --out calib\qwen_image_2512.calib.pt
Write-Host "CALIB_EXIT=$LASTEXITCODE"

Write-Host "`n== W4A8 =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input P:\ComfyBench\diffusion_models\qwen_image_2512_bf16.safetensors `
    --calibration calib\qwen_image_2512.calib.pt `
    --promote-error 0.0 --budget 1.0 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\qwen_image_2512_w4a8.safetensors
Write-Host "W4A8_EXIT=$LASTEXITCODE"

Write-Host "`n== W4A4 =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input P:\ComfyBench\diffusion_models\qwen_image_2512_bf16.safetensors `
    --calibration calib\qwen_image_2512.calib.pt `
    --somente-w4a4 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\qwen_image_2512_w4a4.safetensors
Write-Host "W4A4_EXIT=$LASTEXITCODE"
Write-Host "QWEN_BASE_CONVERSAO_FIM"
