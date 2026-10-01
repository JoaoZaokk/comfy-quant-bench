$ErrorActionPreference='Continue'
Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'
$d='ComfyUI\models\diffusion_models\'
& .\python_embeded\python.exe -s .\tools\quality_ladder.py `
  --reference "$($d)qwen_image_edit_2511_w4a8.safetensors" `
  --models "$($d)qwen_image_edit_2511_w4a4.safetensors" `
  --clip qwen_2.5_vl_7b.safetensors --clip-type qwen_image `
  --vae qwen_image_vae.safetensors `
  --prompt-file bench\prompts_zimage_runtime.txt `
  --seeds 1 2 --steps 20 --size 1024 --cfg 2.5 `
  --sampler euler --scheduler simple `
  --out bench\quality_ladder_qwen_edit
Write-Host "W4A8_EXIT=$LASTEXITCODE"
