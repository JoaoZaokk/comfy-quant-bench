$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES = '0'

# Quatro bracos: a referencia BF16 (38,1 GiB, descarrega inteira e por isso o s/passo dela nao
# se compara com nada), os nossos dois, e o int8 do proprio Comfy-Org -- que em quatro familias
# seguidas nesta bancada saiu mais fiel que o nosso W4A4. Baixar o concorrente e mais barato do
# que descobrir depois que o nosso perde e nao ter com o que comparar.
#
# 20 passos e cfg 2.5 porque e o regime real deste modelo, nao o cfg 1.0 dos destilados. Isso
# dobra o trabalho por passo (cond + uncond) e e o preco de medir o que as pessoas usam.
$d = 'ComfyUI\models\diffusion_models\'
& .\python_embeded\python.exe -s .\tools\quality_ladder.py `
    --reference "$($d)qwen_image_edit_2511_bf16.safetensors" `
    --models "$($d)qwen_image_edit_2511_w4a4.safetensors" `
             "$($d)qwen_image_edit_2511_mixed.safetensors" `
             "$($d)qwen_image_edit_2511_int8_convrot.safetensors" `
    --clip qwen_2.5_vl_7b.safetensors --clip-type qwen_image `
    --vae qwen_image_vae.safetensors `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 2 --steps 20 --size 1024 --cfg 2.5 `
    --sampler euler --scheduler simple `
    --out bench\quality_ladder_qwen_edit
Write-Host "LADDER_QWEN_EXIT=$LASTEXITCODE"
