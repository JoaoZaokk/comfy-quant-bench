$ErrorActionPreference='Continue'; Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES='0'; $env:COMFY_BYPASS_WMI_UNAME='1'

Write-Host "== baixando o VAE 2.2 (48 canais; o 2.1 tem 16 e daria imagem plausivel e errada) =="
& .\python_embeded\python.exe -s -c @"
import os
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','0')
from huggingface_hub import hf_hub_download
import shutil
d = r'F:\COMFY_PORTABLE\ComfyUI\models\vae'
alvo = os.path.join(d,'wan2.2_vae.safetensors')
if os.path.exists(alvo) and os.path.getsize(alvo)==1409400960:
    print('ja existe'); raise SystemExit(0)
p = hf_hub_download('Comfy-Org/Wan_2.2_ComfyUI_Repackaged','split_files/vae/wan2.2_vae.safetensors',local_dir=d)
if os.path.abspath(p)!=os.path.abspath(alvo): os.replace(p,alvo)
for lixo in ('split_files','.cache'):
    x=os.path.join(d,lixo)
    if os.path.isdir(x): shutil.rmtree(x,ignore_errors=True)
print('VAE', os.path.getsize(alvo), 'B', 'OK' if os.path.getsize(alvo)==1409400960 else 'DIFERE')
"@
Write-Host "VAE_EXIT=$LASTEXITCODE"

Write-Host "`n== ladder de video: FP16 original vs W4A8 vs W4A4 =="
$d='ComfyUI\models\diffusion_models\'
& .\python_embeded\python.exe -s .\tools\quality_ladder.py `
    --reference "$($d)wan2.2_ti2v_5B_fp16.safetensors" `
    --models "P:\ComfyBench\diffusion_models\wan2.2_ti2v_5B_w4a8.safetensors" `
             "P:\ComfyBench\diffusion_models\wan2.2_ti2v_5B_w4a4.safetensors" `
    --clip umt5_xxl_fp8_e4m3fn_scaled.safetensors --clip-type wan `
    --vae wan2.2_vae.safetensors `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 --steps 20 --size 480 --frames 33 `
    --sampler euler --scheduler simple `
    --out bench\quality_ladder_wan22
Write-Host "LADDER_WAN_EXIT=$LASTEXITCODE"
Write-Host "WAN_LADDER_FIM"
