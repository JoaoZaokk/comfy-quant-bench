# Wan 2.2 TI2V 5B: calibra em ativacao real e converte. O 5B primeiro porque e um arquivo so,
# faz t2v sozinho e cabe residente -- ciclo completo mais barato da familia.
#
# O perfil `wan_2_2` foi conferido dos DOIS lados contra o `wan2.2_animate_14B_int8_convrot` do
# Comfy-Org: 480 contra 480, zero sobrando de cada lado. No ti2v_5B ele degrada sozinho para 300,
# que e o numero certo -- esse modelo nao tem `cross_attn.k_img`/`v_img`.
#
# Saida em P:, nao em F:. F: tem 15 GB e apagar original e proibido; P: tem 2,1 TB e ja esta
# montado no extra_model_paths.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES = '0'
$env:COMFY_BYPASS_WMI_UNAME = '1'

Write-Host "== calibracao =="
& .\python_embeded\python.exe -s .\tools\calibrate_activations.py `
    --model wan2.2_ti2v_5B_fp16.safetensors `
    --profile wan_2_2 `
    --clip umt5_xxl_fp8_e4m3fn_scaled.safetensors --clip-type wan `
    --prompt-file bench\prompts_zimage_runtime.txt `
    --seeds 1 --steps 20 --size 480 --frames 33 `
    --out calib\wan22_ti2v_5b.calib.pt
Write-Host "CALIB_EXIT=$LASTEXITCODE"

Write-Host "`n== conversao W4A8 (todas as camadas) =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input ComfyUI\models\diffusion_models\wan2.2_ti2v_5B_fp16.safetensors `
    --calibration calib\wan22_ti2v_5b.calib.pt `
    --promote-error 0.0 --budget 1.0 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\wan2.2_ti2v_5B_w4a8.safetensors
Write-Host "W4A8_EXIT=$LASTEXITCODE"

Write-Host "`n== conversao W4A4 pura =="
& .\python_embeded\python.exe -s .\tools\quant_mixed.py `
    --input ComfyUI\models\diffusion_models\wan2.2_ti2v_5B_fp16.safetensors `
    --analysis calib\wan22_ti2v_5b.analysis.json `
    --somente-w4a4 --uncalibrated fail `
    --output P:\ComfyBench\diffusion_models\wan2.2_ti2v_5B_w4a4.safetensors
Write-Host "W4A4_EXIT=$LASTEXITCODE"
Write-Host "WAN_CONVERSAO_FIM"
