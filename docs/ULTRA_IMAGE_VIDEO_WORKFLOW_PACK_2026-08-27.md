# ULTRA IMAGE/VIDEO workflow pack — 2026-08-27

## Resultado

Foi criado um pacote local em:

`ComfyUI/user/default/workflows/ULTRA_IMAGE_VIDEO/`

Os seis workflows novos passam na auditoria local sem node ou modelo ausente:

- `WAN22_T2V_14B_4STEP_DUALGPU_3090_3080TI.json`
- `WAN22_I2V_14B_4STEP_DUALGPU_3090_3080TI.json`
- `QWEN_EDIT_2511_NUNCHAKU_FP8_DUALGPU_3090_3080TI.json`
- `FLUX_FILL_INPAINT_Q8_DUALGPU_3090_3080TI.json`
- `FLUX_FILL_OUTPAINT_Q8_DUALGPU_3090_3080TI.json`
- `UPSCALE_PHOTO_4X_3080TI_FAST.json`

Os arquivos originais não foram alterados.

## Divisão de GPU

| Família | RTX 3090 / cuda:0 | RTX 3080 Ti / cuda:1 |
|---|---|---|
| Wan 2.2 T2V/I2V | DiT high/low FP8, com DisTorch2 limitado a 17 GB e offload seletivo para RAM | UMT5 FP8 e VAE |
| Qwen Edit 2511 | DiT Nunchaku INT4 rank 256 | encoder Qwen VL FP8 e VAE |
| FLUX Fill | DiT GGUF Q8 | CLIP-L, T5 XXL FP8 e VAE |
| Upscale rápido | livre | RealESRGAN x4 em tiles |

O loader simples do Wan chegou a 23,5 GB na 3090 e falhou ao solicitar mais 300 MB. Os workflows finais usam `UNETLoaderDisTorch2MultiGPU` com `virtual_vram_gb=17.0`, `donor_device=cpu` e `eject_models=true`. Esse perfil completou T2V e I2V sem OOM.

## Modelos adicionados

- `diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` — 14,29 GB
- `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` — 1,23 GB
- `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` — 1,23 GB
- `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` — 9,38 GB
- `unet/flux1-fill-dev-Q8_0.gguf` — 12,73 GB

O instalador usa o cache em `.scratch/hf_cache` e cria hardlinks para as pastas do ComfyUI. O cache e o modelo instalado compartilham os mesmos blocos no volume F:, portanto não consomem o dobro do espaço.

## Testes executados

1. Auditoria final: 65 workflows parseados, zero erro de JSON. Os seis novos workflows têm zero referência ausente. Os seis workflows antigos ainda marcados com faltas são independentes deste pacote.
2. Qwen FP8: encoder carregado e texto codificado na RTX 3080 Ti. Uso observado: 11,27 GB na GPU 1 e 355 MB na GPU 0.
3. FLUX Fill Q8: inpaint completo de smoke test, 256×256, sem erro. Uso observado após a geração: 12,5 GB na GPU 0 e 8,4 GB na GPU 1.
4. Wan 2.2 T2V: MP4 válido, 256×256, 17 frames, 8 fps, 2,125 s.
5. Wan 2.2 I2V: MP4 válido, 256×256, 17 frames, 8 fps, 2,125 s.
6. RealESRGAN: imagem 768×768 ampliada para 3072×3072 na RTX 3080 Ti.

Saídas dos smoke tests:

- `ComfyUI/output/ULTRA_WAN/SMOKE_T2V_17F_DISTORCH_00001_.mp4`
- `ComfyUI/output/ULTRA_WAN/SMOKE_I2V_17F_DISTORCH_00001_.mp4`
- `ComfyUI/output/ULTRA_FILL/SMOKE_FLUX_Q8_00001_.png`
- `ComfyUI/output/ULTRA_UPSCALE/SMOKE_REALESRGAN_3080TI_00001_.png`

## Upscaler DAT

O `4xRealWebPhoto_v4_dat2.pth` carrega no loader normal, mas o node `multiGPU_ImageUpscaleWithModelMultiGPU` tenta reconstruir a arquitetura DAT com dimensões padrão incompatíveis. O teste real falhou com chaves inesperadas e `size mismatch`. Por isso o workflow rápido usa `RealESRGAN_x4plus.safetensors`, que completou o teste. Para máxima restauração fotográfica, os workflows SeedVR2 existentes continuam sendo a opção de qualidade.

## Launcher recomendado

- Qwen Edit, FLUX Fill e upscale: `run_nvidia_gpu_8190_ultra_image.bat`
- Wan e demais vídeos longos: `run_nvidia_gpu_8190_ultra_video.bat`

O launcher de vídeo usa `--cache-none` para não manter um DiT anterior ocupando VRAM. O de imagem usa `--cache-ram` para acelerar repetições. Ambos mantêm `--disable-dynamic-vram`, necessário para os loaders Nunchaku atuais.

## Licença do FLUX Fill

O FLUX.1 Fill dev usa a FLUX.1 dev Non-Commercial License. A licença diz que os outputs podem ser usados comercialmente, mas restringe o uso do próprio modelo a finalidades não comerciais/não produtivas sem licença adicional. Para um pipeline de anúncios em produção, trate os workflows FLUX Fill como avaliação local até confirmar uma licença comercial da Black Forest Labs. Qwen Image Edit 2511 e Wan 2.2 estão publicados sob Apache 2.0 e são as rotas mais simples para o pipeline comercial.

## Reprodutibilidade

- Gerar novamente o pacote: `python_embeded\python.exe -s tools\build_ultra_workflow_pack.py`
- Retomar/verificar downloads: `python_embeded\python.exe -s tools\download_ultra_workflow_models.py`
- Auditar workflows: `python_embeded\python.exe -s tools\audit_workflows.py --output docs\workflow-audit-2026-08-27-ultra-pack-final.json`
