# Avaliacao em lote, camada 1 (cabecalho, sidecar, analise)

Ordenado por olha-aqui-primeiro. **Nenhum checkpoint aqui esta aprovado**: esta camada
reprova, aponta e preve; ela nao aprova. Ver o cabecalho de `tools/avaliar.py` para o
motivo medido.

| veredito | arquivo | mediana erro | familia | achados |
|---|---|---|---|---|
| OLHAR | `hidream_o1_image_dev_fp8_scaled.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `ltx-2.3-22b-dev-fp8.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `LTX25-distilled-DiT-comfy-w4a4.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `LTX25-distilled-DiT-comfy-w4a8.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors` | - | - | `sobra`, `backend_sem_registro` |
| OLHAR | `MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors` | - | - | `sobra`, `backend_sem_registro` |
| OLHAR | `flux-2-klein-base-4b-fp8.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `krea2_raw_int8_convrot.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `krea2_turbo_int8_convrot.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `krea2_turbo_w4a4.safetensors` | 0.1199 | - | `erro_sem_faixa` |
| OLHAR | `minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `qwen_image_edit_2511_int8_convrot.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `qwen_image_edit_2511_w4a8.safetensors` | 0.0358 | - | `erro_sem_faixa` |
| OLHAR | `wan2.1_vace_1.3B_fp16.safetensors` | - | - | `vace_em_t2v` |
| OLHAR | `wan2.2_animate_14B_int8_convrot.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `gemma4-12b-ltx25-comfy-w4a8.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `gemma4_e2b_it_int8_convrot.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `gemma4_e4b_it_fp8_scaled.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `gemma_3_12B_it_heretic_w4a8.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `qwen3vl_32b_minimax_h3-int4_convrot.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `qwen3vl_4b_fp8_scaled.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| OLHAR | `qwen3vl_4b_w4a8.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `qwen_2.5_vl_7b_w4a4_convrot.safetensors` | - | - | `backend_sem_registro`, `encoder_destravado_nao_medido` |
| SEM VEREDITO | `diffusion_pytorch_model.safetensors` | - | - | - |
| SEM VEREDITO | `diffusion_pytorch_model.safetensors` | - | - | - |
| SEM VEREDITO | `diffusion_pytorch_model_streaming_dmd.safetensors` | - | - | - |
| SEM VEREDITO | `model.safetensors` | - | - | - |
| SEM VEREDITO | `NovaSR.safetensors` | - | - | - |
| SEM VEREDITO | `ema_vae_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `seedvr2_ema_3b_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `seedvr2_ema_7b_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `seedvr2_ema_7b_sharp_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16_5d9b7e00_precompiled.safetensors` | - | - | - |
| SEM VEREDITO | `wav2vec2_large_english_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `s3gen.safetensors` | - | - | - |
| SEM VEREDITO | `t3_cfg.safetensors` | - | - | - |
| SEM VEREDITO | `ve.safetensors` | - | - | - |
| SEM VEREDITO | `DreamShaper_8_pruned.safetensors` | - | - | - |
| SEM VEREDITO | `Juggernaut-XL_v9.safetensors` | - | - | - |
| SEM VEREDITO | `ace_step_1.5_turbo_aio.safetensors` | - | - | - |
| SEM VEREDITO | `hidream_o1_image_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `hunyuan3d-dit-v2-mv-turbo_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `sam3.1_multiplex_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `sd_xl_base_1.0.safetensors` | - | - | - |
| SEM VEREDITO | `sdpose_wholebody_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `stable_audio_3_medium.safetensors` | - | - | - |
| SEM VEREDITO | `svd_xt.safetensors` | - | - | - |
| SEM VEREDITO | `v1-5-pruned-emaonly-fp16.safetensors` | - | - | - |
| SEM VEREDITO | `clip_l.safetensors` | - | - | - |
| SEM VEREDITO | `t5gemma_b_b_ul2.safetensors` | - | - | - |
| SEM VEREDITO | `sigclip_vision_patch14_384.safetensors` | - | - | - |
| SEM VEREDITO | `DA3NESTED-GIANT-LARGE-1.1 .safetensors` | - | - | - |
| SEM VEREDITO | `Depth-Anything-V2-Small-hf.safetensors` | - | - | - |
| SEM VEREDITO | `Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` | - | - | - |
| SEM VEREDITO | `Flux_Realistic_NSFW_02.safetensors` | - | - | - |
| SEM VEREDITO | `beyond-reality-zimage-v2_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `beyond-reality-zimage-v2_native.safetensors` | - | - | - |
| SEM VEREDITO | `capybara_v0.1.safetensors` | - | - | - |
| SEM VEREDITO | `hunyuanvideo1.5_720p_t2v_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_turbo_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `lotus-depth-d-v1-1.safetensors` | - | - | - |
| SEM VEREDITO | `nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_image_edit_2511_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `rt_detr_v4-x-hgnet_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `svdq-int4_r32-z-image-turbo.safetensors` | - | - | - |
| SEM VEREDITO | `void_pass2.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_animate_14B_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_s2v_14B_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_ti2v_5B_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `z_image_de_turbo_v1_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `z_image_turbo_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `zimage-v2-w4a4.safetensors` | 0.1241 | zimage | - |
| SEM VEREDITO | `zimage_deturbo_w4a4.safetensors` | 0.1211 | zimage | - |
| SEM VEREDITO | `zimage_turbo_w4a4.safetensors` | 0.1228 | zimage | - |
| SEM VEREDITO | `ltx23cond_chk_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_chk_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_gw4a8L_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_gw4a8L_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_gw4a8_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_gw4a8_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hbf16_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hbf16_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4cL_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4cL_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4c_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4c_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4sL_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4sL_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4s_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a4s_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a8L_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a8L_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a8_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_hw4a8_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_srx_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23cond_srx_pos.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23condf_neg.safetensors` | - | - | - |
| SEM VEREDITO | `ltx23condf_pos.safetensors` | - | - | - |
| SEM VEREDITO | `film_net_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `moge_2_vitl_normal_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | - | - | - |
| SEM VEREDITO | `Epic-Realism-Unpruned.safetensors` | - | - | - |
| SEM VEREDITO | `Flux_2-Turbo-LoRA_comfyui.safetensors` | - | - | - |
| SEM VEREDITO | `Krea2HighQuality.safetensors` | - | - | - |
| SEM VEREDITO | `Krea2HighResolution4B.safetensors` | - | - | - |
| SEM VEREDITO | `Krea2HighResolution9B.safetensors` | - | - | - |
| SEM VEREDITO | `LTX23_Product_Commercial_LoRA.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2509-Anything2RealAlpha.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2509-Light-Migration.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2509-Relight.safetensors` | - | - | - |
| SEM VEREDITO | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | - | - | - |
| SEM VEREDITO | `RealisticSnapshot-Zimage-Turbov5.safetensors` | - | - | - |
| SEM VEREDITO | `WAN_dr34mj0b.safetensors` | - | - | - |
| SEM VEREDITO | `Wan_facial_t2v.safetensors` | - | - | - |
| SEM VEREDITO | `axelle-z-v3_000000900.safetensors` | - | - | - |
| SEM VEREDITO | `char_Liria_zimage.safetensors` | - | - | - |
| SEM VEREDITO | `detailed_notrigger.safetensors` | - | - | - |
| SEM VEREDITO | `dmd2_sdxl_4step_lora.safetensors` | - | - | - |
| SEM VEREDITO | `flux_lustly-ai_v1_lora.safetensors` | - | - | - |
| SEM VEREDITO | `flux_nfsw_uncensored_lora.safetensors` | - | - | - |
| SEM VEREDITO | `flux_realism_lora.safetensors` | - | - | - |
| SEM VEREDITO | `gemma-3-12b-it-abliterated_heretic_lora_rank64_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `influencer-full-shot-by-stable-yogi-v2.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_darkbrush.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_dotmatrix.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_identity_edit_v1_2.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_kidsdrawing.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_neondrip.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_rainywindow.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_retroanime.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_softwatercolor.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_style_reference.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_sunsetblur.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_turbo_lora_rank_64_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `krea2_vintagetarot.safetensors` | - | - | - |
| SEM VEREDITO | `low_noise_model.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3-22b-distilled-lora-384.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3-id-lora-talkvid-3k.safetensors` | - | - | - |
| SEM VEREDITO | `ltx2-squish.safetensors` | - | - | - |
| SEM VEREDITO | `ltx2.3-ic-watermark-remove-general.safetensors` | - | - | - |
| SEM VEREDITO | `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `pcm_sd15_normalcfg_4step_converted.safetensors` | - | - | - |
| SEM VEREDITO | `qwen-image-edit-2511-multiple-angles-lora.safetensors` | - | - | - |
| SEM VEREDITO | `realNotRealNSFW_v10.safetensors` | - | - | - |
| SEM VEREDITO | `stock_photography_wan22_LOW_v1.safetensors` | - | - | - |
| SEM VEREDITO | `tryon-klein-4b.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | - | - | - |
| SEM VEREDITO | `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors` | - | - | - |
| SEM VEREDITO | `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors` | - | - | - |
| SEM VEREDITO | `Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | - | - | - |
| SEM VEREDITO | `model.safetensors` | - | - | - |
| SEM VEREDITO | `raft_large_C_T_SKHT_V2-ff5fadd5.safetensors` | - | - | - |
| SEM VEREDITO | `byt5_small_glyphxl_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `gemma-3-1b-it-heretic-extreme-uncensored-abliterated.safetensors` | - | - | - |
| SEM VEREDITO | `gemma_3_12B_it_heretic.safetensors` | - | - | - |
| SEM VEREDITO | `gemma_3_12B_it_heretic_fp8_e4m3fn.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3_text_projection_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `qwen3.5_2b_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `qwen3vl_32b_minimax_h3_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `qwen3vl_4b_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_2.5_vl_7b.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_3_4b.safetensors` | - | - | - |
| SEM VEREDITO | `t5xxl_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `t5xxl_fp8_e4m3fn_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | - | - | - |
| SEM VEREDITO | `model.safetensors` | - | - | - |
| SEM VEREDITO | `void_pass1.safetensors` | - | - | - |
| SEM VEREDITO | `RealESRGAN_x4plus.safetensors` | - | - | - |
| SEM VEREDITO | `LTX23_audio_vae_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `LTX23_video_vae_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `ae.safetensors` | - | - | - |
| SEM VEREDITO | `cogvideox_vae.safetensors` | - | - | - |
| SEM VEREDITO | `full_encoder_small_decoder.safetensors` | - | - | - |
| SEM VEREDITO | `hunyuanvideo15_vae_fp16.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_image_layered_vae.safetensors` | - | - | - |
| SEM VEREDITO | `qwen_image_vae.safetensors` | - | - | - |
| SEM VEREDITO | `sd-vae-ft-mse.safetensors` | - | - | - |
| SEM VEREDITO | `vae-ft-mse-840000-ema-pruned.safetensors` | - | - | - |
| SEM VEREDITO | `wan2.2_vae.safetensors` | - | - | - |
| SEM VEREDITO | `wan_2.1_vae.safetensors` | - | - | - |
| SEM VEREDITO | `taeltx2_3.safetensors` | - | - | - |

## hidream_o1_image_dev_fp8_scaled.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## ltx-2.3-22b-dev-fp8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## LTX25-distilled-DiT-comfy-w4a4.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## LTX25-distilled-DiT-comfy-w4a8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors -- OLHAR

- **OLHAR** `sobra` -- 83 bytes depois do ultimo tensor: `safe_open` recusa o arquivo, o leitor de VRAM dinamica aceita
- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors -- OLHAR

- **OLHAR** `sobra` -- 64 bytes depois do ultimo tensor: `safe_open` recusa o arquivo, o leitor de VRAM dinamica aceita
- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## flux-2-klein-base-4b-fp8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## krea2_raw_int8_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## krea2_turbo_int8_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## krea2_turbo_w4a4.safetensors -- OLHAR

- **OLHAR** `erro_sem_faixa` -- mediana do erro efetivo 0.1199, e nao ha faixa medida para esta familia -- sem veredito neste eixo

## minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## qwen_image_edit_2511_int8_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## qwen_image_edit_2511_w4a8.safetensors -- OLHAR

- **OLHAR** `erro_sem_faixa` -- mediana do erro efetivo 0.0358, e nao ha faixa medida para esta familia -- sem veredito neste eixo

## wan2.1_vace_1.3B_fp16.safetensors -- OLHAR

- **OLHAR** `vace_em_t2v` -- checkpoint VACE: em workflow T2V comum o ComfyUI aplica controle constante em forca total e destroi a saida, sem erro. Confira o braco NAO quantizado antes de acreditar em qualquer numero; use `--vace-strength 0`

## wan2.2_animate_14B_int8_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## gemma4-12b-ltx25-comfy-w4a8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## gemma4_e2b_it_int8_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## gemma4_e4b_it_fp8_scaled.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## gemma_3_12B_it_heretic_w4a8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sidecar presente e SEM campo `backend` -- conversao nossa sem registro); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## qwen3vl_32b_minimax_h3-int4_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## qwen3vl_4b_fp8_scaled.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sem sidecar `.quant.json` -- provavelmente de terceiros); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## qwen3vl_4b_w4a8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sidecar presente e SEM campo `backend` -- conversao nossa sem registro); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## qwen_2.5_vl_7b_w4a4_convrot.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sidecar presente e SEM campo `backend` -- conversao nossa sem registro); o backend eager declara as mesmas capacidades e roda matematica dequantizada
- **OLHAR** `encoder_destravado_nao_medido` -- parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta arvore, entao a matematica quantizada pode rodar -- o oposto do padrao de fabrica. LIDO no fonte, nao executado: confirme com `avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por velocidade, entao um numero medido com a trava presa nao vale mais

## Nao coberto por esta camada

- sobra: nao le os bytes de sobra nem diz de onde vieram; so que estao la
- escala: nao confere o VALOR das escalas, so a presenca e a forma
- forma: nao pega escala torta em int8_tensorwise (ver FORMAS_DE_ESCALA) nem se os bits int4 decodificam certo
- resumo: nao conta forwards; o formato declarado por camada pode nao ser o executado
- backend: e o registro de uma conversao passada, nao uma execucao agora
- erro: preve o erro de predicao do modelo, nunca a imagem final; so uma renderizacao decide
- armadilha: so conhece as duas que ja custaram trabalho aqui; nao procura armadilhas novas
- tamanho: nao confere se a fonte no disco ainda e a mesma que foi quantizada
- dispatch: nada aqui carrega o modelo, entao ninguem contou forward quantizado nem
  chamada a `dequantize`. Um arquivo pode passar tudo acima e rodar dequantizado.
  Isto deixou de ser um buraco em 2026-09-01: `tools/avaliar_despacho.py` (camada 2, com
  GPU) carrega pelo caminho normal do ComfyUI e conta. Rode-a antes de acreditar no campo
  `backend` de qualquer sidecar -- ele registra a conversao, nao a execucao de hoje.
- imagem: nenhuma renderizacao. Nenhum corte medido nesta bancada separa usavel de
  inutilizavel -- 0,1837 correta contra 0,2147 destruida; 0,7173 boa contra 0,8255
  destruida. So alguem olhando decide.
- braco de referencia: o arquivo NAO quantizado nao e exercitado NESTA camada. Isto
  deixou de ser um buraco em 2026-09-01: `tools/avaliar_referencia.py` (camada 3, com
  GPU) mede se a referencia responde ao proprio condicionamento, e no par de verdade
  conhecida do Wan separou o destruido do bom por 3,19x. Rode-a antes de acreditar em
  qualquer numero desta tabela.
