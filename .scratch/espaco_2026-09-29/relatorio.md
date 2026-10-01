# Espaço em F: — levantamento de 2026-09-29 (só leitura, nada apagado)

F: 1,9 TB, 7,4 GB livres. Usado: 1947 GiB, dos quais `ComfyUI/models` 1252 GiB (diffusion_models 677,
text_encoders 219, checkpoints 102, unet 73, SEEDVR2 55, loras 39). Dados: `varredura.json`,
`modelos_detalhe.txt`, `duplicatas_conferidas.txt` (hash de 3 amostras de 16 MB por arquivo; antes de apagar,
conferir sha256 completo).

## A. Cópias idênticas dentro de F: (ganho sem perder nada) — ~33 GiB
| GiB livres | o quê |
|---|---|
| 8,5 | `codec.pth` do s2-pro em 6 pastas de `F:/tts_lab` (1,7 cada, conteúdo igual): manter 1 |
| 8,5 | shards do s2-pro repetidos: `tts_lab/noite_v9/checkpoints/s2-pro` = `tts_lab/models/s2-pro`; `release/_teste_reconstrucao/w4g128` = `models/s2-pro-w4-sim` |
| 7,6 | `F:/IA/heretic/F/` = cópia de `F:/IA/heretic/model-0000{1,2}` |
| ~7 | `qwen38-27b-rtx3090/.../W4A16-AutoRound-fast`: 6 dos 7 shards iguais aos do `W4A16-AutoRound` (só o que difere justifica a pasta) |
| 2,7 | `ptbr-audio-lab/runs/lfm/gold_30m_smoke`: `final/model.safetensors` = `checkpoints/checkpoint_0/` |
| 0,9 | `ltx-2.3-spatial-upscaler-x2-1.1` em `models/upscale_models` e `models/latent_upscale_models` |

## B. Cópia local com gêmea idêntica no NAS (`P:\ComfyBench`) — ~31 GiB, com custo
klein4b_braco0/1/2/3 (4 × 7,2), `ltx-2.3_text_projection_bf16` (2,1), `LTX23_audio_vae_bf16` (0,3).
Apagar a local faz o ComfyUI ler do NAS: carga mais lenta e, para DiT grande com dynamic VRAM, o abort de 27/09.
O braco2 é o usado nos testes do lowbit (K1/K2). Recomendo só se o espaço apertar.

## C. Regeráveis (baixar/compilar de novo) — ~106 GiB
| GiB | o quê |
|---|---|
| 52 | `F:/hf-cache/hub/models--Qwen--Qwen3.8-27B` (base original; as quantizações ficam em `qwen38-27b-rtx3090`) |
| 14 | `F:/hf-cache/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf` |
| 21 | `F:/build_torch` (árvore de compilação do PyTorch) |
| 19 | `COMFY_PORTABLE/.scratch/qat_smoke` (checkpoint 11,1 + aluno 7,2 do smoke do QAT; conferir com o plano do QAT antes) |

## D. Fora do laboratório
`F:/SteamLibrary` 149 GiB (jogo).

## E. Modelos do ComfyUI: precisa de decisão sua (originais: regra do projeto proíbe apagar sem ordem)
Há o mesmo modelo em várias formas; manter só as que você usa libera muito. Maiores grupos:
- Qwen Image Edit 2511: bf16 38,0 + int8_convrot 19,1 + nunchaku int4 13,2 + GGUF Q4_K_M 12,3 + w4a8 10,8 = 93 GiB
- Wan 2.2 animate 14B: bf16 32,2 + int8_convrot 17,1; mais 4 × 13,3 fp8 (i2v/t2v high/low) + s2v 15,3 + ti2v 9,3
- LTX 2.3: dev fp8 27,1 + GGUF Q5_K_M 18,1 + Q6_K 16,6; LTX 2.5: mix4x8 12,9 + w4a8 11,7 + w4a4 10,5
- Krea2: turbo bf16 24,5 + turbo int8 12,6 + raw int8 12,6 + turbo w4a4 7,5
- Qwen3-VL 32B MiniMax H3 TE: bf16 48,0 + int4 13,2; MiniMax H3: FL2VA mixed 14,8 + Ref2VA 14,1 + w4a8 11,7
- Gemma 3 12B heretic: bf16 21,9 + fp8 11,9
- Z-Image: turbo bf16, de-turbo bf16, beyond-reality bf16 e native (4 × 11,5) + 3 w4a4
- SEEDVR2: 7b fp16 + 7b sharp fp16 (2 × 15,3) + 7b fp8 7,9
- klein4b: vários braços/transplantes (7,2 cada) além dos 4 com gêmea no NAS
Candidatos naturais: as versões bf16 que já têm quantização validada, e os artefatos de experimento (klein transp,
QAT) que já estão no NAS ou no HF.
