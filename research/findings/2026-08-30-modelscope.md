# Varredura ModelScope 2026-08-30

## Novo e acionável

### Wan2.2-Animate-2-14B-OrbitQuant-W4A4
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/ApacheOne/Wan2.2-Animate-2-14B-OrbitQuant-W4A4
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Wan 2.2 14B para animação, quantizado em W4A4 com OrbitQuant. Tags: `wan`, `wan-animate-2`, `orbitquant`, `w4a4`, `4bit`, `triton`, `video-generation`
- **por que muda algo aqui:** W4A4 nativo em vídeo + triton kernel (ativação 4-bit), exatamente o escopo desta bancada. OrbitQuant é strategy de quantização desconhecida aqui.
- **o que NÃO foi conferido:** não baixei config.json; método exato do OrbitQuant; se realmente roda triton em sm86 ou é fallback eager

### Z-Image-Turbo-SVDQuant-NVFP4
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/ultranationalism/Z-Image-Turbo-SVDQuant-NVFP4
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Z-Image Turbo texto-para-imagem, quantizado com SVDQuant em NVFP4. Tags: `text-to-image`, `diffusion`, `svdquant`, `nvfp4`, `w4a4`, `quantization`
- **por que muda algo aqui:** SVDQuant com ativação W4A4 em difusão de imagem. Este repo já usa SVDQuant; Z-Image é modelo local. NVFP4 é ativação reduzida.
- **o que NÃO foi conferido:** se a tag w4a4 é ativação 4-bit real ou marketing; não inspeção de metadados do safetensors

### MiniMax-H3-svdquant-nvfp4_r32
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/ModelsLab/MiniMax-H3-svdquant-nvfp4_r32
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** MiniMax-H3 texto-para-vídeo, quantizado com SVDQuant, NVFP4 ativação. Tags: `svdquant`, `w4a4`, `nvfp4`, `video`, `text-to-video`, `quantized`. Grupo r32.
- **por que muda algo aqui:** SVDQuant W4A4 em video-gen, que está aberto no ticket 08 deste repo. MiniMax-H3 é modelo estabelecido em vídeo.
- **o que NÃO foi conferido:** precisão numérica do NVFP4 vs W4A4; performance em 24GB (Classe H3 é 14-18GB em fp16)

### Wan22-i2v-w4a4
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/XXXXinXXXXX/Wan22-i2v-w4a4
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Wan 2.2 image-to-video, W4A4. Owner `XXXXinXXXXX` (possível pseudônimo). 1439 downloads (maior desta varredura).
- **por que muda algo aqui:** W4A4 em image-to-video (subconjunto de vídeo). O número de downloads sugere aceitação prática.
- **o que NÃO foi conferido:** tags do metadado (nenhuma registrada); qual framework (nunchaku? comfy?); se realmente roda ativação 4-bit ou weight-only

## VISTO — sem tags conferidas

### ltx-video-ov-int8
- **nível:** VISTO
- **url:** https://modelscope.cn/models/rainhenry/ltx-video-ov-int8
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** LTX-Video em INT8. Nome sugere OpenVINO backend.
- **por que muda algo aqui:** INT8 quantização em LTX-Video, que esta bancada mede (CORTIQ_LTX25_HANDOFF.md)
- **o que NÃO foi conferido:** metadados, backend real, precisão numérica

### Krea-2-Turbo-W4A4-Nunchaku
- **nível:** VISTO
- **url:** https://modelscope.cn/models/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Krea 2 Turbo com W4A4 sob Nunchaku. Apenas 3 downloads.
- **por que muda algo aqui:** W4A4 + Nunchaku, o stack de quantização ativo nesta bancada. Krea é modelo de imagem.
- **o que NÃO foi conferido:** se realmente usa SVDQuant ou outro backend; quais layers estão quantizados

### Wan2.2-TI2V-5B-Diffusers-int4-AutoRound
- **nível:** VISTO
- **url:** https://modelscope.cn/models/Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Wan 2.2 5B text-to-image-to-video, INT4 com AutoRound (Intel's quantization framework). 91 downloads.
- **por que muda algo aqui:** INT4 em vídeo por rota diferente (AutoRound, não SVDQuant/Nunchaku). Pode indicar alternativa.
- **o que NÃO foi conferido:** ativação quantizada ou weight-only; performance; compatibilidade ComfyUI

## Ausências (com controle)

### Consulta "LTX-Video 量化" (LTX-Video quantized)
- Resultado: **0 achados**
- Controle: "Qwen" retornou 30 resultados, logo a busca não é cega
- Conclusão: não há modelo público documentado com "LTX-Video quantized" chinês combinados no ModelScope

### Consulta "GPTQ 扩散" (GPTQ diffusion)
- Resultado: **0 achados**  
- Controle: "扩散" sozinho retornou 1 resultado (CLIP FP8), logo o instrumento enxerga
- Conclusão: GPTQ não circula como quantização de difusão no ModelScope; circula em LLM apenas

## Consultas rodadas

| # | Consulta | Resultados | Primário |
|---|---|---|---|
| 1 | Controle: Qwen | 30 | baseline (API funciona) |
| 2 | 视频生成 量化 | 1 | SengFuAIAgen (não video model) |
| 3 | 扩散 量化 | 1 | CLIP FP8 (vision encoder) |
| 4 | LTX-Video | 30 | LTX (oficial Lightricks) + GGUF + várias versões |
| 5 | W4A4 | 30 | Qwen3-32B-W4A4 (LLM, fora escopo), mas achei Wan/Krea W4A4 vídeo |
| 6 | SVDQuant | 12 | Z-Image-Turbo, MiniMax-H3, Krea quantizadas |
| 7 | nunchaku | 30 | nunchaku-qwen-image-*, nunchaku-flux.* (difusão quantizada) |
| 8 | HunyuanVideo | 30 | HunyuanVideo oficial + GGUF, nenhum W4A4/SVDQuant |
| 9 | CogVideoX | 30 | CogVideoX oficial, nenhum quantizado achado |
| 10 | INT4 diffusion | 6 | Wan2.2 Intel AutoRound, FLUX.1 torchao, Z-Image MNN |

## Fontes que falharam

Nenhuma. ModelScope respondeu 200 em todas as 10 consultas + 7 detalhes.

## O que não foi coberto

- **Consultas restantes** (ticket de ideias seção 8: Qwen-VL 量化, 显存 优化, LTX-2.5 específico)
- **Modelos não encontrados** porque a busca por nome é substring match: nomes parciais podem devolver 0, não ausência real
- **Inspections de artefato** (config.json, quantization_config.json, README): obtive só metadados da API, não file-tree
- **Gitee**: não tocada nesta execução (próxima fila: "LTX-Video quantizado no Gitee")
- **Confirmar** se "w4a4" nas tags significa ativação real ou é marketing — requer download do safetensors + leitura de metadados
