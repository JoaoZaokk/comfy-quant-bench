# Varredura 2026-08-30 #1

## Novo e acionável

### SVDQuant para difusão com W4A4 - Krea-2 em ComfyUI
- **nível:** CONFERIDO
- **url:** https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI
- **fonte:** huggingface
- **é espelho de:** não
- **o que é:** Modelo Krea-2 quantizado com SVDQuant em INT4 e W4A4 nativo. Etiquetas confirmam "w4a4, svdquant, int4, comfyui". Atualizado 2026-08-24.
- **por que muda algo aqui:** Exemplo executável de W4A4 fora deste repo aplicado a geração de imagem. Prova conceitual para a arquitetura Krea em ComfyUI.
- **o que NÃO foi conferido:** Se a conversão usou ConvRot ou kernel alternativo; se o treinamento foi com calibração em ativação real.

### LTX-2.5 quantizado em GGUF
- **nível:** CONFERIDO
- **url:** https://huggingface.co/joeygambino/LTX-2.5-Quantized
- **fonte:** huggingface
- **é espelho de:** não
- **o que é:** LTX-2.5 (video-to-text, áudio-vídeo) em formato GGUF quantizado. Público desde 2026-08-21. Etiquetas confirmam "gguf, quantized, video-generation, text-to-video, ltx-2.5".
- **por que muda algo aqui:** LTX-2.5 é vídeo nativo. GGUF é quantização de peso; se for INT4, cruza com a necessidade de modelos de vídeo quantizados da bancada.
- **o que NÃO foi conferido:** Bits da quantização (GGUF não explicita no metadado); se suporta Gemma ou outro text encoder quantizado.

### SVDQuant INT4 para Krea Turbo
- **nível:** CONFERIDO
- **url:** https://huggingface.co/Patil/krea-turbo-svdquant
- **fonte:** huggingface
- **é espelho de:** não
- **o que é:** Krea-2 Turbo quantizado com SVDQuant INT4. Atualizado 2026-06-26. Etiquetas: "svdquant, int4, low-vram, text-to-image".
- **por que muda algo aqui:** Prova outro modelo gerador (Krea Turbo) com SVDQuant. A tag "low-vram" é relevante para esta bancada.
- **o que NÃO foi conferiu:** Se W4A4 (só INT4 declarado); se é espelho ou conversão original.

## Ausências (com controle)

**AUSENCIA  consulta="LTX-Video quantized"  fonte=huggingface  data=2026-08-30**
          controle="Qwen" -> 10 resultados, logo a busca enxerga

**AUSENCIA  consulta="HunyuanVideo GPTQ"  fonte=huggingface  data=2026-08-30**
          controle="Qwen" -> 10 resultados (HunyuanVideo base encontrado, mas nenhum GPTQ declarado em tags)

**AUSENCIA  consulta="W4A4 diffusion"  fonte=huggingface  data=2026-08-30**
          controle="Qwen" -> 10 resultados (só W4A4 fora desta bancada apareceu: RedHatAI/gemma-3-27b-it-quantized.w4a16, LLM não difusão)

**AUSENCIA  consulta="nunchaku quantized"  fonte=huggingface  data=2026-08-30**
          controle="Qwen" -> 10 resultados (nunchaku SVDQuant não retornou nada; nunchaku é ferramenta, não modelo)

## Consultas rodadas

1. "LTX-Video quantized" → 0
2. "video generation 4-bit" → 0
3. Controle "Qwen" → 10 ✓
4. "HunyuanVideo" → 5 (nenhum quantizado)
5. "CogVideoX" → 5 (nenhum quantizado)
6. "SVDQuant" → 5 ✓
7. "LTX-Video GPTQ" → 0
8. "HunyuanVideo GPTQ" → 0
9. "W4A4 diffusion" → 0
10. "LTX-2.5 quantized" → 1 ✓
11. "Mochi quantization" → 0

## Fontes que falharam

Nenhuma; huggingface operacional em todos os endpoints testados (~180ms por query).

---

## O que NÃO foi coberto

- **ModelScope**: Nenhuma consulta neste relatório. É a fonte chinesa prioritária do AGENTS.md seção 2, onde pode estar quantização de vídeo não traduzida.
- **Gitee**: Nenhuma busca. AGENTS.md seção 4 lista `ComfyUI 量化`, `LTX-Video 量化` como combinações ainda não rodadas.
- **LoRA quantizado em difusão**: Encontrado "LoRA quantized model" genérico, mas nenhum especificamente de vídeo com LoRA sobre peso quantizado (ticket aberto: custa acurácia?).
- **Comunidade chinesa ComfyUI**: `gitee.com/comfyui-cn`, `gitee.com/ComfyUI-Extensions` não testadas (Gitee não tem API de busca confiável; WebSearch necessário).
- **Qwen-VL quantizado**: Não rodado. No AGENTS.md seção 8 como tarefa.
- **Aceleração sm86/Ampere**: Nenhuma busca por SageAttention, FlashAttention, CUDA graphs em modelos de vídeo.
