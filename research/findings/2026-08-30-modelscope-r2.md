# Varredura ModelScope 2026-08-30 #2

## Novo e acionável

### LTX-2.5 NVFP4 para ComfyUI
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/hf/BennyDaBall-LTX-2.5-22b-distilled-nvfp4-comfy-v2
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** LTX-2.5 text-to-video 22B destilado, NVFP4 4-bit quantizado. Workflow JSON incluído para ComfyUI.
- **por que muda algo aqui:** LTX-2.5 é alvo do projeto; aqui está já quantizado em NVFP4 e pronto para uso em ComfyUI com exemplo de workflow.
- **o que NÃO foi conferido:** se o workflow roda de verdade; accuracy vs fp32

### Flux difusão SVDQuant
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/claudeli1234/flux-svdq-w4a4
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Flux (image diffusion) em SVDQuant com W4A4 (weight 4-bit, activation 4-bit). Dois safetensors: transformer quantizado + unquantized layers.
- **por que muda algo aqui:** SVDQuant + W4A4 em modelo visual; exemplo concreto de estratégia mista (nem tudo quantizado).
- **o que NÃO foi conferido:** qual backend resolveu o kernel; accuracy vs source

### MiniMax-H3 vídeo NVFP4 + ConvRot
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** MiniMax-H3 (video generation) em 5 variantes: INT8 ConvRot, mixed INT4/INT8 ConvRot, NVFP4. 13 safetensors, modelos FL2VA e Ref2VA inclusos.
- **por que muda algo aqui:** ConvRot em produção fora deste repo; NVFP4 como alternativa; múltiplas quantizações no mesmo card.
- **o que NÃO foi conferido:** qual ConvRot é nativo vs caiu para dequantizado; resultado visual

### Wan2.2 I2V W4A4
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/junhaowu/Wan2.2-I2V-A14B-W4A4
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Wan2.2 image-to-video 14B em W4A4 nativo (weight+activation ambos 4-bit). Configuração JSON presente.
- **por que muda algo aqui:** Wan2 I2V quantizado no formato que o projeto estuda; raro em repositório central.
- **o que NÃO foi conferido:** qual backend roda; latência vs fp32

### Wan2.1 I2V W4A4 (nunchaku preparado)
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/XXXXinXXXXX/wan21_i2v_w4a4_r128_fp4_smooth085_transformer
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Wan2.1 image-to-video, W4A4 (r128, smooth 0.85). Configuração presente. De mesmo autor do Wan22-i2v-w4a4 que já está no ledger.
- **por que muda algo aqui:** Segunda geração da mesma linha; diferente grupo size e smooth factor; padrão para calibração mista.
- **o que NÃO foi conferido:** diferença de qualidade vs versão r256 já conhecida

### Qwen3-VL text encoder para MiniMax-H3 NVFP4
- **nível:** CONFERIDO
- **url:** https://modelscope.cn/models/Abiray/Qwen3-VL-32B-Heretic-MiniMax-H3-nvfp4-ComfyUI
- **fonte:** modelscope
- **é espelho de:** não
- **o que é:** Qwen3-VL 32B encoder (base + tail) em NVFP4, integrado para MiniMax-H3. Safetensors de 3 componentes.
- **por que muda algo aqui:** Text encoder já quantizado; pipeline de vídeo MiniMax completo em uma organização (visual + text).
- **o que NÃO foi conferido:** interop com MiniMax-H3 acima; atualização se há matching de formato

## Ausências (com controle)

AUSENCIA  consulta="LTX-Video"  fonte=modelscope  data=2026-08-30
          controle="Qwen" -> 40 resultados, logo a busca enxerga
          apelidos tentados: LTX-Video, LTX-2, LTX-2.5, LTX25
          Achado sob "LTX-2.5", mas não sob o nome histórico. Renomeiação de projeto.

## Consultas rodadas

- w4a4, W4A4 (duplicata, mesmos resultados)
- OrbitQuant (2 resultados)
- SVDQuant (12 resultados)
- nvfp4, NVFP4 (duplicata, mesmos resultados)
- mix4x8 (1 resultado)

Parado em 5 de ~12 termos planejados; continuação prioritária: LTX-2.5, HunyuanVideo, CogVideoX 量化, 视频 量化, donos produtivos (ApacheOne, ModelsLab).

## Fontes que falharam

Nenhuma. ModelScope operacional; latência ~600ms por consulta.
