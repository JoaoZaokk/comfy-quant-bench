# Quantizações novas (06/11/2026): OmniGen2, P-H3-Image, FLUX.3 klein, MOSS-Image

RTX 3090 (sm86), ComfyUI v0.43.2 (OmniGen2) e v0.44.1 (P-H3, que o v0.43.2 ainda não tem), servidor por braço, sem custom nodes, 6 imagens por braço (a primeira também aquece o servidor; o tempo usa as outras)
(prompts neutros), PSNR/SSIM contra a referência do mesmo prompt/seed (`analisa.py`). PSNR entre quantizações tem piso de caos ~28 dB e não é qualidade: a avaliação visual é de um revisor só (o dono).

## OmniGen2 (DiT fp16, 1024², 20 passos, DualCFG)
| braço | s/passo | pico VRAM | PSNR vs fp16 (6 imagens) | arquivo |
|---|---|---|---|---|
| fp16 (ref) | 1,42 | 16,8 GB | — | omnigen2_fp16 (7,9 GB) |
| W8A16 (int8 por linha, só pesos) | 1,43 | 13,4 GB | 24,5 a 30,3 dB (SSIM 0,89 a 0,97) | omnigen2_fp16_w8a16 (4,3 GB) |
| W4A16 (Q4_1 grupo 40, só pesos) | 1,77 (+25 %) | 12,0 GB | 16,1 a 24,2 dB (SSIM 0,55 a 0,91) | omnigen2_fp16_w4a16_g40 (2,9 GB) |
| encoder abliterated (DiT fp16) | 1,44 | 16,8 GB | 12,0 a 19,0 dB (o condicionamento muda de propósito) | qwen_2.5_vl_fp16_abliterated (7,5 GB) |
- K=2520 não aceita ConvRot-256, GEMM INT8 nem Q4_1 de grupo 32: INT8 só como W8A16 e W4A16 com grupo 40 (quantizador próprio, `quant_omnigen2.py`).
- Bug achado: com escala/zero em bf16 a dequantização promove fp16×bf16 para fp32 e dá erro de dtype no linear; no fp16 as escalas têm de ser fp16.
- Abliterated: 71 de 824 tensores diferem da fonte (`embed_tokens`, 35 `o_proj`, 35 `down_proj`), bf16→fp16 sem inf/nan.

## P-H3-Image (DiT, 8 passos sem CFG, text encoder e VAE int8_convrot, sem prompt upsampler)
| braço | s/passo 4 MP | s/passo 8 MP | pico VRAM | PSNR vs int8 |
|---|---|---|---|---|
| int8_convrot (ref; o bf16 de 40 GB não cabe) | 1,26 | 3,26 | 22,7 GB | — |
| w4a8 (oficial) | 1,31 | 3,31 | 22,0 GB | 15,1 a 17,6 dB |
| W4A16 (do w4a8, `full_precision_matrix_mult`) | 2,12 (+62 %) | 4,61 (+39 %) | 22,1 GB | 14,5 a 17,2 dB |
- W4A16 não ganha de w4a8 em PSNR e é mais lento na 3090; usa os mesmos pesos de 4 bits. 8 MP funciona na 3090 (22 GB, 37 s com int8).
- `quant_weight_only.py` ganhou suporte a `.comfy_quant` inline .
- P-H3 só existe no ComfyUI a partir da v0.44.0 (#17506): testado num worktree v0.44.1 (checkout v0.44.1 à parte); o vivo (v0.43.2) não roda.

## Nota: fp8 e nvfp4 daqui NÃO são só-pesos
No ComfyUI os formatos fp8 e nvfp4 quantizam também a ATIVAÇÃO (`quantize_input` padrão True em `QUANT_ALGOS`); sem `input_scale` a escala é dinâmica. Na 3090 roda emulado
(quantiza/dequantiza, matmul em bf16). Logo: o FP8 e o NVFP4 feitos aqui são fp8 W8A8 e fp4 W4A4, ambos com escala de ativação dinâmica; o NVFP4 oficial da BFL usa
escala ESTÁTICA calibrada (`input_scale`). Só o W4A16 (flag `full_precision_matrix_mult`) mantém ativação em bf16. 
O ganho do meu NVFP4 sobre o oficial vem, portanto, da escala dinâmica de ativação (e de a BFL manter blocos bf16 nas pontas, que não mexi), não de ser só-pesos.

## FLUX.3 klein (DiT, 1024², 16 passos, cfg 1; text encoder int8 e VAE iguais em todos)
| braço | exec_s | pico VRAM | PSNR vs bf16 (6 imagens) | origem |
|---|---|---|---|---|
| bf16 (ref) | 7,62 | 16,5 GB | — | BFL |
| int8_convrot | 5,23 (-31 %) | 11,5 GB | 20,5 a 31,3 dB | BFL |
| NVFP4 oficial (com input_scale, BFL) | 7,96 | 11,1 GB | 16,3 a 24,9 dB | BFL |
| FP8 e4m3 (escala por tensor, ativação dinâmica) | 7,95 | 11,5 GB | 24,4 a 30,7 dB | meu (`quant_fp.py`) |
| NVFP4 (ativação dinâmica) | 8,12 | 9,4 GB | 15,5 a 23,2 dB | meu |
| W4A16 só-pesos (Q4_1 g32) | 11,85 (+55 %) | 9,9 GB | 18,6 a 22,2 dB | meu |
| **W4A4 ConvRot (nativo na 3090, `quant_fp.py --formato w4a4`)** | **4,16 (−45 %)** | **9,0 GB** | 10,6 a 18,9 dB | meu |
- W4A16 se aplica ao FLUX.3 (258 camadas, K = 3072/9216, todos aceitos). Sem `full_precision_matrix_mult` o caminho quantizado do comfy-kitchen 0.2.40 com grupo 32 dá imagem preta (NaN); com a flag funciona.
- fp8 e nvfp4 rodam na 3090 por emulação (dequantiza para bf16): sem ganho de velocidade (+4 a +6 %), só de VRAM. O int8 é o único mais rápido que o bf16.
- O NVFP4 oficial calibrado (BFL) ficou ~1 dB acima do meu NVFP4 em PSNR; o FP8 superou os dois 4-bit.

## MOSS-Image (dmd8, 1024², 8 passos, cfg 1; text encoder int8)
| braço | exec_s | pico VRAM | PSNR vs bf16 |
|---|---|---|---|
| bf16 (ref) | 5,26 | 20,3 GB | — |
| int8_convrot (oficial/local) | 3,46 (-34 %) | 15,8 GB | 11,1 a 15,9 dB |
| FP8 (ativação dinâmica) | 5,34 | 15,8 GB | 17,3 a 25,9 dB |
| NVFP4 (ativação dinâmica) | 5,45 | 13,9 GB | 12,3 a 16,2 dB |
| W4A16 só-pesos (Q4_1 g32) | 7,34 (+40 %) | 14,4 GB | 15,4 a 17,3 dB |
- Não há FP8/NVFP4 calibrado do MOSS para baixar (Comfy-Org só publica bf16 e int8). O int8 do dmd8 ficou mais longe do bf16 que o FP8 em PSNR: o modelo é destilado em 8 passos e ruidoso; ordenar por PSNR aqui não diz qual é melhor, precisa de olho humano.

## Avaliação visual (um revisor, 6 imagens por braço; não é medição)
- MOSS: INT8 e FP8 bons; W4A16 "muito bom", ao nível do INT8; uma imagem do W4A16 tem cabelo atravessando um chapéu, ausente no bf16 da mesma seed.
- FLUX.3: o NVFP4 oficial (BFL, `input_scale` estático) tem aparência "artificial"; o NVFP4 de escala dinâmica de ativação (este trabalho) segue o estilo do bf16. W4A4 ConvRot: mãos e estrutura se desfazem (o mesmo que os demais W4A4), possivelmente só servindo para paisagens.
- OmniGen2: modelo antigo, abaixo do que foi anunciado; o encoder abliterated muda a conduta do condicionamento de forma visível e desigual entre prompts (causa não investigada).
- P-H3-Image: aceitável, estilo não preferido.

## Publicado (HF, conta JoaoZaokk; cartões em `bench/hf/`)
- `MOSS-Image-SFT-8steps-FP8-NVFP4-W4A16` (Apache-2.0), `OmniGen2-W8A16-W4A16` (Apache-2.0).
- `FLUX.3-klein-FP8-NVFP4-W4A16-W4A4` (FLUX Kommunity License, não comercial; LICENSE, NOTICE com aviso de atribuição e de modificação).
- `P-H3-Image-W4A16` (MiniMax H3 Community License: direitos só fora de UE, Reino Unido, Coreia do Sul e EUA; LICENSE e NOTICE no repo).
- `Qwen2.5-VL-3B-abliterated-fp16-ComfyUI` (Qwen Research License, não comercial; abliteração de huihui-ai, aqui só conversão para fp16).
- Ferramentas: `tools/quant_fp.py` (fp8, nvfp4, w4a16, w4a4), `tools/quant_omnigen2.py` (W8A16, W4A16 de grupo 40), suporte a `.comfy_quant` inline em `tools/quant_weight_only.py`; testes `test_quant_fp.py` e `test_quant_omnigen2.py`.
- Os rótulos `quantization` dos arquivos fp8, nvfp4 e w4a4 publicados foram corrigidos no cabeçalho (cópias; o tamanho do cabeçalho não muda, só o texto): antes diziam "só pesos".

## Critério escrito antes de medir
Pedido: (1) OmniGen2: W4A16, INT8 e text encoder abliterated; (2) P-H3-Image: W4A16; (3) FLUX.3 klein: ver se W4A16 se aplica;
(4) qualidade de NVFP4 e FP8 (3090) em FLUX.3 e MOSS, com pesos calibrados baixados quando existirem.

## Fatos verificados antes de começar (06/11)
- P-H3-Image já vem em bf16, int8_convrot e w4a8 (PrunaAI, config por camada em tensores `.comfy_quant` inline). O W4A16 sai do w4a8
  oficial: mesmos pesos de 4 bits, ativações BF16 (`full_precision_matrix_mult`). Não requantiza do bf16.
- OmniGen2 (DiT fp16 7,9 GB, hidden 2520): K=2520 não é múltiplo de 256 (ConvRot) nem de 32 (AWQ Q4_1 do gguf) nem de 16 (GEMM INT8).
  Só `linear_2` (K=10240) e `norm1.linear` (K=1024) aceitam os formatos prontos. W8A8/ConvRot e W4A16-g32 não se aplicam à maior parte do modelo.
- FLUX.3 klein: a BFL publica bf16, int8_convrot e **nvfp4** oficial; sem fp8. O DiT bf16 (11 GB) tem profile `flux_1` nas ferramentas.
- MOSS: Comfy-Org publica bf16 e int8_convrot; não há fp8/nvfp4 pronto.
- Na sm86 (3090) fp8 e nvfp4 não têm kernel nativo (comfy-kitchen: "emulated ops"): rodam por dequantização para bf16, sem ganho de velocidade.

## Aprovação de conversão (por arquivo)
1. Mesmos nomes de tensor da fonte; tensores de peso copiados byte a byte quando só muda a configuração (conferência por amostra + hash).
2. Sidecar `.quant.json`, escrita atômica `.partial` → rename, nunca sobrescrever a fonte.
3. Quantizações novas: erro relativo do peso por camada gravado no sidecar.

## Aprovação de qualidade (por modelo), na GPU 05, um modelo por servidor, lock próprio
- Mesmo grafo, prompt e seed; 4 imagens por braço (prompts neutros), servidor quente (1 aquecimento descartado).
- Referência: o maior formato que cabe (OmniGen2 fp16; P-H3 int8_convrot, pois o bf16 de 40 GB não cabe em 24 GB; FLUX.3 bf16; MOSS bf16).
- Métricas: PSNR e SSIM contra a referência (PSNR/SSIM por imagem), tempo do KSampler, pico de VRAM.
  A avaliação visual é do dono. PSNR não é qualidade: serve para ordenar braços, não para aprovar.
- P-H3: 4 MP (edição) e 8 MP (texto→imagem, tamanho recomendado) — o README diz 8 MP para texto→imagem e 4 ou 8 MP para edição.
- Velocidade: só vale como ganho se a diferença passar a dispersão (máx − mín) em n = 3.
- NVFP4/FP8: sem calibração própria. NVFP4 oficial da BFL (FLUX.3) entra como está. Onde não existe (MOSS, FLUX.3 fp8) só se gera
  versão sem calibração. ERRO deste critério, achado depois: fp8 e nvfp4 do ComfyUI quantizam também a ativação (ver a nota acima), então não são só-pesos; só o w4a16 mantém ativação em bf16.

## Limites conhecidos
- 4 imagens por braço não separam diferenças pequenas; PSNR entre quantizações diferentes tem piso de caos (~28 dB).
- A 3090 roda fp8/nvfp4 emulados; qualidade medida aqui vale para a GPU de maior capacidade, tempo não.
