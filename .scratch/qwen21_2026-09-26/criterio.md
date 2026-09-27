# Qwen-Image-2.1: BF16 junto x Comfy-Org e quantizações contra BF16 (critério escrito antes de medir, 26/09/2026)

## 1. Junção dos shards x BF16 da Comfy-Org
- `tools/junta_shards_safetensors.py` juntou `P:\ComfyBench\originais\Qwen-Image-2.1\transformer` (2 shards diffusers,
  297 tensores) em `qwen_image_2.1_bf16_junto_dos_shards.safetensors` (14.230.284.784 B).
- Comparar com `qwen_image_2.1_bf16.safetensors` da Comfy-Org (14.230.280.616 B, declarado) por
  `tools/compara_checkpoints_byte.py`: arquivo, bytes por tensor, chaves/formas.
- Expectativa: mesmos tensores com bytes iguais (Comfy-Org reembalou sem converter). Qualquer diferença de chave
  (fusão de gate/up, renomeação) ou de bytes é registrada como achado.
- Teste funcional: os dois arquivos carregam no `UNETLoader` e, se os tensores forem iguais, a imagem sai
  idêntica (PSNR infinito) com mesma seed.

## 2. Quantizações contra BF16
- 6 prompts (texto em letreiro, retrato, pôster tipográfico, paisagem, produto, contagem com mão) × seeds 42 e 7,
  1024², 25 passos, cfg 1, euler/simple (template oficial), encoder `qwen3vl_8b_w4a8` fixo, mesmo VAE.
- DiTs: bf16 (Comfy-Org, referência), bf16junto, int8_convrot (Comfy-Org), mixed_balanced W4A8 (NidAll),
  int4 r128 SVDQuant (mesmertech, via `comfy-qwen21-nunchaku`).
- Métricas por imagem contra o BF16 da mesma seed/prompt (`tools/metricas_imagem.py`): PSNR, SSIM, MS-SSIM e grão.
  LPIPS fica de fora (pesos VGG não estão em cache; baixar precisa de autorização).
- Mesma seed com trajetória divergente derruba PSNR/SSIM sem defeito visível: as métricas ordenam fidelidade à
  trajetória do BF16, não qualidade absoluta. Por isso também registro it/s e VRAM, e olho as imagens SFW.
- Previsão: int8 > mixed W4A8 > int4 em fidelidade; int4 mais rápido; mixed com menos VRAM.

## 3. Nossas quantizações (adendo escrito antes de gerar/medir, 26/09 ~18:55)
- Fonte: `qwen_image_2.1_bf16.safetensors` da Comfy-Org (layout fundido `gate_up`, o mesmo de que a int8 da
  Comfy-Org e a mixed da NidAll partiram). Perfil `qwen_image21` novo nos conversores, derivado das 192 camadas que
  as duas publicações quantizam (dry-run: 192/192 em w4a8, w4a4 e int8).
- Builds: A `_w4a8` (uniforme, group 16, convrot 256, codebook); B `_int8_convrot` (quant_int8, CPU/eager ou CUDA);
  C `_w4a4_convrot` (uniforme); D `_mixed` (quant_mixed com calibração real do BF16: W4A4 onde o erro medido cabe,
  W4A8 onde não).
- Comparações casadas: A x mixed NidAll (mesmo formato, algoritmos de codebook diferentes); B x int8 Comfy-Org
  (mesmo formato; se o quantizador for o mesmo, bytes iguais — `compara_checkpoints_byte.py`); C e D x int4 SVDQ
  mesmertech (4 bits contra 4 bits).
- Erro por camada: `tools/erro_por_camada.py` (novo) sobre a calibração real (`calibrate_activations.py`,
  perfil `qwen_image21`, 6 prompts da bateria, seed 42, 25 passos), rel-RMSE de saída, kernels reais
  (comfy-kitchen / Nunchaku). Não aprova build; ordena perdas por camada e tipo.
- Render: mesma bateria (6 prompts × 2 seeds), métricas contra bf16, it/s, VRAM, e avaliação cega MiMo.
- Previsão: B ≈ int8 Comfy-Org (talvez idêntico); A ≈ mixed NidAll; C pior que int4 SVDQ (sem ramo low-rank);
  D entre C e A. Se C/D saírem com artefatos visíveis, fica registrado como negativo.

## 4. Avaliação cega MiMo V2.6 Pro (adendo antes de rodar)
- `avalia_mimo.py`: cópias com nome neutro (sha256 com sal), rubrica fixa 0–10 (aderência, texto, artefatos,
  anatomia, detalhe, geral) + ranking por prompt/seed com ordem embaralhada fixa.
- Controle de ruído do juiz: bf16 e bf16junto (se PSNR infinito) são a mesma imagem avaliada duas vezes às cegas;
  a diferença de nota entre elas é o ruído do juiz. Diferenças entre builds menores que isso não contam.

## 5. Shift a 2048² (adendo antes de rodar, 26/09 ~19:05)
- Issue ComfyUI #16447 / PR #16553 (aberto): o ComfyUI fixa mu = 0,69 (valor de 1024²) para qualquer resolução; o
  scheduler oficial usa mu = 0,5 + 0,4·(N−256)/7936, N = tokens do latente → 1,3129 a 2048².
- Teste sem mudar código: braço `padrao` (KSampler puro) × braço `mu131` (node nativo ModelSamplingFlux, base 0,5,
  max 0,6935 a 2048×2048 → mu 1,3127). DiT int8 Comfy-Org, 25 passos, 3 prompts (letreiro, retrato, paisagem) × seeds
  42 e 7, decode tiled. `shift_terminal` 0,02 não reproduzido.
- Avaliação: MiMo pareado cego (ordem embaralhada) + minha inspeção. Não há referência BF16 "certa" aqui: é
  preferência de qualidade entre dois schedules, não fidelidade. Previsão (do relato da issue): mu131 melhor a 2048².
