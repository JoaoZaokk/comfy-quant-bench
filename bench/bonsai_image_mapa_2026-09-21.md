# Bonsai Image: o mapa de promocao/democao, LIDO do artefato publicado (2026-09-21)

Tudo abaixo e **LEITURA** de arquivo publicado, nao execucao. Nada foi rodado na placa.

## Primeiro: nao e Flux Schnell, e FLUX.2-klein-4B

Seis repos de `prism-ml`, 18 a 21/05/2026, todos `diffusers:Flux2KleinPipeline`, apache-2.0:
`bonsai-image-{binary,ternary}-4B-{unpacked,gemlite-Nbit,mlx-Nbit}`. FLUX.1-schnell aparece so
como LINHA DE COMPARACAO na tabela deles (23,8 GB, GenEval 0,716) -- nao e a base.

## O mapa esta publicado inteiro, em 6 KB

`bonsai-image-ternary-4B-gemlite-2bit/transformer-gemlite-int2/quantization_config.json`:

    format gemlite-int2-ternary-g128   bits 2   group_size 128   solver ternary
    input/output dtype fp16            quantized_count 100   skipped_count 9   pack_seconds 150,48

**Quantizado (100 lineares, TODOS no mesmo tratamento ternario g128):**

    transformer_blocks.0..4   (5 double-stream)  x 12 = 60
      attn.to_q  to_k  to_v  add_q_proj  add_k_proj  add_v_proj  to_add_out  to_out.0
      ff.linear_in  ff.linear_out  ff_context.linear_in  ff_context.linear_out
    single_transformer_blocks.0..19 (20 single)  x  2 = 40
      attn.to_qkv_mlp_proj   attn.to_out

**Preservado em FP16 (9), por `skip_patterns` -- lista escrita a mao:**

    x_embedder   context_embedder   proj_out   norm_out.linear
    time_guidance_embed.timestep_embedder.linear_1 e .linear_2
    double_stream_modulation_img.linear   double_stream_modulation_txt.linear
    single_stream_modulation.linear

## Achado 1: a receita SOTA nao tem criterio por camada nenhum

`skip_patterns` e uma allowlist por NOME, e as 100 camadas escolhidas recebem **exatamente o mesmo**
tratamento (ternario, g128, fp16 scale). Nao existe erro medido por camada, nao existe promocao
seletiva, nao existe calibracao no config. **A forma e identica ao nosso `PROFILE_PATTERNS`** de
`tools/quant_w4a8.py:54`. Esta bancada construiu `quant_mixed.py` para escolher formato por camada
medindo em ativacao real; a referencia publica de 1,58 bit nao faz isso.

## Achado 2: eles PULAM a modulacao -- e esta bancada mediu que ela e a camada MAIS FACIL

Os tres `*_modulation.linear` estao na lista de pulados. O `CLAUDE.md` deste repo registra, medido:
`adaLN_modulation` e a camada de **menor** erro em W4A4 de todo o bloco -- **0,1263 contra 0,1569**
das demais -- e registra que o argumento "ninguem poe modulacao em 4 bits, logo nao se deve" era
consenso usado como evidencia.

Colisao direta, e as duas leituras possiveis sao distinguiveis por experimento:
  (a) a 1,58 bit a modulacao realmente quebra, e 4 bits nao diz nada sobre 1,58; ou
  (b) e o mesmo consenso herdado, e pular custa capacidade de graca.
**Nenhuma das duas foi medida.** O experimento que separa: quantizar a modulacao do Klein 4B nos
dois regimes e comparar contra o braco que a preserva, com o controle que tem de falhar.

## Achado 3: dois fingerprints de TREINO, nao de PTQ

1. `manifest.json` do unpacked: `"model_version": "ternary g128 (bf16 master)"`. **"bf16 master"** e
   o vocabulario de peso latente de QAT (BitNet-like), nao de PTQ.
2. O `transformer/config.json` do bonsai tem **3 chaves que o original nao tem** (620 contra 541 B):
   `enable_time_sign_embed`, **`musubi_block_swap_device: "cpu"`**, **`musubi_blocks_to_swap: 0`**.
   `musubi-tuner` e ferramenta de FINE-TUNING com block-swap para VRAM curta. Config de treino
   vazada no artefato de release.

Isso e leitura de metadado, **nao prova**. O whitepaper deles nao descreve o metodo (medido antes:
outline sem secao de metodo; `QAT` 0, `PTQ` 0, `we train` 0 em texto extraido por pypdf 6.19.0,
25 paginas), entao o artefato e a unica fonte.

**O teste decisivo, e ele e barato porque os tres arquivos sao alinhaveis:**

    original  black-forest-labs/FLUX.2-klein-4B   transformer/...safetensors  7.751.109.744 B
    ternario  bonsai-image-ternary-4B-unpacked    idem                        7.751.109.712 B
    binario   bonsai-image-binary-4B-unpacked     idem                        7.751.109.712 B

**32 bytes de diferenca** -- mesma estrutura, mesmos shapes, so os valores mudaram. Por camada,
sobre as 100 quantizadas:
  - concordancia de SINAL com o original. ~100% -> PTQ (arredondaram o original).
    ~50% -> TREINADO, e ai nenhuma regra de promocao explica o resultado.
  - o peso unpacked e igual a `nearest_ternary(original * escala_do_grupo_de_128)`? residuo ~0 -> PTQ.
  - numero de valores distintos por grupo de 128: ternario tem de dar 3, binario 2.
  - e o controle: as 9 pulados tem de ser **byte a byte identicas** ao original. Se nao forem, elas
    tambem foram treinadas, e a leitura "PTQ com allowlist" morre na hora.

sha256 do ternario esta publicado no manifest deles -- `fa5fd182...cf4e35964` -- entao o download se
confere por prova positiva, nao por tamanho. Baixando para `F:\bonsai-re\`.

## Achado 4: o numero de 1,21 GB e idealizado; o arquivo tem 1,43 GiB

README deles: "1.21 GB model-level Bonsai representation; 1.54 GB CUDA packed deployment". O arquivo
real `state_dict.pt` tem **1.540.457.482 B = 1,43 GiB**. O 1,21 vem da conta
`log2(3) + 16/128 = 1,71 bit/peso`, nao de um arquivo. Citar o arquivo.

## Achado 5: a qualidade CAI, e o README chama de "muito perto"

Tabela deles, H100, GenEval no protocolo oficial 512x512:

    modelo                    transformer GB   GenEval   HPSv3   DPG-Bench
    FLUX.2 Klein 4B (base)          7,75        0,819    12,84     0,853
    Bonsai Ternary 4B               1,21        0,723    12,22     0,851
    Bonsai Binary 4B                0,93        0,671    11,15     0,822
    FLUX.1-schnell                 23,8         0,716    12,67     0,848

GenEval cai **11,7%** do base para o ternario (0,819 -> 0,723) e 18,1% para o binario. DPG quase nao
se move (0,853 -> 0,851). Ou seja: **as tres metricas discordam sobre o tamanho do dano**, que e o
mesmo padrao que esta bancada ja mediu com divergencia de latente contra imagem. Numeros deles, nao
medidos aqui, sem replicacao independente conferida.

## Nao coberto

Nada executado. Os pesos nao foram comparados ainda (download em curso). O pack gemlite nao foi
aberto, entao **nao se sabe se gemlite INT2 roda matematica de 2 bits de verdade ou desempacota para
fp16** -- a mesma pergunta que aqui se responde com contagem de dispatch, e que decide se o ganho
deles e memoria ou tempo. As metricas deles nao foram replicadas. O MLX nao foi olhado. E a hipotese
treino-vs-PTQ segue sendo leitura de metadado.
