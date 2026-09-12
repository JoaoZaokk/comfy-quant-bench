# Handoff — 2026-09-12, meio da varredura por famílias

Escrito a pedido do dono, antes de compactar o contexto. Tudo abaixo foi **executado nesta
máquina**. Onde eu só li código, a linha diz isso.

---

## 0. A REGRA DE TRABALHO, que o dono fixou e que substitui qualquer ordenação minha

**Imagem primeiro, vídeo depois. Uma família por vez, e a família só fecha quando sobe.**

```
para cada FAMÍLIA:
    1. converter TUDO que a família usa -- difusão + text encoder + o que mais houver
       (VAE NAO: o dono disse explicitamente que não se quantiza)
    2. medir: erro por camada, ladder de qualidade com braço de referência, despacho
    3. subir ao HuggingFace com TODAS as provas: números, imagens, o que não foi coberto,
       e os CRÉDITOS da fonte no card
    4. commitar no git a documentação, com os LINKS dos repos do HF
    5. só então ir para a próxima família
```

Ordem das famílias, na ordem dada pelo dono:

```
IMAGEM   Krea2  ->  Z-Image  ->  Qwen Image (edit e não-edit, repos separados)  -> [as demais no disco]
VÍDEO    LTX  ->  Wan  ->  [os demais que a máquina aguenta]
```

**"Edit e não-edit" são repositórios separados no HF**, não um repo com dois arquivos.

### O obstáculo do passo 4, que precisa da decisão dele

**O repo raiz NÃO tem remoto.** `git remote -v` volta vazio. Os cards já publicados apontam para
`https://github.com/JoaoZaokk/comfy-quant-bench`, mas esse remoto não está configurado aqui, então
**nenhum push sai desta máquina** sem alguém adicioná-lo. E não é só rodar `git remote add`: este
diretório é uma instalação viva do ComfyUI com ~758 GiB de modelos, e o `.gitignore` é uma
**allowlist** — publicar exige conferir o que a allowlist deixa entrar. É decisão do dono.

---

## 1. O que está NO AR agora

| família | repo HuggingFace | conteúdo |
|---|---|---|
| Z-Image | [Z-Image-Turbo-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Z-Image-Turbo-W4A4-ConvRot) | w4a4 + misto, oficial Tongyi-MAI |
| Z-Image | [Z-Image-De-Turbo-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Z-Image-De-Turbo-W4A4-ConvRot) | w4a4 + misto, ostris |
| Z-Image | [Beyond-Reality-Z-Image-v2-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Beyond-Reality-Z-Image-v2-W4A4-ConvRot) | anterior a esta sessão |
| encoders | Qwen3-4B-W4A4, Qwen2.5-VL-7B-W4A4, Gemma-3-12B-W4A8 | anteriores |
| outros | Wan2.1-VACE-1.3B, HunyuanVideo-1.5-720p | anteriores |

Os 8 arquivos do Z-Image subidos nesta sessão foram **conferidos byte a byte contra o Hub depois
do upload** — upload que termina sem erro não prova que o arquivo do outro lado é o do disco.

---

## 2. Estado por família

### Krea 2 — medido e documentado, NÃO publicado

| arquivo | formato | GiB | erro/camada |
|---|---|---|---|
| `krea2_turbo_bf16` | fonte | 24,48 | — |
| `krea2_turbo_int8_convrot` | Comfy-Org | 12,57 | — |
| `krea2_turbo_w4a4` (nosso) | 224 × w4a4 | **7,50** | **0,1199** |
| `krea2_turbo_mixed` (nosso) | 173+51 | 7,70 | — |
| encoder `qwen3vl_4b_w4a8` (nosso) | 252 × w4a8 | **3,41** | — |
| encoder `qwen3vl_4b_w4a4_convrot` (nosso) | 252 × w4a4 | 3,19 | — |

Cadeia: **33,0 → 11,2 GiB (2,95x)**. Qualidade: 40 renderizações, nenhuma quebrada; o int8 do
Comfy-Org vence o nosso W4A4 em **10 de 10** corridas pareadas, e o nosso é 1,47x mais rápido e
1,68x menor. **Teto procurado e não alcançado**: o único eixo do formato topa em 0,1377 e o
modelo não quebra lá.

Encoder: **W4A8 é a escolha** — 2,53x mais fiel que W4A4 a 76 tokens (0,1438 contra 0,3637) por
0,22 GiB a mais, e soltar o cadeado não custa nada nele enquanto rende 2,09x.

Edição: desvio do cachorro **fechado** — reproduz na semente 1002 só no W4A4 (1 de 9 contra 0 de 9
nos outros dois braços) e some com numeral explícito na instrução.

**Por que não subiu:** Krea 2 Turbo é **gated** com licença própria (`krea-2-community-license`).
O Comfy-Org redistribui derivados publicamente sob `license: other`, o que é prática aceita e
**não é uma licença que eu tenha lido**. Card pronto em `bench/hf/krea2-turbo-quant/README.md`.
**Precisa do sim do dono.**

### Z-Image — FECHADA

Os dois oficiais nunca tinham sido convertidos; agora estão, medidos (72 renderizações, nenhuma
quebrada) e publicados. **11,46 → 3,06 GiB e 0,903 → 0,341 s/passo.**

Achado que viaja: **a receita mista não transfere entre um modelo e o fine-tune dele.** No Turbo
oficial o misto vence 11/12 pareadas; no De-Turbo vence 7/12 e a ferramenta imprime "split
decisions". O erro por camada não previa (0,1228 contra 0,1211, 1,4% de diferença).

### Qwen Image Edit 2511 — convertido e medido por camada, SEM FOTO AINDA

    qwen_image_edit_2511_bf16          40.861.031.560 B   fonte, conferida contra o Hub
    qwen_image_edit_2511_int8_convrot  20.499.083.824 B   Comfy-Org, conferida
    qwen_image_edit_2511_w4a4          10.306.850.912 B   NOSSO, 840/840
    qwen_image_edit_2511_mixed         10.603.610.352 B   NOSSO, 607+233

**38,05 → 9,60 GiB (3,96x)**, e 1,99x menor que o int8 público. `err_w4a4` mediana **0,1080**, o
menor de toda a bancada num modelo de **20.430.401.088 parâmetros**.

**O QUE ESTÁ RODANDO NESTE MOMENTO:** `.scratch/ladder_qwen.ps1`, 4 braços × 6 prompts × 2
sementes × 20 passos a cfg 2,5 = 48 renderizações. Estava no braço BF16 (38,1 GiB descarregando),
working set 43,4 GiB, disco a 62 MB/s, **0 latentes escritos**. Log em `.scratch/ladder_qwen.log`.
Se morreu, o comando é o próprio `.ps1`.

**Falta:** ladder terminar, despacho, encoder `qwen_2.5_vl_7b` remedido com o cadeado solto (o
build publicado é ANTERIOR ao destravamento, então os números dele descrevem o caminho
dequantizado), cards, upload em **dois repos** (edit e não-edit), e o **Qwen-Image 2512 base**,
que ainda não foi baixado.

---

## 3. A mudança no ComfyUI que esta sessão fez, e que se perde num update

**O cadeado do text encoder saiu.** Até hoje um encoder quantizado guardava peso de 4/8 bits e
fazia a conta dequantizada — economizava VRAM, não economizava tempo, kernel nunca alcançado.
Duas travas independentes, qualquer uma sozinha bastando:

    comfy/sd.py       set_model_compute_dtype(torch.float32) -> comfy_force_cast_weights
    comfy/sd1_clip.py mixed_precision_ops(..., full_precision_mm=True)

Agora as duas saem juntas quando o checkpoint tem camadas quantizadas de verdade, e
`--disable-quantized-text-encoder` repõe o antigo. Medido no Gemma W4A8 contra o BF16 original:
**1,11x menos fiel por 3,77x mais rápido**, e 21,6x mais rápido que o BF16 a 3,13x menor.
Controle passou (encoder não-quantizado intocado) e a suíte do ComfyUI dá **45 falhas / 1186
passam / 58 erros idêntico com e sem a mudança**.

**`ComfyUI/` não é rastreado pelo repo raiz e `update/update_comfyui.bat` apaga isso em silêncio.**
O diff está versionado em `patches/comfyui_text_encoder_quantized_math.patch`. Se o ComfyUI for
atualizado, reaplicar.

---

## 4. Inventário do disco, que é o mapa das próximas famílias

62 arquivos `.safetensors` de difusão, **757,8 GiB**. `D:` é um SMB montado — 408 GiB chegam pela
rede e "D: offline" é estado normal, não defeito.

    familia          tipo    arq    GiB
    ltx              VIDEO    11   222,5
    qwen_image       imagem    5    89,8
    wan              VIDEO     7    86,1
    qwen-image       imagem    6    71,6
    krea2            imagem    5    64,8
    minimax          VIDEO     4    52,2
    zimage           imagem    9    45,0
    flux             imagem    5    34,6
    z_image          imagem    2    22,9
    capybara         VIDEO     1    15,5
    hunyuanvideo     VIDEO     1    15,5
    beyond-reality   imagem    1    11,5

**Famílias de imagem ainda intocadas: Flux** (5 arquivos, 34,6 GiB, inclui `flux-2-klein-base-4b-fp8`
que esta bancada já mediu rodando **dequantizado**) e o que estiver em `outro`.

**LTX está inteiro no disco, nada a baixar:** transformer destilado e dev BF16 (42,0 GB cada),
int8-convrot do Comfy-Org, nosso w4a8 (12,52 GB), nvfp4 de terceiros, o w4a4 da riftcast (1440
camadas, 4 bits de verdade), encoder `gemma4-12b-with-proj` BF16 (26,26 GB) e int8, VAE de vídeo
(1,47 GB) e de áudio (365 MB). Workflow de referência pronto: `LTX25-int8-acceptance-v2.json`,
cadeia A/V completa, hoje em 512×512 × 49 quadros.

**O vídeo de 10 s pedido é o risco da etapa, não a conversão.** LTX quer 8n+1 quadros; a 25 fps,
10 s = **249 quadros**, 5,08x o latente do workflow atual.

---

## 5. Armadilhas que vão morder de novo

- **LTX 2.5 exige `--disable-dynamic-vram`.** Sem a flag morre com
  `aimdo: hostbuf_read_file_slice: device copy failed` depois de tentar montar 35 GB num cartão de
  24, e a mensagem **não nomeia o subsistema culpado** — parece "o modelo é grande demais".
- **NÃO tomar `Assert-GpuLock` antes de ferramenta que entra no `BenchGuard` sozinha**
  (`quality_ladder`, tudo que passa por `_timing.compare()`). Ela recusa a própria corrida.
  Tomar o lock à mão só para conversor e sonda.
- **`CUDA_VISIBLE_DEVICES=0` nos benchmarks**: o cortex ocupa ~2,3 GiB da 3080 Ti e o teto do
  guarda é 2,00 GiB. Esconder a placa faz a recusa olhar só a que está em uso.
- **Matar ComfyUI pela ÁRVORE**, nunca pelo pid: `main.py --windows-standalone-build` se
  re-executa como filho e o órfão segura 20 GiB invisivelmente.
- **`TaskStop` não mata o filho.** Conferir com `Get-CimInstance Win32_Process` depois.
- **Uma corrida só não é medição.** O primeiro encode de qualquer braço é 2,8–3,5x a mediana.
  Eu publiquei "soltar o cadeado deixa o W4A4 mais lento" a partir de uma corrida e a afirmação
  **inverteu** com cinco repetições.
- **Varrer ao redor do ponto não é medir o ponto.** 48 renderizações disseram "o desvio do
  cachorro não reproduz" — nenhuma delas usava a semente onde ele acontece.
- **O decode do `quality_ladder` morre** com `hostbuf_allocate` sobre `lib` None, sempre. Os
  latentes ficam salvos; decodificar depois com
  `tools/decode_latents.py <dir>/latents --vae <vae>`. Os números do ladder não dependem do decode.

---

## 6. Defeitos de ferramenta consertados nesta sessão

- `calibrate_activations.py` lia um arquivo de prompts inteiro como **um** prompt — seis prompts
  viravam um prompt de seis linhas. **O mesmo defeito havia sido corrigido no `quality_ladder.py`
  horas antes**; este era o irmão esquecido. Não falhava: calibrava errado.
- `probe_te_cadeado.py` cronometrava a referência **uma** vez e os braços N vezes. O mesmo arquivo
  mediu 282,2 ms e 434,1 ms em chamadas consecutivas.
- O mesmo probe imprimiu **cosseno 1,002611** — fora do domínio da métrica, erro de acumulação em
  float32 sobre 192 M elementos. Agora compara em float64.
- `probe_quant_dispatch.py` ganhou `--te-travado`, o braço de controle sem o qual o A/B do cadeado
  compararia dois braços destravados.
- Perfis novos: `qwen3vl` (encoder do Krea2, decoder um segmento mais fundo) e `qwen_image`
  (840/840 derivado do checkpoint do Comfy-Org, conferido também contra `named_modules()`).

---

## 7. Estado da máquina, medido agora

    F:            58 GB livres de 1,9 TB   <- APERTADO. Qwen 2512 + LTX vão pedir espaço.
    3090          ladder do Qwen ocupando
    3080 Ti       ~2,0 GiB do cortex, NAO TOCAR
    GPU_BENCH.lock  tomado por quality_ladder pid 73460
    git           master, 15 commits novos nesta sessão, SEM REMOTO
    ComfyUI       4 arquivos modificados (o cadeado), patch em patches/

Liberar espaço é seguro apagando **saídas nossas** já medidas e documentadas (nunca um original):
os builds do teto do Krea2 já foram apagados assim — sidecar e análise ficam, o arquivo regenera
em 85 s.

---

## 8. O que eu faria a seguir, na ordem

1. **Esperar o ladder do Qwen**, decodificar os latentes, montar a folha de contato, olhar.
2. **Despacho** nos dois builds do Qwen; **remedir o encoder** `qwen_2.5_vl_7b` com o cadeado
   solto, nos dois formatos, contra o BF16.
3. **Fechar a família Qwen Image Edit**: card em inglês com créditos ao Comfy-Org e ao Qwen,
   upload, doc no git com o link.
4. **Qwen Image 2512 base** — baixar, mesma cadeia, repo separado.
5. **LTX**, com o vídeo de 10 s no original antes de qualquer comparação.
6. **Wan**.

E em cada uma: subir só depois de ter foto, despacho contado e o que-não-foi-coberto escrito.
