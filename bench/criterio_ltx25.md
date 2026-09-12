# Critério: LTX 2.5 22B distilled, ConvRot W4A4

**Escrito em 2026-09-12, ANTES de baixar a fonte e antes de qualquer conversão.** Nenhuma
previsão aqui pode ser ajustada depois de ver o resultado; uma que errar fica registrada como
errada, do jeito que as duas do Krea2 e a do Z-Image ficaram.

---

## O que existe, medido na API do HF (fonte primária, não post de blog)

`Lightricks/LTX-2.5`, consultado 2026-09-12. O token do dono **já tem acesso ao gate** —
conferido por `get_hf_file_metadata`, que é HEAD e não aceita termo nenhum.

| arquivo | bytes | GiB |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-bf16` | 42.018.190.584 | 39,13 |
| `ltx-2.5-22b-dev-transformer-bf16` | 42.018.190.584 | 39,13 |
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | 21.504.034.224 | 20,03 |
| `ltx-2.5-22b-distilled-transformer-nvfp4` | 18.721.548.408 | 17,44 |
| `gemma4-12b-with-proj-ltx-2.5-bf16` | 26.263.858.182 | 24,46 |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot` | 15.372.969.374 | 14,32 |
| `ltx-2.5-video-vae-bf16` | 1.472.223.346 | 1,37 |
| `ltx-2.5-audio-vae-bf16` | 364.866.540 | 0,34 |

**A própria Lightricks publica um `comfy-int8-convrot`.** Isso é novo nesta bancada: nas quatro
famílias anteriores o braço INT8 que venceu o nosso W4A4 era de terceiro (Comfy-Org, Winnougan).
Aqui é do autor do modelo, o que torna a comparação mais dura e mais honesta.

## O que já está no disco, e o buraco

Nenhum DiT de LTX **sem quantização** existe nesta máquina — confirmado por dois caminhos
(`find -iname "*ltx*"` e listagem por tamanho de `diffusion_models`, `unet`, `checkpoints`,
`diffusion_models_gguf`). O maior LTX local é `ltx-2.3-22b-dev-fp8` (27,15 GiB), que já é fp8.
Do 2.5 só há as três builds quantizadas do riftcast e o encoder `gemma4-12b-ltx25-comfy-w4a8`.

**A conta do disco, com 58 GB livres:**

```
vídeo "todo em original" = transformer BF16 39,13 + encoder BF16 24,46 + VAE 1,37 = 64,96 GiB
                                                                          NAO CABE
com o encoder w4a8 que já está aqui         = 39,13 +  9,88 (local) + 1,37 = 50,38 GiB
                                                                          cabe, sobra ~7 GB
```

O segundo arranjo cabe mas **não é "todo em original"**, que é o que o goal pede. A escolha entre
liberar espaço e aceitar um braço misto é do dono; este arquivo registra a aritmética para que a
decisão não seja tomada por acidente.

## Previsões

Escritas antes de qualquer medição, cada uma com o que a refuta.

**P1 — `err_w4a4` mediano cai entre 0,10 e 0,20.**
Apoio fraco, e digo por quê: os dois únicos modelos acima de 13 B nesta bancada mediram 0,1080
(Qwen Edit, 20,4 B) e 0,1837 (Hunyuan, ~13 B). **A monotonia por tamanho já morreu duas vezes
aqui** (Krea2 e Qwen Edit), então isto é um intervalo de referência, não um mecanismo.
*Refuta:* mediana fora de [0,10; 0,20].

**P2 — o `comfy-int8-convrot` da Lightricks sai mais fiel que o nosso W4A4, em divergência de
latente pareada.** Seria a quinta família seguida na mesma direção, agora contra o autor do
modelo.
*Refuta:* nosso W4A4 vence em 50% ou mais das corridas pareadas.

**P3 — o dano da quantização aparece primeiro como incoerência TEMPORAL, não como perda de
qualidade por quadro.** Um quadro isolado do braço quantizado será difícil de distinguir do
original; o vídeo em movimento não será.
*Refuta:* quadros individualmente piores com o movimento estável — isto é, o erro se distribuindo
igual no tempo.

**P4 — 249 quadros (10 s a 25 fps, 8n+1) NÃO cabem em 24 GiB na resolução nativa**, mesmo com o
transformer quantizado, e a corrida vai precisar de offload, tiling temporal, ou as duas placas.
*Refuta:* a geração completa roda residente na 3090 sem offload.

**P5 — o braço BF16 do vídeo de 10 s leva mais de 20 minutos na 3090** com o modelo descarregado
inteiro (39,13 GiB contra 24 GiB de VRAM).
*Refuta:* menos de 20 minutos de relógio.

**P6 — a máquina cai ou reinicia pelo menos uma vez durante a suíte do LTX.**
Não é piada: cinco boots em dois dias, e uma queda hoje às 19:40:10 (`Kernel-Power 41`, sem
bugcheck, sem WHEA) que apagou uma corrida inteira de 48 renderizações. É por isso que o
`quality_ladder.py` passou a gravar cada latente na hora.
*Refuta:* a suíte inteira roda sem um reinício.

## O que este critério NÃO cobre

- Não diz qual `convrot_groupsize` usar. W4A4 aceita 16/64/256/1024; nada foi medido em LTX.
- Não cobre o **áudio**. O LTX 2.5 gera áudio junto e há um `audio-vae` separado; VAE não se
  quantiza, mas o caminho de áudio não foi analisado e pode ter Linears próprias.
- Não cobre o `nvfp4` (17,44 GiB), que é um quinto formato e precisaria de hardware Blackwell
  para executar nativo — nesta máquina rodaria emulado, o que mediria outra coisa.
- Não define o prompt do vídeo de 10 s. Ele tem de ser escrito antes de gerar e ficar fixo nos
  braços, ou a comparação não é pareada.
- Nada aqui foi executado. É um plano com previsões, e está marcado como tal.
