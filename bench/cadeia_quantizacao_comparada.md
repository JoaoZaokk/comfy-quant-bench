# A cadeia de quantização comparada — menor tamanho, melhor qualidade, melhor velocidade

Síntese de 2026-09-13. Toda linha aqui foi **medida nesta máquina**, com o braço original
renderizado no mesmo prompt, mesma semente e mesmos passos. Nenhum número vem de README de
terceiro.

O pedido era achar **o melhor lado de cada um**. A resposta não é um formato: é um eixo.

---

## 1. O eixo que decide: a ATIVAÇÃO, não o peso

`convrot_w4a4` e `asym_w4a8_int8` carregam **os mesmos pesos de 4 bits**. A única diferença é o
caminho da ativação — 4 bits contra 8. Medido em três famílias, com o original ao lado:

| modelo | W4A4 | divergência | W4A8 | divergência |
|---|---|---|---|---|
| Qwen-Image-Edit 2511 | **estática pura** | 1,7440 | bom | 0,4997 |
| Qwen-Image 2512 | **granulado colorido** | 1,3369 | bom | 0,3875 |
| Wan 2.2 TI2V 5B | **borrão** | 0,3847 | bom | 0,2115 |

**Três famílias, três falhas que não se parecem, uma causa.** Peso sobrevive a 4 bits em todas;
ativação não sobrevive em nenhuma.

Isto contraria o que esta bancada acreditava até setembro: o W4A4 **funciona** no Z-Image
(0,1254), no Krea2 (0,1199) e no LTX 2.5 do riftcast. Então não é "W4A4 nunca presta" — é que os
modelos onde ele presta são os de 2025, e as três famílias de 2026 medidas aqui quebram.

## 2. O melhor lado de cada formato, medido

| build | fonte | GiB | razão | s/passo | vs original | veredito |
|---|---|---|---|---|---|---|
| Qwen-Image-Edit 2511 W4A8 | 38,05 | **10,79** | 3,53x | 1,575 | — | bom |
| Qwen-Image 2512 W4A8 | 38,05 | **10,79** | 3,53x | 1,566 | **3,65x mais rápido** | bom |
| LTX 2.5 22B W4A8 | 39,13 | **11,66** | 3,36x | — | **1,95x mais rápido** | bom |
| Wan 2.2 TI2V W4A8 | 9,31 | **2,75** | 3,39x | 0,994 | **1,59x mais rápido** | bom |

**A razão de compressão do W4A8 é notavelmente estável: 3,36x a 3,53x** em quatro modelos de
1,3 B a 22 B, em três arquiteturas diferentes. Não é 4x — o peso vira 4 bits, mas escalas,
zero-points e tudo que não é `Linear` ficam.

## 3. Contra a quantização oficial, quando ela existe

Onde o autor do modelo publica a própria quantização, ela é o adversário certo:

| família | oficial | nosso | menor por | mais rápido por | mais fiel |
|---|---|---|---|---|---|
| Qwen-Image-Edit | int8 Comfy-Org 19,09 GiB | W4A8 10,79 | **1,77x** | 1,12x mais lento | **deles** |
| LTX 2.5 | int8 Lightricks 20,03 GiB | W4A8 11,66 | **1,72x** | **1,20x** | **deles** (MAE 4,10 vs 7,81) |

**A troca é explícita e sempre na mesma direção:** o nosso é quase o dobro de menor, e o deles é
mais fiel. Em cinco famílias seguidas nesta bancada o braço INT8 ganhou em fidelidade — e no LTX
esse braço era da **própria Lightricks**.

Quem tem VRAM sobrando deve usar o oficial. Quem não tem, ganha 8,4 GiB de volta e paga em
fidelidade — e a folha de contato de cada card mostra quanto.

## 4. As métricas, e para que lado cada uma erra

Esta é a parte que mais custou a aprender, e é a mais útil.

**Erro por camada ordena formatos, não localiza penhascos.** Ele acertou que o W4A8 é 3,02x melhor
que o W4A4 no Qwen Edit. Não avisou que 4 bits de ativação caem de um penhasco em vez de degradar.
E `0,1080` — o Qwen Edit em W4A4 — é o **menor erro por camada que já produziu lixo aqui**, abaixo
do Krea2 (0,1199) e do Z-Image (0,1254), que funcionam.

**Divergência de latente é ENVIESADA para falha macia.** O build **borrado** do Wan mede **0,3847**;
o build **perfeito** do Qwen base mede **0,3875**. O borrado pontua melhor. O mecanismo: borrão é
uma mudança *pequena* de latente — suavizar afasta menos da referência do que ir para algo nítido
e diferente. Um build que degrada suavemente sempre se elogia nessa métrica.

**Nenhuma das duas aceita um build sozinha.** Só a renderização decide, e por isso toda folha de
contato sobe junto com o peso.

## 5. O que NÃO está coberto

- **Um cartão (RTX 3090, sm86), um sampler, um scheduler.** As comparações de velocidade valem
  para esta máquina.
- **Nenhuma métrica perceptual.** "Indistinguível" é veredito humano numa folha de contato.
- **`convrot_groupsize` 256 em tudo.** O W4A4 aceita 16/64/1024 e o W4A8 só 256 — então o eixo do
  groupsize não pôde ser varrido nos dois lados.
- **Nenhum build misto nas três famílias novas.** A escolha foi o eixo limpo — tudo-W4A4 contra
  tudo-W4A8 — e não uma busca pelo ótimo entre eles. Um misto com poucas camadas em A4 pode
  existir e não foi procurado.
- **O braço BF16 da EDIÇÃO do Qwen não roda nesta máquina** (quatro falhas em
  `bench/qwen_edit_bf16_inalcancavel.md`), então a medição de edição compara contra o int8 do
  Comfy-Org, **não contra o original**.
- **SDXL e família UNet estão fora por arquitetura**, não por falta de trabalho — ver
  `bench/inventario_imagem_restante.md`.
