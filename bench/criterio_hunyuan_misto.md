# Critério, escrito antes de rodar — 2026-08-31

## A teoria em teste

O erro **absoluto** por camada decide se o modelo quebra, independente de qual modelo é. A linha de
uso fica entre `0,124` (Z-Image W4A4, imagem boa) e `0,214` (HunyuanVideo W4A4, imagem destruída).

O custo de descer a ativação para 4 bits é o mesmo nos dois (`a4/a8` = 3,17 contra 3,05). O que
separa é a base: `err_w4a8` mediano `0,0394` contra `0,0695`, ou seja o **peso**.

## O experimento

`quant_mixed.py --promote-error T` promove para 8 bits toda camada com `err_w4a4 > T`. Contagens já
conhecidas da análise de 2026-08-19:

```
T = 0,40   ->  31/432 promovidas  ( 7,2%)
T = 0,25   -> 140/432 promovidas  (32,4%)
T = 0,15   -> 408/432 promovidas  (94,4%)   <- padrão da ferramenta
```

Três builds, e a mesma maçã de agosto: 480x480, um quadro, seed 12345, 6 passos, cfg 6,
euler/simple.

## O que cada desfecho significa — decidido agora, não depois

| se a maçã voltar em | então |
|---|---|
| **0,40** (só 7% promovidas) | A teoria da MEDIANA está errada. Poucas camadas catastróficas dominam, e o que importa é a cauda, não o nível geral. Isso **contradiz** o que escrevi hoje. |
| **0,25** (32%) | A linha de uso fica entre a mediana resultante desse build e `0,214`. Teoria sobrevive e ganha um número. |
| **só em 0,15** (94%) | A linha fica abaixo de `0,15`. "Hunyuan misto" é praticamente "Hunyuan W4A8" — nenhum ganho prático sobre o que agosto já sabia, mas a teoria sobrevive. |
| **em nenhum** | A teoria está **morta**. W4A8 no Hunyuan já é sabidamente bom (foto de agosto), então um build de 94% em 8 bits que ainda quebra significa que o erro por camada não é o eixo. |

## O que este experimento NÃO responde

Por que o peso do Hunyuan é 1,76x pior que o do Z-Image. Mede que é, não por quê.

Uma semente, um prompt, 480x480, um quadro. Sem métrica perceptual — o julgamento é olho humano
sobre "quebrou ou não quebrou", que é o único uso para o qual esta bancada já mediu que a imagem
final serve.

E as duas calibrações que originam os números têm passos, resolução e execuções diferentes
(8/1024/4 contra 6/512/2). A razão `a4/a8` é interna a cada modelo e sobrevive a isso; as medianas
absolutas comparadas entre modelos, menos.
