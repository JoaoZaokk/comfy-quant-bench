# LoRA sobre peso quantizado: critério escrito antes de renderizar — 2026-09-13

O dono pediu: *"verifica se os LoRAs estão funcionando do jeito que deveria funcionar"*. A única
medição anterior desta bancada (2026-08-19) respondia outra pergunta — se o kernel nativo
continuava sendo chamado com LoRA aplicado (680/680, sim). **Nunca foi medido se o LoRA chega
inteiro ao peso**, nem o que custa.

## O mecanismo, lido no código antes de medir

`LoraLoaderModelOnly` sobre um checkpoint quantizado **não mantém o LoRA como ramo separado**.
`ModelPatcher.patch_weight_to_device` (`comfy/model_patcher.py:899`) faz, por chave com patch:

```
convert_weight   ->  W.dequantize()                                    comfy/ops.py:1449
calculate_weight ->  W + delta, em lora_compute_dtype (fp16 nesta placa)
set_weight       ->  W.requantize_from_float(W', scale="recalculate",
                                             stochastic_rounding=seed)   comfy/ops.py:1455-1457
```

Dequantiza, soma, **requantiza para os mesmos 4 bits**, com escalas recalculadas e arredondamento
estocástico. O peso volta a ser `QuantizedTensor` do mesmo layout — por isso o kernel nativo
continua — mas o delta de um LoRA é tipicamente **menor que o passo da grade de 4 bits**, então
ele só pode sobreviver *em média* (arredondamento estocástico não tem viés) e cada camada tocada
ganha ruído novo.

O outro caminho, `LoraLoaderBypassModelOnly` (`comfy_extras/nodes_lora_debug.py`, rotulado "for
debugging"), mantém o delta como ramo BF16 de baixo posto no forward e **não toca o peso**. É o
braço de comparação.

## Camada 1 — no peso, pelo caminho real (`tools/probe_lora_requant.py`)

Previsões escritas no docstring da ferramenta antes de rodar (commit `d113a3a`):

- **P1** sobrevivência mediana (projeção de `d_eff` sobre `delta`) entre 0,9 e 1,1. Refuta: < 0,8 ou > 1,2.
- **P2** `err_depois / err_antes ≈ √2 = 1,41` — ruído novo independente e do mesmo tamanho do
  velho. Refuta: < 1,1 ou > 2,0.
- **P3** `ruido_extra > 1` na maioria das camadas — o ruído acrescentado é maior que o LoRA. Refuta: mediana < 0,5.

Um controle foi acrescentado depois da primeira rodada, e antes de qualquer conclusão: **requantizar
W0 sem delta nenhum** pelo mesmo `set_weight`. Se isso já custa erro, a requantização não é
idempotente e parte do "custo do LoRA" é custo de *qualquer* patch — inclusive de um que falhou.

### Medido

| modelo, formato | LoRA | chaves | magnitude `|δ|/|W|` | sobrevivência | cosseno | ruído/LoRA | erro de peso: antes → só requant → com LoRA | razão |
|---|---|---|---|---|---|---|---|---|
| Z-Image v2, W4A4 cg256 | RealisticSnapshot Turbo v5 (r32) | 480 → 240 alvos, 180 pesos, 0 sem mapa | 0,095 | **1,000** | 0,51 | 1,7x | 0,157 → 0,163 → **0,238** | 1,45x |
| Krea2 Turbo, W4A4 cg256 | krea2 turbo LoRA rank 64 | 535 → 271, 0 sem mapa | 0,0086 | **1,000** | 0,14 | 7,2x | 0,162 → 0,168 → **0,176** | 1,07x |
| Wan 2.2 TI2V 5B, W4A8 | `stock_photography_wan22_LOW` (LoRA de **14B**) | 300 casam pelo NOME, 300 chaves sem mapa | **0** (todo `calculate_weight` falha) | — | — | — | 0,0731 → **0,0835** → 0,0835 | 1,14x sem LoRA nenhum |

Erro de peso = `‖W − W_bf16‖ / ‖W_bf16‖` por camada, mediana da amostra (12, 21 e 24 camadas).
**É erro de peso, não o `err_w4a4` de ativação da tabela de bandas do CLAUDE.md** — as duas
colunas não se comparam.

- **P1 confirmada nas duas famílias onde há delta.** O LoRA está lá, em média, com três casas.
- **P2 confirmada no Z-Image (1,45x) e REFUTADA no Krea2 (1,07x).** O modelo de "ruído
  independente do mesmo tamanho" estava errado: o ruído extra **cresce com a magnitude do
  próprio LoRA** (arredondamento estocástico move cada entrada com probabilidade ∝ |δ|/passo), então
  um LoRA pequeno custa pouco em absoluto e muito em relativo (7,2x o próprio delta), e um LoRA
  grande custa muito em absoluto (0,157 → 0,238) e pouco em relativo (1,7x).
- **P3 confirmada nas duas.** O que a requantização acrescenta é sempre maior que o LoRA em si.
  O cosseno de 0,51 e 0,14 diz o mesmo: o que mudou no peso é o delta **mais** ruído de 2 a 7
  vezes o tamanho dele.
- **O controle de delta zero pegou uma armadilha própria de modelos quantizados.** Um LoRA de
  outra arquitetura (14B sobre 5B) casa pelo nome, falha na forma dentro de `calculate_weight`, o
  ComfyUI **loga `ERROR lora ... shape` e segue** — e o peso é requantizado assim mesmo.
  Resultado: **nada do LoRA é aplicado e o modelo piora 14%** em erro de peso. Num BF16 a mesma
  falha reescreve o peso igual e é inofensiva. A única evidência é uma linha de log.

## Camada 2 — no que sai (previsões ANTES de renderizar)

O peso diz que o LoRA está lá com ruído. Se o ruído importa, só a saída diz. Dois desenhos, cada
um com o controle que pode falhar:

**Qwen-Image-Edit 2511 W4A8 + `Lightning-4steps-V1.0`, 4 passos, cfg 1.** O LoRA Lightning
*é* o que torna 4 passos possíveis, então ele tem um teste funcional binário embutido:

- **R1** W4A8 **sem** LoRA a 4 passos/cfg 1 sai visivelmente inacabado ou errado (o controle
  tem de falhar). Se sair bom, o teste não distingue nada e é descartado.
- **R2** W4A8 **com** LoRA (fusão) a 4 passos obedece às três instruções (pêra verde, cachecol
  vermelho, CLOSED legível) com a imagem acabada. Refuta: parece o controle.
- **R3** W4A8 + LoRA em **bypass** também obedece. Refuta: parece o controle.
- **R4 (dica, não resultado)** o bypass fica mais perto do int8+Lightning do que a fusão fica,
  medido por MAE e pela divergência na região intocada (`analisa_edicao.py`). Trajetória livre é
  caótica, então isto é tendência, não prova.

**LTX 2.5 W4A8 + `ltx2-squish`** (LoRA de efeito do LTX 2.0, r32, só `attn1`): 49 quadros com e
sem, fusão e bypass, mesma semente.

- **R5** as chaves casam (mesmos nomes de `transformer_blocks.N.attn1.to_{q,k,v,out}` em 2.0/2.3/2.5).
  Refuta: 0 chaves mapeadas — aí é LoRA de outra arquitetura, não teste de quantização.
- **R6** o efeito do LoRA é visível no braço fundido (o vídeo com LoRA difere do sem LoRA MAIS
  do que duas sementes diferem entre si). Refuta: diferença com/sem menor que a entre sementes.

**LTX 2.3 W4A8 + `LTX23_Product_Commercial_LoRA`** (r16, inclui `to_gate_logits`, que o perfil
NÃO quantiza): mesmo desenho, depois que a conversão do 2.3 terminar.

## Não coberto

Uma força (1,0) por LoRA; um LoRA por modelo; amostra de camadas, não todas. O ruído medido é
no peso — a camada 2 é quem diz se ele importa. O bypass não foi medido na camada 1 porque, por
construção, não toca o peso; entra só na camada 2. Não se mediu LoRA sobre text encoder.
