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
| LTX 2.5 22B, W4A8 | `ltx2-squish` (LoRA do LTX **2.0**, r32) | 2304 → 1152 alvos, 0 sem mapa | 0,013–0,086 onde há delta; **zero em 16 de 24** | 0,956 (n=8) | 0,39 | 3,1x | 0,0731 → 0,0837 → 0,0837 (zero) / até 0,114 | 1,15x |
| LTX 2.5 22B, W4A8 | `LTX23_Product_Commercial` (LoRA do 2.3, r16) | 3264 → 1632, 0 sem mapa | 0,0021 | **0,909** (0,87–0,93) | 0,05 | **19x** | 0,0731 → 0,0836 → 0,0838 | 1,15x |
| Qwen-Image-Edit 2511, W4A8 | `Lightning-4steps-V1.0` (r64) | 2160 → 720, 0 sem mapa | 0,0005 | **0,858** (0,71–0,91) | 0,01 | **103x** | 0,0731 → 0,0836 → 0,0835 | 1,14x |

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
- **As duas linhas do LTX (medidas depois das três primeiras, mesmo critério) acrescentam duas
  coisas.** (a) O `ltx2-squish` traz **`lora_B` identicamente zero em 768 das 1152 matrizes** —
  todas as famílias de áudio e cruzadas (`audio_attn1/2`, `audio_to_video`, `video_to_audio`) e
  metade dos `to_out`; lido do arquivo, não deduzido. O ComfyUI casa a chave, aplica um delta
  ZERO e **requantiza a camada assim mesmo**: 2/3 das camadas tocadas por esse LoRA pagam +14 %
  de erro de peso por nada. É o mesmo mecanismo da armadilha do Wan abaixo, agora vindo de um LoRA
  legítimo da própria família. (b) No layout `asym_w4a8_int8` (codebook) a sobrevivência **não é
  1,000**: 0,956 e 0,909 — um delta pequeno perde 5–9 % ao ser requantizado, viés que o
  `convrot_w4a4` (Z-Image, Krea2: 1,000) não mostrou. E com |δ|/|W| = 0,002 o ruído acrescentado
  é **19x** o LoRA: no peso, o LoRA de produto está enterrado; se ele ainda faz efeito é a camada
  2 quem diz.
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

## Camada 2 — MEDIDO (2026-09-13, depois das previsões acima)

**Qwen-Image-Edit 2511 W4A8 + Lightning, 4 passos, cfg 1, semente 1, três pares
(`bench/qwen_edit_lora/grade_lightning_4passos.png`):**

- **R1 confirmada — o controle falha, nos dois formatos.** Sem o LoRA, a 4 passos: o cachecol
  NÃO aparece, a placa continua OPEN, a maçã vira um híbrido pintalgado, e mesa, pele e tijolo
  saem sobreafiados. INT8 e W4A8 falham do mesmo jeito. O teste distingue.
- **R2 confirmada — W4A8 + LoRA fundido obedece às três instruções**, imagem acabada: pêra verde
  lisa, cachecol de tricô vermelho, CLOSED legível.
- **R3 confirmada — bypass também obedece**, as três.
- **R4 (dica) não se sustenta**: fusão vs bypass no W4A8 divergem **1,26** na região quieta
  (`analisa_edicao.py`), contra **3,54** (fusão) e **3,85** (bypass) de cada um para o
  INT8+Lightning. O bypass NÃO fica mais perto do INT8 que a fusão — a diferença entre os dois
  caminhos de LoRA é menor que a diferença entre formatos, e cabe no ruído de trajetória.

**E a camada 1 deste mesmo par, medida no mesmo dia, dizia o contrário do que a imagem mostra.**
No peso, o Lightning sobre o W4A8 do Qwen é o pior caso da tabela: |δ|/|W| = 0,0005, sobrevivência
**0,86** (P1 refutada aqui: 14 % do delta perdido, até 29 % em camadas isoladas), cosseno 0,01,
ruído acrescentado **103x o LoRA** — a camada com LoRA é indistinguível da camada só requantizada
(0,0835 vs 0,0836). Pela camada 1, o LoRA estaria enterrado. Pela camada 2, ele funciona
inteiro. **A leitura é a mesma que esta bancada já fez para o erro por camada dos formatos: o
número por peso ordena e alarma, mas não decide — 720 camadas de ruído sem viés se cancelam no
forward, e 14 % de magnitude a menos a força 1,0 não muda o comportamento.** O que decide é a
saída, com o controle que tem de falhar ao lado.

Não coberto na camada 2: uma semente, três pares, força 1,0, um LoRA (o de 4 passos — o mais
robusto por construção, porque muda o regime inteiro; um LoRA de estilo sutil pode se perder onde
este não se perdeu).

**LTX 2.5 W4A8 + `ltx2-squish`, 49 quadros, semente 1234, referência = mesmo W4A8 sem LoRA
(`bench/ltx25/lora/contato_av.png`, `bench/ltx25/lora/comparacao_av.json`):**

```
braço                         MAE   PSNR   SSIM   mov  | log-mel   SNR      lag
squish fundido               17,92  21,6  0,877  0,52 |  0,120    6,9 dB   0 ms
squish bypass                14,74  22,7  0,886  0,56 |  0,101    8,1 dB   0 ms
sem LoRA, OUTRA semente      28,26  17,1  0,777  1,36 |  0,750   -0,6 dB  +78 ms
referência (movimento 0,48)
fundido vs bypass, um contra o outro:  MAE 6,30  SSIM 0,959  | log-mel 0,082  SNR 13,5 dB
```

- **R5 confirmada**: 2304 tensores → 1152 alvos casados, 0 sem mapa (os nomes de `attn1` são os
  mesmos em 2.0/2.3/2.5). Metade deles com `lora_B` zero, como a camada 1 já tinha lido.
- **R6 REFUTADA como escrita**: a diferença com/sem LoRA (17,9) é MENOR que a diferença entre
  duas sementes (28,3). Mas a leitura literal engana, e o desenho tinha um furo: **o prompt não
  trazia palavra-gatilho nenhuma** (o LoRA é de efeito, "squish", e foi rodado com o prompt do
  farol). O que se mediu foi a perturbação de carregá-lo, não o efeito dele. Ainda assim o dado
  útil está lá: **fundido e bypass caem na MESMA composição** (farol mais perto e mais claro, mais
  pássaros, mais espuma — visível na folha) e distam **6,3** um do outro, contra 15–18 da
  referência e 28 de uma troca de semente. O ruído de requantização moveu o vídeo por um terço do
  que o LoRA moveu e um quarto do que a semente move; e não mudou o nível nem deslocou o áudio
  (lag 0, RMS igual), enquanto a outra semente mudou os dois (RMS −28 vs −20 dBFS, lag +78 ms).
- Não coberto: sem gatilho, o efeito próprio do LoRA não foi exercitado — um teste com a palavra
  certa no prompt é o que faltaria para fechar R6 de verdade.

LTX 2.3 Product Commercial: renders na fila, abaixo quando saírem.

## Não coberto

Uma força (1,0) por LoRA; um LoRA por modelo; amostra de camadas, não todas. O ruído medido é
no peso — a camada 2 é quem diz se ele importa. O bypass não foi medido na camada 1 porque, por
construção, não toca o peso; entra só na camada 2. Não se mediu LoRA sobre text encoder.
