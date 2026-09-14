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

**LTX 2.3 W4A8 + `LTX23_Product_Commercial_LoRA` (r16), 49 quadros, semente 1234, SEM gatilho
no prompt (o do farol), referência = mesmo W4A8 sem LoRA, condicionamento salvo (o mesmo arquivo)
nos quatro braços, MEDIDO 2026-09-14 00:50 — **INVÁLIDO, descoberto às 01:00** (ver a nota logo
abaixo da tabela; movido para `bench/ltx23/lora_ltxv_saver/` e `lora_par_ltxv_saver/`):**

```
braço                         MAE   PSNR   SSIM    mov  | log-mel   SNR      lag
fundido                      10,65  25,6  0,696   8,43 |  0,151    7,4 dB     0 ms
bypass                       10,43  25,9  0,732   9,10 |  0,088    8,8 dB     0 ms
sem LoRA, OUTRA semente      11,10  25,1  0,627  12,23 |  0,359   -2,5 dB  -196 ms
referência (movimento 13,14, -22,6 dBFS)
fundido vs bypass, um contra o outro:  MAE 2,40 [1,47-2,74]  SSIM 0,960  | log-mel 0,136  SNR 6,3 dB  lag 0
```

- **NOTA (01:00): estes quatro braços NÃO são o farol.** O controle de identidade (W4A8, 249 quadros,
  mesmo condicionamento salvo, contra o render com encoder vivo) deu **MAE 75,9 / SSIM 0,19 /
  log-mel 1,05** — quadros de ruído marrom, áudio de ruído (`bench/ltx23/cond_identity_ltxv_saver/
  contato_av.png`). Causa, lida no código depois de medir: o encoder do LTX 2.3 devolve
  `extra = {"unprocessed_ltxav_embeds": True}` (`comfy/text_encoders/lt.py:201-204`) e o modelo só
  aplica `caption_projection` + connectors com essa chave (`comfy/model_base.py:1185` →
  `av_model.py:583`); o `LTXVSaveConditioning` grava só o tensor e o `LTXVLoadConditioning` devolve
  sem a chave — o contexto de 6144 canais passa por já processado e entra cru na cross-attention.
  Os números abaixo medem LoRA sobre ruído condicionado errado; ficam como registro do que a
  ferramenta errada produz, e são substituídos pela rodada da fila j (condicionamento completo por
  `tools/ltx_encode_lowcommit.py` + `VoidLoadConditioningFull`). O par fundido-vs-bypass de 2,40 também.
- No vídeo, aqui a distância com/sem LoRA (10,4–10,7) fica DENTRO da distância semente-a-semente
  (11,1) — no 2.5 era 15–18 contra 28. No áudio o LoRA move 2,4–4x menos que a semente (0,088–0,151
  contra 0,359) e não desloca nada (lag 0 contra −196 ms). Os dois braços com LoRA são mais calmos
  que a referência (movimento 8,4–9,1 contra 13,1); a outra semente não (12,2).
- **Fundido e bypass distam 2,40 entre si** (SSIM 0,960), contra 10,4–10,7 de cada um para a
  referência: o ruído de requantização (sobrevivência 0,908, ruído 20x o delta no peso) move o
  vídeo por um quarto do que o LoRA move. Mesma leitura do 2.5 (6,3 contra 15–18).
- Mesmo furo do 2.5: sem o gatilho, isto mede a perturbação de carregar o LoRA, não o efeito dele.
  O gatilho existe e está no arquivo: `ss_tag_frequency = {"1_srx_commercial": {"srx_commercial": 1}}`,
  `ss_base_model_version = ltx2`.

**Rodada SEM gatilho refeita sobre condicionamento VÁLIDO (fila j, MEDIDO 2026-09-14 01:25;
`bench/ltx23/lora/`, par em `bench/ltx23/lora_par/`).** Mesmo prompt do farol, 49 quadros, W4A8,
os quatro braços lendo o MESMO arquivo de condicionamento (`ltx23condf`, formato completo, controle
de identidade MAE 1,74 contra o encoder vivo):

```
braço                    MAE vs ref        PSNR    SSIM   mov  | log-mel  SNR      lag       RMS
LoRA fundido            22,92 [21,1-27,0]  16,79  0,644  2,91 |  0,543  -1,5 dB   -0,1 ms  -12,0 dBFS
LoRA bypass             23,98 [22,4-27,5]  16,48  0,632  2,96 |  0,613  -1,5 dB  -17,7 ms  -14,1 dBFS
sem LoRA, semente 4321  80,99 [80,8-81,2]   8,55  0,452  1,91 |  1,083  -2,9 dB  -85,9 ms  -12,2 dBFS
fundido vs bypass        5,13 [4,7-5,8]    26,60  0,894       |  0,232  -0,05 dB   0,0 ms
referência: sem LoRA, semente 1234, RMS -11,9 dBFS, mov 2,0
```

- **Os números da rodada inválida (10,65 / 11,10 / 2,40) não sobrevivem em valor, mas a ORDEM
  sobrevive**: LoRA move menos que uma semente (22,9-24,0 contra 81,0), e fundido-vs-bypass move
  um quarto do que o LoRA move (5,1 contra 23). Mesma leitura do 2.5 (6,3 / 15-18 / 28).
- **O que o olho vê na folha**: os dois braços com LoRA mantêm a composição da referência (mesmo
  farol, mesmas rochas, mesma câmera) e mudam o acabamento — céu mais claro, mais aves, tons
  limpos; a outra semente muda a composição inteira (farol centrado, céu laranja), que é o que
  MAE 81 com faixa [80,8-81,2] quer dizer. Sem gatilho o LoRA já age como estilo; o que ele NÃO faz
  é mudar o que é gerado.
- Áudio: fundido não muda nível (-12,0 contra -11,9) nem tempo (-0,1 ms); bypass desloca 17,7 ms e
  cai 2,2 dB (dentro dos ±3 dB da A2); a outra semente desloca 85,9 ms. Fundido e bypass entre si:
  0,0 ms, -0,05 dB.
- Sampler (barra do tqdm no log do servidor, 49 quadros, 8 passos): sem LoRA 0,55 s/it, fundido
  0,55 s/it, **bypass 0,69 s/it** — o ramo de baixo posto no forward custa 25% por passo; o fundido
  custa zero porque o kernel é o mesmo. A parede da corrida (170-200 s) é 97% carga do modelo por
  SMB com `--cache-none`, não sampler.

**Rodada COM gatilho — previsões escritas ANTES de renderizar (2026-09-14 00:58).** Prompt:
`srx_commercial, a sleek matte black wireless headphone rotating slowly on a white pedestal, soft
studio lighting, clean seamless background, product commercial, gentle camera push-in`; negativo
padrão; 49 quadros; braços: sem LoRA semente 1234 (referência), fundido 1234, bypass 1234, sem LoRA
semente 4321. Condicionamento gerado FORA do servidor por `tools/ltx_encode_lowcommit.py` (o
encoder de 24 GB não cabe no commit do servidor pelo leitor normal — ver CLAUDE.md), o mesmo
arquivo nos quatro braços; antes disso o mesmo tool recodifica o prompt do farol e compara com o
`ltx23cond` que o servidor gravou (autoteste do caminho, resultado impresso, não presumido).

- **R7** com o gatilho, `MAE(fundido vs referência) > MAE(outra semente vs referência)` — o LoRA
  muda o que é gerado, não só onde a trajetória cai. Refuta: fundido ≤ outra semente (como ficou
  sem gatilho: 10,65 contra 11,10).
- **R8** fundido e bypass ficam mais perto um do outro do que qualquer um da referência
  (`MAE(fundido, bypass) < min(MAE(fundido, ref), MAE(bypass, ref))`), como nas duas rodadas sem
  gatilho (2,40 contra 10,4; 6,3 contra 15). Refuta: o par ≥ o mínimo.
- **R9 (olho, na folha, não é métrica)** os braços com LoRA têm cara de comercial — fundo limpo,
  produto centrado, luz de estúdio, movimento suave — e a referência sem LoRA, com o mesmo prompt,
  não necessariamente. Se a referência já parecer comercial, o gatilho não separa nada e R9 fica
  indecidível, não confirmada.

**Resultados da rodada com gatilho (MEDIDO 2026-09-14 01:37; `bench/ltx23/lora_trigger/`, par em
`bench/ltx23/lora_trigger_par/`; condicionamento `ltx23cond_srx`, formato completo, o mesmo arquivo
nos quatro braços):**

```
braço                    MAE vs ref        PSNR    SSIM   mov  | log-mel  SNR      lag       RMS
LoRA fundido            40,85 [35,2-47,3]  10,44  0,702  4,55 |  0,588  -1,2 dB   -0,3 ms  -14,9 dBFS
LoRA bypass             40,63 [35,7-47,8]  10,50  0,705  4,70 |  0,719  -1,2 dB  +14,8 ms  -17,0 dBFS
sem LoRA, semente 4321  57,49 [45,7-74,7]   9,33  0,699  5,88 |  1,747  -3,4 dB  +41,5 ms  -11,8 dBFS
fundido vs bypass        7,54 [5,3-11,5]   22,21  0,870       |  0,328  -0,5 dB    0,0 ms
referência: sem LoRA, semente 1234, RMS -12,3 dBFS, mov 1,25; controles: silêncio 7,90, ruído branco 2,97
```

- **R7 REFUTADA.** 40,85 < 57,49: mesmo com o gatilho, o LoRA move menos que uma semente. Mais
  perto do que sem gatilho (0,71 da semente contra 0,28), mas do mesmo lado. A métrica não
  distingue "muda o que é gerado" de "cai em outro lugar" — ver R9.
- **R8 CONFIRMADA.** 7,54 < min(40,85; 40,63); SSIM 0,870 entre fundido e bypass contra 0,70 de
  cada um para a referência. O ruído de requantização (sobrevivência 0,908, ruído 20x o delta no
  peso) move o vídeo por um quinto do que o LoRA move — terceira rodada com essa proporção (2.5:
  6,3/15; farol: 5,1/23; gatilho: 7,5/41).
- **R9 INDECIDÍVEL, pela cláusula escrita antes:** a referência sem LoRA já é um comercial — fones
  centrados em pedestal branco, luz de estúdio, fundo limpo — porque o prompt é um comercial. O
  gatilho não separa "cara de comercial". O que a folha mostra e a métrica confirma é outra coisa:
  **os dois braços com LoRA executam o "rotating slowly" do prompt** (vista lateral no quadro 1,
  frontal no 49; movimento 4,55-4,70) **e a referência quase não gira** (movimento 1,25, 3,6x
  menos); o fone dos braços com LoRA é outro desenho, mais refinado. Isso é efeito do LoRA no que
  é gerado, visível a olho — e não é R9, que perguntava por estética, não por obediência ao
  movimento. Fica como observação, não como previsão confirmada.
- Áudio: fundido não desloca (-0,3 ms) e cai 2,6 dB; **bypass desloca 14,8 ms e cai 4,7 dB**,
  fora dos ±3 dB da A2. Segunda rodada em que o bypass abaixa o nível mais que o fundido (farol:
  -2,2 dB contra -0,1). Duas observações no mesmo sentido; mecanismo não isolado.
- Sampler (barra do tqdm, 49 quadros, 8 passos): sem LoRA 0,55 s/it, fundido 0,55, bypass 0,68,
  outra semente 0,55 — o bypass custa 24% por passo, o fundido nada.
- **O que responde à pergunta do dono ("os LoRAs funcionam como deveriam?") no 2.3 W4A8:** o LoRA
  chega à saída — muda o vídeo (41 MAE, o giro pedido, outro desenho) e o som (log-mel 0,59 contra
  1,75 de uma semente) — e fundido e bypass concordam entre si (7,5). O que a bancada NÃO tem é um
  render do mesmo LoRA sobre a referência BF16 para dizer se o efeito é o mesmo que no modelo
  original; não foi feito nesta rodada.

## Não coberto

Uma força (1,0) por LoRA; um LoRA por modelo; amostra de camadas, não todas. O ruído medido é
no peso — a camada 2 é quem diz se ele importa. O bypass não foi medido na camada 1 porque, por
construção, não toca o peso; entra só na camada 2. Não se mediu LoRA sobre text encoder.
