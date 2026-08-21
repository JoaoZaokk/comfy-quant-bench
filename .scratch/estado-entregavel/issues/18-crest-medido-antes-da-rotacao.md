# `activation_balance.py` e `plot_weight_balance.py` medem crest no tensor errado

Type: task
Status: resolved

## Question

Duas ferramentas calculam "crest factor" (max|x| / rms) sobre o tensor **pré-rotação** e
atribuem o número ao caminho ConvRot, que roda **pós-rotação**. A escala real do W4A4 é absmax
por linha do peso/ativação **já rotacionado** (`comfy_kitchen/backends/eager/convrot_w4a4.py:112`
segundo `AUDITORIA_2026-08-18.md:94`).

`tools/plot_weight_balance.py`, dentro de `measure()`:

```python
rotated = _rotate_weight(w, h, args.convrot_groupsize)   # linha 91 — calculado
...
crests.append((w.abs().max() / rms).item())              # linha 94 — usa `w`, não `rotated`
```

`tools/activation_balance.py`, dentro de `measure()`:

```python
rotated = rotate(x, h, args.convrot_groupsize)            # linha 83 — calculado
row_absmax = x.abs().amax(dim=-1).clamp(min=1e-10)        # linha 85 — usa `x`, não `rotated`
```

Em ambos os casos `rotated` existe na função — é usado para a régua de erro de quantização — mas
o crest reportado (a métrica que motiva a leitura "o W4A4 recupera ~2,9 dos 4 bits porque a
escala cobre um crest de 12-28", `CLAUDE.md`) vem do tensor errado.

`tools/weight_balance.py` é a referência que faz certo: calcula as duas coisas separadamente
(`AUDITORIA_2026-08-18.md:94` cita `weight_balance.py:82-100` como o exemplo correto).

## O que a medição sustenta

Nada medido nesta rodada — GPU proibida. O achado é leitura direta das três funções lado a lado;
não precisa de GPU para confirmar o mecanismo (comparar as três linhas basta), mas precisa de GPU
para saber **quanto** o número publicado muda quando corrigido.

## Critério de fechamento

Fecha quando `activation_balance.py` e `plot_weight_balance.py` calcularem crest sobre `rotated`
(não sobre `w`/`x`), no mesmo padrão que `weight_balance.py` já usa. Não fecha só reescrevendo o
comentário — o cálculo tem que mudar de tensor.

Se a correção for feita sem GPU disponível (é troca de uma variável, não precisa rodar), ainda
assim marcar como **não remedido numericamente** até alguém rodar as duas versões lado a lado e
registrar quanto o crest publicado muda — isso é `[GPU]` e fica para outra rodada.

## Resolução — 2026-08-21, lock `claude:rodada3-gpu-lane`

**Consertado nos dois arquivos, e o número muda muito — de forma desigual entre modelos.**

O conserto **não** foi trocar `w` por `rotated`. Foi reportar **os dois**, porque são estatísticas
diferentes: o cru diz o que a rotação tem de consertar, o rotacionado diz o que a escala precisa
cobrir. Trocar em silêncio mudaria o sentido de um número já publicado sem que ninguém percebesse.

```
.\python_embeded\python.exe -s tools\plot_weight_balance.py \
  --model "LTX2.5 distilled=D:/ComfyUI-Models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --model "Gemma4 12B TE=D:/ComfyUI-Models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --out <scratch>/weight_balance_rot.png
```

| modelo | crest cru | crest rotacionado | fator |
|---|---|---|---|
| LTX 2.5 22B distilled | 16,1 | **7,0** | 2,3x |
| Gemma4 12B TE | 19,5 | **15,7** | 1,24x |

8 camadas amostradas por modelo, `convrot_groupsize` padrão.

**O achado que o gráfico antigo escondia:** a rotação vale quase o dobro no LTX do que no Gemma.
Os dois pareciam equivalentes (16,1 contra 19,5) e não são — depois da rotação são 7,0 contra 15,7.
Isso é diferença de arquitetura, e estava invisível porque as duas barras mediam o tensor errado.

O rótulo do gráfico dizia literalmente **"median crest N — the W4A4 scale reaches to here"**
apontando para o número pré-rotação. Agora aponta para o rotacionado e diz `(rotated)`.

**Uma corrida basta aqui, e vale dizer por quê:** isto não é medição de tempo. É estatística
determinística sobre bytes de peso lidos do disco — a mesma entrada dá o mesmo número. A regra
"uma corrida não é medição" existe para efeitos que a contenção move; este não é um deles.

**Não coberto:** `activation_balance.py` recebeu o mesmo conserto (imprime `crest per token, RAW`
e `crest per token, ROTATED`) mas **não foi executado** — precisa de ativações capturadas de uma
corrida de sampling, que é outra ferramenta e outro ticket. O conserto dele está provado por
`py_compile` e por leitura lado a lado com o de `plot_weight_balance.py`, não por execução.
Nenhum número de erro de quantização foi re-medido; a escada A/B/C/D acima é a que a ferramenta já
produzia e não depende do crest.
