"""Registra a camada 2 do LoRA (Qwen Edit Lightning) no criterio, no card do Qwen Edit e no CLAUDE.md."""
import io

# ---- 1. criterio_lora.md: Qwen row + layer-2 results
p = 'bench/criterio_lora.md'
s = io.open(p, encoding='utf-8').read()
old = ("| LTX 2.5 22B, W4A8 | `LTX23_Product_Commercial` (LoRA do 2.3, r16) | 3264 → 1632, 0 sem mapa | "
       "0,0021 | **0,909** (0,87–0,93) | 0,05 | **19x** | 0,0731 → 0,0836 → 0,0838 | 1,15x |\n")
new = old + ("| Qwen-Image-Edit 2511, W4A8 | `Lightning-4steps-V1.0` (r64) | 2160 → 720, 0 sem mapa | 0,0005 | "
             "**0,858** (0,71–0,91) | 0,01 | **103x** | 0,0731 → 0,0836 → 0,0835 | 1,14x |\n")
assert old in s
s = s.replace(old, new)
old2 = "## Não coberto\n\nUma força (1,0) por LoRA"
new2 = """## Camada 2 — MEDIDO (2026-09-13, depois das previsões acima)

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
este não se perdeu). LTX squish e LTX 2.3 Product Commercial: renders em curso, abaixo quando
saírem.

## Não coberto

Uma força (1,0) por LoRA"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8').write(s)
print('criterio_lora: layer 2 recorded')

# ---- 2. Qwen Edit card: Lightning section (English)
p = 'bench/hf/qwen-image-edit-2511-quant/README.md'
s = io.open(p, encoding='utf-8').read()
anchor = "## What is NOT covered"
assert s.count(anchor) == 1
sec = """## Does a LoRA work on this file? The Lightning 4-step LoRA, with the control that has to fail

A LoRA loaded the normal way (`LoraLoaderModelOnly`) onto a quantized weight is not kept as a
separate branch: ComfyUI dequantizes the weight, adds the delta, and **requantizes it back to
4 bits** (`comfy/ops.py:1449-1457`). So the question "does the LoRA arrive?" has two layers, and
they disagree here — which is the finding.

**At the weight** (`tools/probe_lora_requant.py`, 24 sampled layers, the real
`patch_weight_to_device` path): `Qwen-Image-Edit-2511-Lightning-4steps-V1.0` matches all 720
target weights (0 unmapped keys); its delta is tiny (|δ|/|W| median 0.0005); after requantization
**86 % of the delta survives** (71–91 % per layer), the cosine between what the LoRA asked and what
landed is 0.01, and the noise the requantization adds is **103x the LoRA itself**. Per-layer weight
error against the BF16 source goes 0.0731 → 0.0835 — indistinguishable from requantizing with no
LoRA at all (0.0836). By this layer the LoRA looks buried.

**At the output**, on the three edits above, 4 steps, cfg 1.0, seed 1:

![Lightning LoRA, five arms](images/lora_lightning_4steps.png)

Columns: input · INT8 without the LoRA · INT8 + Lightning · **this file** without the LoRA ·
**this file** + Lightning merged (the normal loader) · **this file** + Lightning in bypass
(`LoraLoaderBypassModelOnly`, which keeps the delta as a BF16 branch and never touches the weight).

The Lightning LoRA is what makes 4 steps possible, so it carries its own control: **without it,
4 steps must fail — and they do, identically in INT8 and in this file**: no scarf, the sign still
says OPEN, the apple becomes a speckled hybrid, every texture oversharpened. **With it, all three
LoRA arms obey all three instructions** — smooth green pear, red knitted scarf, legible CLOSED.
Merged and bypass on this file differ by 1.26 in the region both left alone
(`tools/analisa_edicao.py`), against 3.5–3.9 between either of them and INT8 + Lightning: the
difference between the two LoRA paths is smaller than the difference between formats, and inside
trajectory noise.

**So the weight-level numbers overstate the damage, in the same way per-layer error does for
formats on this bench: they rank and they alarm, they do not decide.** Seven hundred layers of
unbiased noise cancel in the forward pass, and losing 14 % of a rank-64 delta at strength 1.0 does
not change the regime it installs. The output, with the control that has to fail beside it, is the
judge. One caveat travels with this: the 4-step LoRA is the most robust kind by construction — it
changes the whole sampling regime. A subtle style LoRA could be lost where this one was not, and
that was not measured here.

"""
s = s.replace(anchor, sec + anchor)
io.open(p, 'w', encoding='utf-8').write(s)
print('qwen card: Lightning section added')

# ---- 3. CLAUDE.md: Qwen row + output-level conclusion
p = 'CLAUDE.md'
s = io.open(p, encoding='utf-8').read()
old3 = ("LTX 2.5 22B W4A8           LTX23 Product Commercial r16 0.0021    0.909     0.05    19x         "
        "0.0731 -> 0.0836 -> 0.0838   (1.15x)\n```\n")
new3 = ("LTX 2.5 22B W4A8           LTX23 Product Commercial r16 0.0021    0.909     0.05    19x         "
        "0.0731 -> 0.0836 -> 0.0838   (1.15x)\n"
        "Qwen-Edit 2511 W4A8        Lightning 4-step r64         0.0005    0.858     0.01    103x        "
        "0.0731 -> 0.0836 -> 0.0835   (1.14x)\n```\n\n"
        "**And the output contradicts the weight, which is the finding.** Rendered the same day\n"
        "(`bench/qwen_edit_lora/grade_lightning_4passos.png`, criterion R1-R4 written first in\n"
        "`bench/criterio_lora.md`): at 4 steps without the LoRA both INT8 and W4A8 **fail** — no scarf, the\n"
        "sign stays OPEN, a speckled apple-pear hybrid, oversharpened texture (the control that had to\n"
        "fail, failed); with the LoRA, INT8, W4A8 merged and W4A8 bypass **all obey all three instructions**;\n"
        "merged vs bypass differ by 1.26 in the untouched region against 3.5-3.9 between either and INT8.\n"
        "The layer where the LoRA looked most buried (86% survival, noise 103x the delta) is the one whose\n"
        "output is intact. **Weight-space per-layer numbers rank and alarm; they do not decide** — the same\n"
        "lesson this file already records for per-layer error across formats. Caveat: a 4-step LoRA changes\n"
        "the whole regime and is the most robust kind; a subtle style LoRA was not tested at the output.\n")
assert old3 in s
s = s.replace(old3, new3)
io.open(p, 'w', encoding='utf-8').write(s)
print('CLAUDE.md: Qwen row + output conclusion')
