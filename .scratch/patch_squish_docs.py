"""Registra o resultado do LoRA squish no LTX 2.5 (camada 2) no criterio e no CLAUDE.md."""
import io

p = 'bench/criterio_lora.md'
s = io.open(p, encoding='utf-8').read()
old = """Não coberto na camada 2: uma semente, três pares, força 1,0, um LoRA (o de 4 passos — o mais
robusto por construção, porque muda o regime inteiro; um LoRA de estilo sutil pode se perder onde
este não se perdeu). LTX squish e LTX 2.3 Product Commercial: renders em curso, abaixo quando
saírem."""
new = """Não coberto na camada 2: uma semente, três pares, força 1,0, um LoRA (o de 4 passos — o mais
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

LTX 2.3 Product Commercial: renders na fila, abaixo quando saírem."""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('criterio_lora: squish recorded')

p = 'CLAUDE.md'
s = io.open(p, encoding='utf-8').read()
old2 = """The layer where the LoRA looked most buried (86% survival, noise 103x the delta) is the one whose
output is intact. **Weight-space per-layer numbers rank and alarm; they do not decide** — the same
lesson this file already records for per-layer error across formats. Caveat: a 4-step LoRA changes
the whole regime and is the most robust kind; a subtle style LoRA was not tested at the output.
"""
new2 = old2 + """
On LTX 2.5 W4A8 with `ltx2-squish` (49 frames, same seed, no trigger word in the prompt — a design
hole, so this measures the cost of loading the LoRA, not its effect): merged and bypass land on the
**same** composition and sit 6.3 MAE apart, against 15-18 from the no-LoRA reference and 28 from a
seed change; neither moved the audio level or timing, while the other seed moved both (RMS -28 vs
-20 dBFS, lag +78 ms). The requantization noise perturbs the output by a third of what the LoRA does
and a quarter of what a seed does. `bench/ltx25/lora/`.
"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8').write(s)
print('CLAUDE.md: squish line added')
