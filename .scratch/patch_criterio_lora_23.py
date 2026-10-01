"""bench/criterio_lora.md: resultados da rodada 2.3 SEM gatilho (medida 2026-09-14 00:50) e as
previsoes R7-R9 da rodada COM gatilho, escritas ANTES de renderiza-la."""
import io

p = 'bench/criterio_lora.md'
s = io.open(p, encoding='utf-8').read()
old = "LTX 2.3 Product Commercial: renders na fila, abaixo quando saírem."
assert s.count(old) == 1, s.count(old)
new = """**LTX 2.3 W4A8 + `LTX23_Product_Commercial_LoRA` (r16), 49 quadros, semente 1234, SEM gatilho
no prompt (o do farol), referência = mesmo W4A8 sem LoRA, condicionamento salvo (o mesmo arquivo)
nos quatro braços (`bench/ltx23/lora/`, par em `bench/ltx23/lora_par/`), MEDIDO 2026-09-14 00:50:**

```
braço                         MAE   PSNR   SSIM    mov  | log-mel   SNR      lag
fundido                      10,65  25,6  0,696   8,43 |  0,151    7,4 dB     0 ms
bypass                       10,43  25,9  0,732   9,10 |  0,088    8,8 dB     0 ms
sem LoRA, OUTRA semente      11,10  25,1  0,627  12,23 |  0,359   -2,5 dB  -196 ms
referência (movimento 13,14, -22,6 dBFS)
fundido vs bypass, um contra o outro:  MAE 2,40 [1,47-2,74]  SSIM 0,960  | log-mel 0,136  SNR 6,3 dB  lag 0
```

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

Resultados da rodada com gatilho: abaixo quando saírem."""
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print('criterio_lora: 2.3 sem gatilho + R7-R9 escritos')
