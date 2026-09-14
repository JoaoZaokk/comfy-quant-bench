# Critério escrito ANTES de medir: fechamento das seis conversões sem verificação de saída, e o empate do áudio

Escrito 2026-09-14 02:20, com a fila j terminada e nada abaixo ainda medido. Cada item tem a
previsão e o que a refuta. Os números que vierem depois entram na seção "Vereditos", com hora.

## Estado no momento da escrita

- 53 sidecars `.quant.json` nas raízes do yaml; **28 são órfãos** (peso apagado numa limpeza
  anterior; a maioria já publicada no Hub). Dos seis sem verificação de saída, cinco têm peso:
  `gemma_3_12B_it_w4a8` (P:), `gemma_3_12B_it_heretic_w4a4_convrot`, `_w4a4_smooth`,
  `ltx-2.5-22b-distilled-transformer-bf16_int8` e `_int8_convrot` (D:, 20,03 GiB cada). O sexto,
  `capybara_v0.1_w4a8`, é órfão: só reconvertendo (fonte `capybara_v0.1.safetensors` presente).
- Condicionamento do farol em formato completo já existe (`ltx23condf`, encoder de fábrica BF16 na
  3080 Ti); render W4A8 sobre ele: `.scratch/ltx23av_w4a8_condf.json` (identidade MAE 1,74).
- Licença Krea 2 lida (pdftotext): distribuição de derivados permitida (§2.1) com nome começando
  por "Krea", cópia do acordo, arquivo Notice e declaração de modificação (§3.1–3.2); comercial só
  abaixo de US$ 1M/ano (§2.3). Vai subir cumprindo isso.

## A — o Gemma de fábrica em W4A8 como encoder do LTX 2.3

Mesmo prompt, mesma projeção, encoder `gemma_3_12B_it_w4a8` (336 camadas `asym_w4a8_int8`, matemática
dequantizada pelas duas travas do ComfyUI), na 3080 Ti, comparado com `ltx23condf` (BF16, mesma
placa). Depois render W4A8 249 quadros sobre esse condicionamento, comparado com o render sobre o
condicionamento BF16.

- **A1** rel-L2 do condicionamento entre 0,05 e 0,30 contra o BF16 (o heretic W4A8 mediu 2,1e-1 de
  erro na destrava; é outra grandeza, mesma ordem). Refuta: < 0,01 (não quantizou) ou > 0,5.
- **A2** o render é a MESMA cena (farol, rochas, aves) com MAE entre 3 e 12 contra o render BF16 —
  acima do rodapé do condicionamento (1,74), na ordem da distância transformer (10,39). Áudio
  log-mel entre 0,1 e 0,4. Refuta: cena diferente (MAE > 30), ruído, ou MAE abaixo de 1,74.

## D — os três builds do Gemma heretic contra o heretic BF16

Encoder heretic BF16 (`gemma_3_12B_it_heretic.safetensors`, 23,5 GB, no disco desde 2026-09-01)
codifica o farol → `ltx23cond_hbf16`. Depois `_w4a8`, `_w4a4_convrot`, `_w4a4_smooth`, cada um
comparado com o hbf16. Render W4A8 249 quadros sobre cada um dos quatro; referência = render sobre
hbf16.

- **D1** ordem do rel-L2 contra o BF16: `w4a8 < w4a4_smooth < w4a4_convrot`. O SmoothQuant existe
  para tirar outlier de canal antes do 4 bits (razão de outlier 82,5 → 9,6 medida na conversão);
  o smoke de kernel deu 0,187 no smooth contra ~0,25 no convrot. Refuta: convrot ≤ smooth (suavizar
  não paga no condicionamento), ou w4a8 pior que algum w4a4.
- **D2** os quatro renders são a mesma cena; a ordem de MAE contra o render hbf16 repete a ordem de
  D1. Refuta: qualquer braço em cena diferente ou ruído; ou ordem trocada.
- **D3 (contexto, não previsão forte)** heretic BF16 contra fábrica BF16: rel-L2 > 0,3 (são pesos
  diferentes) e render em cena parecida mas não igual (MAE 10–40).

## C — os dois int8 nossos do LTX 2.5

`int8_tensorwise` sem e com ConvRot, 20,03 GiB cada, no protocolo do 2.5 (249 quadros, 3 passos,
encoder int8 da Lightricks VIVO no processo, como os três braços do card). Referência: o render
BF16 já existente (`.scratch/ltx25av_bf16.json`).

- **C1** os dois ficam entre 3 e 6 de MAE do BF16, na faixa do int8 da Lightricks (4,10), e melhor
  que o nosso W4A8 (7,81). Refuta: > 7,81 (int8 nosso pior que 4 bits = build quebrado) ou > 2x a
  Lightricks.
- **C2** o com ConvRot é igual ou melhor que o sem (a rotação reduz outlier de ativação para int8).
  Refuta: sem ConvRot melhor por mais de 20%.
- **C3 (risco)** o servidor sobrevive às duas cargas de 20 GB pelo leitor normal (+40 GB de commit
  transiente cada). Já sobreviveu duas vezes (int8 Lightricks e W4A8). Se morrer, é a loteria do
  commit, e o fallback é condicionamento salvo do 2.5.

## E — capybara W4A8, reconvertido

`quant_w4a8 --profile hunyuan_video_15` sobre `capybara_v0.1.safetensors` na 3080 Ti, saída
`capybara_v0.1_w4a8r`. Ladder do Hunyuan (maçã, semente 12345, 6 passos, 480 px, 1 quadro),
referência = capybara BF16.

- **E1** imagem coerente (maçã na mesa), divergência de latente na faixa do `hv15_w4a8` (o Hunyuan
  de fábrica em W4A8 renderiza bem). Refuta: imagem destruída — e aí o capybara entra como segunda
  família onde o W4A8 quebra, o que seria achado.
- **E2** sha256 da reconversão ≠ do sidecar órfão? Não previsto — o conversor é determinístico
  salvo codebook; se bater com `4317156b…`, melhor; se não, registra.

## G — o empate do áudio (W4A8 = Q6_K a 8 passos) testado a 3 passos

Mesmos três braços (BF16 por GGUF, Q6_K, W4A8), mesmo condicionamento `ltx23condf`, sigmas
`0.909375, 0.725, 0.421875, 0.0` (a cauda da lista de 8 do 2.3, que é exatamente o schedule de 3
passos do 2.5), 249 quadros.

- **G1** se o empate a 8 passos é saturação (fase descorrelacionada, log-mel no piso), a 3 passos
  o Q6_K separa: log-mel(Q6_K) ≤ 0,75 × log-mel(W4A8). Refuta: os dois dentro de 10% um do outro —
  e aí a leitura muda para "o ramo de áudio é insensível à quantização do transformer neste nível",
  hipótese nova, não medida.
- **G2** no vídeo a ordem Q6_K < W4A8 se mantém a 3 passos. Refuta: inverte.

## O que NÃO vai estar coberto, já sabido

Uma semente, um prompt, 512 px, uma placa por braço. Text encoders medidos com as travas do ComfyUI
(matemática dequantizada) — é o que o usuário recebe. O capybara é uma reconversão: o arquivo
publicado, se for, é o novo, não o de 2026-09-01. Nenhuma métrica de áudio validada contra ouvido.

## Achados de execução (antes de qualquer veredito, 03:05)

- **O caminho medido nos encoders quantizados é o DESTRAVADO.** Esta árvore do ComfyUI carrega o
  patch `patches/comfyui_text_encoder_quantized_math.patch` (quatro arquivos modificados, flag
  `--disable-quantized-text-encoder` em `comfy/cli_args.py:111`), então um encoder quantizado roda
  matemática quantizada por padrão — o que este servidor entrega. O ComfyUI de fábrica roda
  dequantizado. Os encodes de A e D são feitos nos DOIS caminhos (a ferramenta ganhou
  `--stock-locks`); só o destravado é renderizado. A1 e D1 valem para o destravado; o travado entra
  como coluna ao lado.
- **Encoder quantizado de 12B não cabe na 3080 Ti com os ~8 GB que sobram do cortex.** Os quatro
  encodes quantizados morreram de `CUDA error: out of memory` na 3080 Ti (W4A8 8,3 GB, W4A4 6,9 GB),
  enquanto o BF16 de 23,5 GB passa pela carga parcial do ComfyUI. Refeitos na 3090 (passada 2), com
  o servidor derrubado. Não medido: se a carga parcial simplesmente não cobre o caminho quantizado
  ou se é a ativação dequantizada que estoura.
- **A reconversão do capybara recusou por disco**: F: tinha 10,34 GiB livres, o conversor pede
  16,51 (estimativa conservadora — o hv15_w4a8 real tem 8,24 GiB). Refeita para P: (2 TB livres).
- **O sampler do BF16 a 3 passos (G) rodou sob pressão de memória** — encode do heretic BF16 na
  3080 Ti ao mesmo tempo (commit chegou a 183 GiB de pico, pagefile expandido para 166) — e fez
  3 passos em 2:48 (41–56 s/it) contra 8,05 s/it na fila j. **A distância de G não depende disso;
  a coluna de tempo de G não vale.**

## Vereditos

**G — G1 e G2 CONFIRMADAS (03:16; `bench/ltx23/av_3steps/`).** A 3 passos o som separa, e muito:

```
braço           MAE vs BF16        SSIM   mov  | log-mel   SNR onda   conv    RMS
Q6_K terceiro    7,52 [6,4-8,7]   0,821  2,73 |  0,063    +8,35 dB   0,164  -32,8 dBFS
W4A8 nosso      13,32 [10,8-15,5] 0,672  2,88 |  0,223    -1,05 dB   0,687  -36,1 dBFS
referência BF16 (GGUF): RMS -32,7 dBFS, mov 2,71; controles: silêncio 7,13, ruído branco 1,96
```

log-mel(Q6_K) = 0,063 ≤ 0,75 × 0,223 = 0,167 → G1 confirmada (3,5x de separação). Q6_K < W4A8 no
vídeo → G2 confirmada. O detalhe que explica o empate de 8 passos: a 3 passos o **Q6_K continua em
fase com a referência** (SNR de onda +8,35 dB, convergência espectral 0,164) e o W4A8 não (−1,05 dB,
0,687); a 8 passos os dois estavam em −1 dB — as duas trajetórias de áudio já tinham se
descorrelacionado e o log-mel dos dois caiu no mesmo piso (~0,16). **O empate era saturação da
métrica sob descorrelação de fase, não insensibilidade do ramo de áudio.** Bônus que não estava
previsto: a 3 passos o W4A8 abaixa o nível em 3,4 dB (fora dos ±3 dB da A2), o que não fazia a 8.

**Ressalva que muda como isto se lê:** a 3 passos o LTX 2.3 distilled 1.1 **não renderiza a cena
normal** — os três braços (BF16 incluído) mostram galhos secos e um bando de aves em tons de
cinza/marrom, com o farol quase escondido (`contato_av.png`), em vez do farol/rochas/ondas dos 8
passos. O schedule de 3 passos (a cauda do de 8) é fora do regime do modelo. A comparação é
internamente consistente (mesmo condicionamento, mesma referência degradada) e responde só à
pergunta do mecanismo do empate; **não diz nada sobre o 2.3 nas configurações em que se usa.**
Tempo do sampler nesta rodada: BF16 41,6 s/it sob pressão de memória (não vale); Q6_K 5,09 e W4A8
2,26 s/it, iguais aos da fila j.

**C — C2 e C3 CONFIRMADAS; C1 CONFIRMADA para o int8+ConvRot e REFUTADA para o int8 sem rotação
(03:36; `bench/ltx25/int8_ours/`).** Protocolo do card do 2.5 (encoder int8 da Lightricks VIVO, 3
passos, 249 quadros, semente 1234), referência o render BF16 de 13/09:

```
braço                               MAE vs BF16        PSNR   SSIM   mov  | log-mel  SNR onda  conv   RMS    | s/it (3 passos)   parede
int8 Lightricks (comfy-int8-convrot) 4,10 [3,61-4,61]  29,71  0,941  1,78 |  0,041   +11,22   0,124  -38,8  | (passada 4)       801 s
int8_tensorwise nosso, SEM rotação   8,23 [7,72-8,87]  25,15  0,889  1,59 |  0,060    +8,97   0,166  -38,6  | 2,02              494 s
int8_tensorwise + ConvRot nosso      4,19 [3,64-4,71]  30,18  0,944  1,78 |  0,043   +11,58   0,119  -38,7  | 2,12              482 s
W4A8 nosso                           7,81 [7,16-8,84]  25,39  0,895  1,71 |  0,120    +3,42   0,311  -38,3  | (passada 4)       511 s
referência BF16: mov 1,73, RMS -38,7 dBFS; 8,48 s/it (25 s; comfy_8190_b.err, parede 761 s)
controles: silêncio 6,98; ruído branco no RMS da referência 1,47
```

- **C2 confirmada, e não por pouco:** com a rotação o nosso int8 cai em cima do da Lightricks (4,19
  contra 4,10 nos quadros, faixas sobrepostas; 0,043 contra 0,041 no som; SNR 11,6 contra 11,2) —
  a mesma receita, `int8_tensorwise` + ConvRot, reproduz o mesmo resultado a partir do mesmo BF16.
  Sem a rotação, os MESMOS 8 bits de peso e de ativação ficam **2x mais longe nos quadros** (8,23) e
  1,4x no som (0,060). A folha mostra o que o número diz: os dois braços rotacionados ficam na
  composição do BF16; o int8 sem rotação desenha o farol maior, mais perto e mais vermelho — a mesma
  deriva que o W4A8 faz.
- **C1 refutada no int8 sem rotação, e a leitura que eu tinha colado à refutação ("> 7,81 = build
  quebrado") não vale.** 8,23 contra 7,81 é 5 %, dentro da sobreposição das faixas por quadro
  ([7,72-8,87] contra [7,16-8,84]); a cena é a mesma (SSIM 0,889 contra 0,895) e no som o int8
  sem rotação ainda ganha do W4A8 por 2x (0,060 contra 0,120; SNR +9,0 contra +3,4). O que a
  refutação mede de verdade: **nos quadros, tirar a rotação custa o mesmo que descer o peso para
  4 bits com rotação.** Um int8 por tensor sobre uma ativação de 8192 canais com outlier é uma
  escala grossa; a rotação é o que torna essa escala aceitável. Não é build quebrado; é a receita
  errada.
- **C3 confirmada:** o servidor k sobreviveu às duas cargas de 20 GB pelo leitor normal (commit
  transiente de +40 GB cada), 4 de 4 nesta bancada para arquivos de 20 GB (int8 Lightricks, W4A8 e
  os dois nossos). A loteria do commit é dos arquivos de 39 GB.
- **Velocidade:** os dois int8 nossos amostram 3 passos em 6 s (2,02 e 2,12 s/it, barra do sampler
  no `comfy_8190_k.err`) contra 25 s do BF16 (8,48 s/it, carga parcial). O por-passo do int8 da
  Lightricks e do W4A8 não foi capturado em 13/09 — o log daquele servidor foi sobrescrito, e o
  único que sobreviveu (b) só entregou o BF16 depois de a ferramenta aprender a ler `Prompt
  executed in 00:12:41` (formato do ComfyUI acima de 600 s, `main.py:400`). Passada 4 re-renderiza
  os dois só para a barra.
- A parede da corrida ordenou de novo pelo carregamento (482-801 s para 6 s de sampler); o
  `s_por_quadro` do JSON não é velocidade.

**A — A2 CONFIRMADA; A1 ficou ABAIXO da faixa prevista, sem cair na refutação (03:50–03:54;
`bench/ltx23/encoder_cond_factory.json`, `bench/ltx23/encoder_w4a8/`).** Os encodes quantizados
foram refeitos na 3090 (passada 2, servidor derrubado), nos DOIS caminhos:

```
condicionamento vs Gemma BF16 (mesmo prompt)   rel-L2 pos/neg    cos       max|d|  bit-iguais(bf16)
W4A8 destravado (kernel int8, esta árvore)      0,0429 / 0,0403  0,99907    7      4,2 % / 4,9 %
W4A8 travado (dequantizado, ComfyUI de fábrica) 0,0420 / 0,0396  0,99911    7      4,4 % / 5,0 %
heretic BF16 (escala: outro modelo, mesmo prompt) 0,0997 / 0,0929  0,99506  12      1,4 % / 1,4 %

render W4A8 (transformer) sobre o condicionamento W4A8 destravado, contra o render sobre o BF16:
MAE 6,75 [6,40-7,15]  PSNR 24,67  SSIM 0,894  mov 1,90 (ref 2,02) | log-mel 0,087  SNR +4,35 dB  lag 0  conv 0,248  RMS -23,8 (ref -24,0)
```

- **A1:** previsto rel-L2 entre 0,05 e 0,30; medido 0,043 / 0,040. Errei o piso por 15 %, para o
  lado bom; a refutação (< 0,01 = não quantizou; > 0,5) não disparou. O que o par de linhas diz e
  que eu não tinha previsto: **o peso de 4 bits custa 0,04 e o caminho de ativação int8 custa 0,001**
  — destravar as duas travas é de graça no W4A8, nas duas direções (pos e neg discordam de sinal).
- **A2 confirmada:** mesma cena (farol, rochas, aves, mesma luz), MAE 6,75 dentro de 3–12, som 0,087
  logo abaixo de 0,1–0,4 (melhor que o previsto). Abaixo da distância que a quantização do próprio
  transformer impõe (10,39) e 4x o rodapé do condicionamento salvo contra vivo (1,74). Nível igual,
  lag zero, movimento não congelou.
- **Discrepância que fica aberta:** a medição de 2026-08-31 pelo monkeypatch (card do heretic)
  disse que destravar o W4A8 adicionava 1,84e-1 de erro relativo no output do encoder; hoje, pela
  flag, os dois caminhos diferem 0,001 no condicionamento projetado. Outro tensor (saída crua do
  encoder contra condicionamento depois da projeção), outro instrumento (monkeypatch contra patch
  na fonte), mesmo arquivo. Uma das duas medições está errada ou a projeção esconde a diferença;
  não resolvido hoje, e registrado nos dois cards em vez de trocado em silêncio.
- A folha de contato saiu com o título errado na primeira gravação ("encoder locked = dequantized
  math", herdado do script da passada 1) e foi regravada com o título certo sem tocar nos números.

**D — D1 CONFIRMADA (nos dois caminhos); D2 CONFIRMADA na ordem e REFUTADA na cena para os dois
W4A4; D3 REFUTADA (04:08; `bench/ltx23/encoder_cond_heretic.json`, `bench/ltx23/encoder_heretic/`,
`bench/ltx23/encoder_heretic_vs_factory/`).** Referência: render W4A8 sobre o condicionamento do
heretic BF16 (encodado na 3080 Ti); os três builds encodados na 3090, destravados e travados; só o
destravado renderizado. Sampler 2,30 s/it nos quatro renders (o encoder não está no processo).

```
build do heretic     rel-L2 vs heretic BF16 (pos/neg)        | render vs render-BF16
                     destravado (kernel)   travado (dequant.) | MAE               PSNR   SSIM   mov  | log-mel  SNR     RMS
W4A8                 0,0421 / 0,0410       0,0427 / 0,0402    |  5,45 [4,94-5,99] 24,67  0,915  1,94 |  0,080   +2,59  -23,8
W4A4 smooth          0,1583 / 0,1480       0,0818 / 0,0740    | 29,82 [29,2-30,6] 14,67  0,645  1,69 |  0,220   -0,24  -24,8
W4A4 convrot         0,2207 / 0,2453       0,1111 / 0,0924    | 38,12 [35,9-40,4] 13,11  0,567  1,95 |  0,193   -0,33  -24,4
(D3) fábrica BF16 → heretic BF16: rel-L2 0,0997 / 0,0929, cos 0,995; render MAE 9,34, SSIM 0,832, log-mel 0,131
referência: mov 1,90, RMS -23,8; controles: silêncio 8,38, ruído branco 1,76
```

- **D1 confirmada, e nos dois caminhos:** w4a8 (0,042) < smooth (0,158) < convrot (0,221) destravado;
  0,043 < 0,082 < 0,111 travado. Suavizar canal antes da rotação paga 1,4x no condicionamento
  (a pergunta que ficou aberta em 2026-09-01: paga, e não é o termo maior). **Destravar custa 2x no
  W4A4** (0,111 → 0,221; 0,082 → 0,158) e 0,001 no W4A8: a ativação de 4 bits é o termo grande,
  a de 8 bits é de graça.
- **D2: a ordem repete (5,45 < 29,8 < 38,1) e os dois W4A4 NÃO são a mesma cena.** A folha mostra
  o que o número não diz: não é ruído — é um farol coerente, bem iluminado, numa ilha rochosa sob
  céu azul com nuvens brancas, **de dia**, onde o prompt pede "dusk" e o BF16, o W4A8 e o encoder
  de fábrica desenham a silhueta ao crepúsculo. O condicionamento derivou o suficiente (cos
  0,977–0,988) para perder uma palavra. A refutação escrita era "cena diferente (MAE > 30)": o
  convrot (38,1) dispara; o smooth (29,8) fica 0,6 % abaixo do número e é visivelmente a mesma cena
  diurna — a folha decide, não a segunda casa decimal. **Os pesos W4A4 do heretic não sobem para o
  Hub**; sobem as provas.
- **D3 refutada:** heretic BF16 contra fábrica BF16 mede rel-L2 0,10 (previsto > 0,3) e o render
  fica a 9,34 (previsto 10–40, errado por 7 %). A ablação move o condicionamento 2,3x mais que a
  quantização W4A8 e MANTÉM a cena; os W4A4 movem 1,6–2,2x mais que a ablação e a perdem. A virada
  de cena está em algum ponto entre 0,10 e 0,16 de rel-L2 neste prompt: faixa, não linha (um prompt,
  uma semente).
- **Não coberto e vai para a passada 6:** os W4A4 no caminho TRAVADO (o do ComfyUI de fábrica, 2x
  mais perto no condicionamento) não foram renderizados; sem isso não se sabe se "só com as travas"
  é um jeito seguro de usar esses arquivos. `ltx23cond_hw4a4cL` e `hw4a4sL` já existem.

**C, adendo da passada 4 (04:30):** re-render do int8 da Lightricks e do W4A8 só pela barra do
sampler, mesmo protocolo: **Lightricks int8 2,14 s/it, W4A8 2,26 s/it** (3 passos, 6 s cada; uma
corrida cada). Com os 2,02 / 2,12 dos nossos int8 e os 8,48 do BF16, os quatro braços quantizados
ficam dentro de 12 % um do outro e o W4A8 não é mais rápido por passo que o int8 neste tamanho — o
que ele compra é 11,66 GiB contra 20,03. Os dois re-renders deram MAE 4,10 e 7,81 contra o BF16,
iguais aos de 13/09 até a segunda casa (terceira vez que o render se repete pixel a pixel). Parede:
847 s para o int8 (carga por SMB disputando a rede com o upload do Hub) e 419 s para o W4A8.

**E — E1 CONFIRMADA; E2: a reconversão é BYTE-IDÊNTICA à de 2026-09-01 (04:35; `bench/capybara_w4a8/`).**
Ladder do Hunyuan (maçã, semente 12345, 6 passos, 480 px, 1 quadro), referência capybara BF16,
in-process na 3090 depois do conserto do dtype (passada 5):

```
braço                         divergência de latente vs capybara BF16   imagem
capybara W4A8 (reconvertido)  0,1439                                    maçã vermelha na mesa, mesma composição; um pouco mais mole, sem o cabo
capybara W4A4 (2026-09-01)    0,7072  (bench/capybara_previsao)         destruída
```

- **E1 confirmada:** coerente, mesma cena, 0,14 contra 0,71 do W4A4 no mesmo checkpoint, prompt e
  semente. O que o W4A8 perde é fino — o cabo da maçã vira dois toquinhos e a cor fica um pouco
  menos saturada — o amolecimento que a divergência favorece (ver Wan 2.2 no CLAUDE.md). Sem número
  do `hv15_w4a8` para comparar lado a lado (a ladder dele não está em `bench/`); a comparação que
  vale é com o W4A4 do mesmo arquivo.
- **E2:** sha256 `4317156bea06f76cbb2f8b40c715da063442c52c94c8fb4091e95fa7cb1b1fcc`,
  8.847.634.096 B — bate com o prefixo `4317156b…` e o tamanho registrados do arquivo apagado de
  2026-09-01 (convertido então na 3090 em 18 s; agora na 3080 Ti em 391 s, para P: por SMB). O
  conversor W4A8 com codebook é determinístico entre placas.
- **Velocidade não publicada:** a ladder deu 1,415 s/passo no BF16 e 6,557 no W4A8 numa corrida
  única, com a referência de 15,5 GiB carregada no mesmo processo antes — residência do segundo
  braço desconhecida, e o card do Hunyuan já registra uma corrida única invertendo o sinal. Fica
  como não resolvido, não como velocidade do W4A8.
- O decode in-process morreu com o `hostbuf_allocate` conhecido (motivo de existir o
  `decode_latents.py`); as duas imagens foram decodificadas à parte na 3080 Ti com o VAE certo.
- Antes disso, a passada 1 morreu em `time_in` por dtype (BF16 uniforme × modelo fp16) — segundo
  modo da armadilha da carga preguiçosa, corrigido em `quality_ladder.py` e `_dynamic_vram.py`
  (registrado no CLAUDE.md, seção do DynamicVRAM).

**D, adendo da passada 6 (04:46; `bench/ltx23/encoder_heretic_locked/`): no caminho TRAVADO os dois
W4A4 MANTÊM a cena.** Mesmo transformer, mesma semente, condicionamento dos builds com as travas do
ComfyUI de fábrica (peso de 4 bits, matemática dequantizada):

```
build (caminho)             rel-L2 cond.   MAE vs render-BF16    SSIM   log-mel  SNR     cena
W4A4 convrot, travado       0,111          16,38 [15,4-17,4]     0,786  0,111    +5,40   crepúsculo (mantida)
W4A4 smooth, travado        0,082          27,41 [26,3-28,5]     0,699  0,160    -0,03   crepúsculo (mantida)
W4A4 convrot, destravado    0,221          38,12 [35,9-40,4]     0,567  0,193    -0,33   dia (perdida)
W4A4 smooth, destravado     0,158          29,82 [29,2-30,6]     0,645  0,220    -0,24   dia (perdida)
```

- A virada de cena aperta para **entre 0,11 (convrot travado, mantida) e 0,16 (smooth destravado,
  perdida)** de rel-L2 neste prompt.
- **A ordem do render no caminho travado (convrot 16,4 < smooth 27,4) INVERTE a ordem do
  condicionamento (smooth 0,082 < convrot 0,111).** Uma semente; registrado, não explicado. A
  distância de condicionamento ordena formatos grosseiramente (W4A8 ≪ W4A4) e não ordena dois W4A4
  vizinhos — a mesma lição do erro por camada, agora no encoder.
- Decisão mantida: os pesos W4A4 não sobem. No caminho que esta árvore usa por padrão eles mudam a
  cena; no de fábrica ficam a 3–5x da distância do W4A8 por 8 % menos memória e sem ganho de
  velocidade (travado = dequantizado). Provas e sidecars sobem.

## Fechamento (05:00)

A, C, D, E, G medidos, com veredito por previsão acima. Publicado nesta rodada: Krea 2 Turbo W4A4
(gated, licença cumprida), Qwen3-VL 4B W4A8, Gemma 3 12B de fábrica W4A8 (repo novo), os dois int8
do 2.5 (no repo do 2.5), capybara W4A8 (no repo do Hunyuan), provas e sidecars dos W4A4 do heretic
(no repo do heretic); READMEs do 2.5, heretic, Hunyuan e 2.3 refeitos. Não publicado, de propósito:
os pesos W4A4 do heretic. Ferramentas corrigidas: `sampler_tempo_do_log.py` (formato HH:MM:SS),
`quality_ladder.py` + `_dynamic_vram.py` (cast ao dtype de cálculo, pulando quantizados).

O que fica aberto: (1) monkeypatch 2026-08-31 (destravar W4A8 custa 0,18 no output cru) contra
flag 2026-09-14 (0,001 no condicionamento projetado) — não reconciliado; (2) velocidade do capybara
W4A8 (corrida única sob residência desconhecida); (3) a divergência do `hv15_w4a8` não está em
`bench/` para comparar com o capybara; (4) tudo aqui é um prompt e uma semente por braço.
