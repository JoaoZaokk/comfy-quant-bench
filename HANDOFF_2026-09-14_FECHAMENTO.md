# Handoff LTX 2.3 — 2026-09-14 00:50 (escrito para sobreviver a uma compactação)

## O que está RODANDO agora (não relançar)

- `.scratch/fila_ltx23h.sh` → log `.scratch/fila_ltx23h.log`. Servidor normal
  (`--disable-dynamic-vram`, logs `.scratch/comfy_8190_h.{log,err}`). Faz: rodada de LoRA do 2.3
  (W4A8, 49 quadros, `--cond-from ltx23cond`, sem encoder residente): `ltx23lora_base` (191 s, OK),
  `ltx23lora_merge` (209 s, OK), `ltx23lora_bypass` (182 s, OK), `ltx23lora_seed2` (começou
  00:47:24) → `compara_av` → `bench/ltx23/lora/{contato_av.png,comparacao_av.json}`. Depois o
  controle de identidade: W4A8 249 quadros com `--cond-from` (`ltx23av_w4a8_cond`) comparado ao
  render com encoder vivo (`.scratch/ltx23av_w4a8.json`) → `bench/ltx23/cond_identity/`
  (previsão: pixel-idêntico, MAE 0,0, como no 2.5). Termina com `FILA_LTX23H_DONE`.
- `.scratch/fila_ltx23i.sh` → log `.scratch/fila_ltx23i.log`. Espera `FILA_LTX23H_DONE`, faz
  `/free`, e roda o braço BF16 do 2.3 por **GGUF** (`--gguf ltx-2.3-22b-distilled-1.1-BF16.gguf`,
  `--video-vae LTX23_video_vae_bf16.safetensors`, `--cond-from ltx23cond`, 249 quadros) →
  `.scratch/ltx23av_bf16.json` → `compara_av` 3 braços vs BF16 → `bench/ltx23/av/`. Se falhar:
  `BF16_23_FALHOU_8` e comparação vs Q6_K em `bench/ltx23/av_vs_q6k/`. Termina `FILA_LTX23I_DONE`.
- Monitores: `bz4qrqb4p` (logs das filas h/i), `b2frmyofb` (err do servidor). Parar com TaskStop
  quando as filas acabarem.

## O que foi MEDIDO nesta madrugada (tudo em processo nu, sem ComfyUI)

Probes no scratchpad da sessão (`C:\Users\joaoz\AppData\Local\Temp\claude\F--COMFY-PORTABLE\
2f537c44-.../scratchpad/`): `probe_commit_mmap.py`, `probe_safeopen_trace.py`,
`probe_double_map.py`, `probe_cow_offset.py`. Copiar para `tools/` antes de commitar (ainda NÃO
estão no repo).

- `safetensors.safe_open(framework="pt")` num arquivo de 39,13 GiB: **+40,8 GiB de commit ao
  abrir**, antes de ler tensor (duas views copy-on-write: memmap2 + `UntypedStorage.from_file
  (shared=False)`; +80,2 GiB enquanto as duas vivem). `torch.empty` do modelo: +40,7. mmap
  somente-leitura: 0. `from_file(shared=True)`: 0.
- Teto de commit 124,8 GiB (63,6 RAM + pagefile 61,2 GiB em C:, gerido pelo sistema, `?:\pagefile.sys`),
  ~70 GiB já comprometidos por outros processos (docker/WSL etc.). Quando a cobrança força o
  pagefile a crescer, a view às vezes volta com o limite aumentado e a cobrança não feita, e a
  primeira leitura dá `access violation` em `torch/storage.py:471 __getitem__` — reproduzido:
  safe_open + get_tensor em W: 3/3 mortes; mapeamento duplo à mão em C: 1/1 morte; as mesmas
  chamadas sobrevivem em outras corridas (T5-C, trace-C 2x). **Não é o SMB** (julgamento anterior,
  já corrigido nos 4 documentos: card 2.5 [re-subido ao Hub, LTX25_README_OK], CLAUDE.md,
  README_execucao, W4A4_PROGRESS parte 50).
- Cronologia das 6 mortes do braço BF16 (logs `.scratch/comfy_8190_*.err`): ltx23 (19:00,
  silenciosa) e ltx23b (19:21) = checkpoint único + DisTorch de W:, encoder vivo; ltx23c (20:55) e
  d (21:05) = transformer extraído + UNETLoaderDisTorch2 de W:, encoder vivo; e (22:16) = de C:,
  cond salvo, morreu em `nn.Linear.__init__` (torch.empty); f (22:21) = dynamic VRAM,
  `HostBuffer.read_file_slice failed`. g (22:24) = morte na rodada de LoRA ao construir o ENCODER
  (24,4 GB) com o W4A8 já carregado — mesma assinatura torch.empty.
- Saída: `tools/safetensors_to_gguf_bf16.py` (novo, não commitado) escreveu
  `C:/ComfyBench/ltx-2.3/ltx-2.3-22b-distilled-1.1-BF16.gguf` (42.035.412.384 B, 39,15 GiB, arch
  ltxv, F32 2672 / BF16 1772, template Q6_K conferido 4444/4444, F32 idênticos 32/32, `config`
  copiado). `tools/probe_gguf_bf16_equivalence.py` (novo, não commitado): 12/12 camadas
  IDÊNTICAS (bytes, dequant, saída do Linear com bias) na cuda:1 →
  `.scratch/gguf_bf16_equivalence_ltx23.json`. Loader GGUF = memmap RO (0 commit) + sem torch.empty.
- LoRA no peso (2.3 W4A8 + Product Commercial, já medido, `.scratch/lora_requant_ltx23_product.json`):
  3264 tensores → 1632 alvos, sobrevivência 0,908 (0,87–0,93), cosseno 0,046, ruído 20x,
  err 0,0731 → 0,0837 (zero) → 0,0838, P1–P3 confirmadas. Igual ao 2.5 (mesmo layout).
- **Trigger do LoRA: `srx_commercial`** (`ss_tag_frequency`), base `ltx2`, r16, 337 MB em
  `ComfyUI/models/loras/`. A rodada atual NÃO usa o trigger (mede só a perturbação de carregar,
  o mesmo furo do 2.5).

## Próximos passos, em ordem

1. Ler resultados: `bench/ltx23/lora/comparacao_av.json` + `contato_av.png` (olhar a folha);
   `bench/ltx23/cond_identity/` (esperado MAE 0,0); `bench/ltx23/av/` (BF16 GGUF vs Q6_K, W4A8,
   W4A4). Registrar s/quadro do GGUF só como informativo.
2. Teste do EFEITO do LoRA (fecha o furo): escrever ANTES em `bench/criterio_lora.md` R7 (com
   `srx_commercial` no prompt, fundido vs sem-LoRA difere MAIS que semente-vs-semente) e R8
   (fundido ≈ bypass). Precisa de condicionamento novo: o encoder de 24,4 GB não pode subir no
   servidor pelo leitor normal (2x+1x = 71 GiB de commit → loteria). Plano: `tools/ltx_encode_
   lowcommit.py` em processo nu: monkeypatch `comfy.utils.load_torch_file` por leitor memmap
   somente-leitura + `torch.frombuffer` (0 commit), chamar `LTXAVTextEncoderLoader.execute(
   text_encoder, ckpt_name, device)` (`comfy_extras/nodes_lt_audio.py:169`, usa
   `comfy.sd.load_clip(ckpt_paths=[encoder, proj_ckpt], clip_type=LTXV)`) + `CLIPTextEncode.encode`
   (`clip.encode_from_tokens_scheduled(clip.tokenize(text))`), gravar como o
   `LTXVSaveConditioning` (`conditioning_data_{i}` bf16 contíguo + `attention_mask_{i}`, metadata
   `num_conditionings/dtype/created_at`, em `models/embeddings/<nome>.safetensors`). Rodar com
   `CUDA_VISIBLE_DEVICES=1` (3080 Ti, carga parcial) ou CPU. Autoteste: recodificar o prompt do farol
   e comparar bit a bit com `models/embeddings/ltx23cond_pos.safetensors`. Depois 4 renders de 49
   quadros (sem LoRA, fundido, bypass, sem LoRA outra semente) com prompt tipo "srx_commercial, a
   sleek matte black wireless headphone rotating slowly on a white pedestal, soft studio lighting,
   clean background, product commercial" → `compara_av` → `bench/ltx23/lora_trigger/`.
3. Estatística da loteria (só DEPOIS que a GPU acabar — probes de commit pesado podem matar o
   servidor): T10 em C: ×N e etapa A ×N, contar mortes/sobrevivências.
4. Documentar: card `bench/hf/ltx23-22b-w4a8/README.md` (trocar `TBD_MEASUREMENT_TABLE` por tabela
   vídeo+áudio 4 braços + seção LoRA; seção do mecanismo JÁ está lá); `bench/criterio_ltx23.md`
   (P1–P6 vereditos; P6 refutada: BF16 não rodou por DisTorch, rodou por GGUF); `bench/criterio_lora.md`
   (linha 2.3 + R7/R8); CLAUDE.md (parágrafo 2.3 + trigger); W4A4_PROGRESS parte 51; README.md
   (GitHub, trocar o "In progress" da linha 81 e acrescentar a linha do repo 2.3 na tabela).
5. `git add -f bench/ltx23/{av,lora,cond_identity,lora_trigger}/{contato_av.png,comparacao_av.json}`
   + `git add tools/safetensors_to_gguf_bf16.py tools/probe_gguf_bf16_equivalence.py` + probes
   copiados; commit; `git push origin master:main`.
6. Hub: `.scratch/sobe_ltx23.py` (cria `JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot`; exige
   todas as provas, inclusive `ComfyUI/output/ltx23av_bf16_av_00001_.mp4` e `_audio_00001.flac`).
   Peso em `P:\ComfyBench\checkpoints\ltx-2.3-22b-distilled-1.1_w4a8.safetensors` (+ .quant.json).
7. Parar monitores (`bz4qrqb4p`, `b2frmyofb`).

## Regras que continuam valendo

Só `.\python_embeded\python.exe -s`; nunca mexer no pagefile (é do dono; registrar, não mudar);
nunca apagar original; o GGUF BF16 em C: é derivado meu (pode ficar como referência carregável).

## ATUALIZAÇÃO 01:05 — o condicionamento salvo pelo LTXVSaveConditioning estava QUEBRADO

- Controle de identidade (fila h): W4A8 249 quadros com `--cond-from ltx23cond` vs encoder vivo =
  **MAE 75,9 / SSIM 0,19 / log-mel 1,05** (ruído). Causa lida no código: o encoder do 2.3 devolve
  `extra={"unprocessed_ltxav_embeds": True}` (`comfy/text_encoders/lt.py:201-204`); o modelo só aplica
  caption_projection + connectors com a chave (`comfy/model_base.py:1185` → `av_model.py:583`); o
  LTXVSaveConditioning/LTXVLoadConditioning (ComfyUI-LTXVideo) perdem a chave → contexto de 6144 canais
  entra cru. TODA a rodada de LoRA da fila h e o BF16 da fila i (interrompido) eram lixo. Resultados
  movidos para `bench/ltx23/{cond_identity,lora,lora_par}_ltxv_saver/`; `criterio_lora.md` marcado.
- Correção: (1) `tools/ltx_encode_lowcommit.py` agora grava float32 + TODAS as opções
  (`opt_{i}_{k}` tensores, `__metadata__["options_{i}"]` JSON); (2) novo nó `VoidLoadConditioningFull`
  em `custom_nodes/comfy-void-stage-tools/__init__.py` (rastreado; stub em ComfyUI/custom_nodes/)
  que restaura tudo e RECUSA arquivo sem options; (3) `tools/ltx_video.py --cond-from` usa esse nó.
- Encoder na 3080 Ti vs no servidor (3090): rel L2 1,0e-3, 87% dos elementos bit-iguais em bf16,
  max abs 1,0 numa faixa [-148, 294] — `.scratch/encode_lowcommit_chk.json`. Não é bit-idêntico
  (placa/carga parcial diferente); registrar, não esconder.
- RODANDO: encodes `.scratch/encode_lowcommit2.log` (ltx23condf = farol formato completo;
  ltx23cond_srx = comercial com gatilho) → depois `.scratch/fila_ltx23j.sh` (log `fila_ltx23j.log`,
  servidor `comfy_8190_j.*`): (a) identidade W4A8 249 condf vs vivo → `bench/ltx23/cond_identity`;
  (b) LoRA farol x4 → `bench/ltx23/lora` + `lora_par`; (c) LoRA gatilho x4 → `bench/ltx23/lora_trigger`
  + `lora_trigger_par`; (d) Q6_K e W4A4 249 condf; (e) BF16 GGUF 249 condf → `bench/ltx23/av` (4 braços,
  MESMO condicionamento) ou `av_vs_q6k`. Marcador `FILA_LTX23J_DONE`. Monitores: bx8r5z3z3, bl9m7nibr.
- Velocidade: s/passo depende do que mais está residente (W4A8 79,7 s/passo com o encoder de 22,7 GB
  parcialmente na placa vs 26,4 sem encoder). Na fila j nenhum braço tem encoder → coluna de
  velocidade comparável entre os 4 braços; a do BF16 por GGUF é streaming (informativa).
- Os JSONs `.scratch/ltx23av_{w4a8,w4a4,q6k}.json` (encoder vivo) continuam válidos como renders do
  farol; os `_condf` são os que entram na comparação final.


## ATUALIZACAO 01:17 (depois da compactacao)

- Encodes em formato completo PRONTOS: `ltx23condf_{pos,neg}` (farol) e `ltx23cond_srx_{pos,neg}`
  (comercial com gatilho), ambos com `options_0 = {"pooled_output": null, "unprocessed_ltxav_embeds": true}`.
  `--compare ltx23cond`: mesma forma, dif_max 1 (pos) / 0,25 (neg) contra o salvo em bf16.
- Servidor j subiu (pid no log `.scratch/comfy_8190_j.*`), `VoidLoadConditioningFull` registrado e
  listando os oito arquivos. Traceback no arranque = `comfyui_tinyterranodes` (config.ini deles), alheio.
- **(a) IDENTIDADE PASSOU**: W4A8 249 quadros, condf vs encoder vivo -> MAE 1,74 [1,46-2,08], PSNR
  35,75, SSIM 0,977; audio log-mel 0,074, SNR 4,46 dB, lag 0 ms, RMS -24,0 vs -23,9 dBFS, silencio 0%.
  Contra 75,9 / 0,19 / 1,05 do saver quebrado. 26,0 s/passo (208 s total) sem encoder no processo.
  `bench/ltx23/cond_identity/{contato_av.png,comparacao_av.json}`.
- Fila j esta em (b) LoRA farol x4 desde 01:13; depois (c) gatilho x4, (d) Q6_K + W4A4 249, (e) BF16
  GGUF 249, comparacao em `bench/ltx23/av`. Marcador final `FILA_LTX23J_DONE`.
- Feito nesta janela: `W4A4_PROGRESS.md` parte 51 (mecanismo, GGUF, armadilha do condicionamento,
  residencia 3x, LoRA no peso do 2.3; falta a subsecao de resultados); `bench/criterio_ltx23.md`
  secao Vereditos com P6 REFUTADA (P1-P5 pendentes); `tools/ltx_video.py` grava `encoder: null`
  quando `--cond-from`; git add das 7 ferramentas novas + node + `ltx_video.py` + evidencia
  `*_ltxv_saver`. `.scratch/sobe_ltx23.py` reescrito para ler nomes de MP4/FLAC dos JSONs de corrida
  (os renders antigos do LoRA farol ocupam `_00001_`; os novos saem `_00002_`).
- Falta: tabela do card (TBD_MEASUREMENT_TABLE) + secao LoRA + Running it; criterio P1-P5 e
  criterio_lora R7-R9; CLAUDE.md paragrafo de resultados 2.3; README GitHub linha "In progress";
  parte 51 resultados; git add -f dos bench/ltx23/{cond_identity,lora,lora_par,lora_trigger,
  lora_trigger_par,av}; commit; push master:main; `.scratch/sobe_ltx23.py`.


## CORRECAO 01:22 -- o "3x de residencia" era instrumento errado

`s_por_passo` do `ltx_video.py` = parede da corrida inteira / passos (carga do modelo, encoder, sampler,
VAEs, mux). A barra tqdm do log do servidor e o instrumento por passo: W4A8 249 quadros, 8 passos em
18 s (2,30 s/it) COM encoder vivo (log `comfy_8190_ltx23.err`) e 8 passos em 18 s (2,30 s/it) com
condicionamento salvo (`comfy_8190_j.err`). Os 638 s vs 208 s eram carga do encoder por SMB + encode.
W4A4 1,60 s/it, Q6_K 5,12 s/it (`comfy_8190_ltx23b.err`). A linha "Velocidade: s/passo depende..."
acima fica como registro do erro. Corrigido em CLAUDE.md, card 2.3 (2 lugares), parte 51, card 2.5
(coluna "s/frame" virou "whole run"; sem afirmacao de velocidade entre bracos; re-subido ao Hub),
`ltx_video.py` grava `nota_tempo` no JSON. Para a tabela do card 2.3: coluna do sampler vem das barras
8/8 do `comfy_8190_j.err` por corrida (uma barra por prompt, impressa duas vezes), parede vem do JSON.


## ESTADO 01:52 -- fila j PRONTA, commit empurrado, upload em andamento

- `FILA_LTX23J_DONE` 01:46. BF16 por GGUF carregou PARCIAL (20,7 GB na placa, 19,6 descarregados),
  8 passos em 66 s (8,05 s/it), parede 155 s, sem morte. Q6_K 3,59 / 0,163; W4A8 10,39 / 0,163; W4A4
  14,45 / 0,281 (MAE / log-mel vs BF16). Som EMPATA W4A8 com Q6_K -- registrado como nao explicado.
  LoRA com gatilho: fundido 40,9 / bypass 40,6 / outra semente 57,5 / par 7,5 (R7 refutada, R8
  confirmada, R9 indecidivel). Sem gatilho valido: 22,9 / 24,0 / 81,0 / 5,1.
- Card 2.3 completo (0 TBD): tabelas geradas por `.scratch/tabela_card23.py`, secoes de
  condicionamento, LoRA, Running it, NOT covered. criterio_ltx23 P1-P6 fechados; criterio_lora
  R7-R9 fechados; CLAUDE.md (tabela 2.3 + paragrafo LoRA + correcao do instrumento); parte 51 com
  resultados; README GitHub (paragrafo + linha na tabela de repos).
- **Commit `3671cd1` empurrado para `origin main`** (e75c413..3671cd1). 7 ferramentas novas +
  `sampler_tempo_do_log.py` + node + `ltx_video.py` + 18 arquivos de evidencia em bench/ltx23.
- Upload `.scratch/sobe_ltx23.py` rodando em background (log `.scratch/sobe_ltx23.log`): README e
  LICENSE subiram; o peso de 15,9 GB esta subindo; depois sidecar, av/*, conditioning/*, lora/*.
  Sucesso = `LTX23_UPLOAD_OK` no log. Repo: JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot.
- Servidor j (pid 37204) morto por mim (ocioso). Monitores bx8r5z3z3 e bl9m7nibr parados.
- Sobrando, opcional e SO com a GPU livre: estatistica da loteria do commit (T10-C x N); a mesma
  comparacao de audio a 3 passos para testar o empate; o LoRA sobre o BF16 por GGUF.

- 01:58 `LTX23_UPLOAD_OK`: 25 provas + .gitattributes = 26 arquivos no Hub, conferidos por listagem; peso 16.651.422.654 B conferido por tamanho.

## Fechamento 2026-09-14 05:00 — "fecha todos eles e sobe pro hf os modelos e comita no gh"

FEITO. Commit `6a16e96` em `master`, empurrado para `origin/main`. Tudo abaixo esta no
`bench/criterio_fechamento_2026-09-14.md` (criterio ANTES, vereditos por previsao com hora), na
parte 52 do `W4A4_PROGRESS.md`, na secao "closing round" do CLAUDE.md e nos cards.

Contagem: 53 sidecars, 25 com peso, 44 verificados na saida, 6 nao -> os 6 medidos (A, C, D, E) mais
o empate do audio (G). Passadas: 1 (G, D3, C), 2 (encodes na 3090 + renders A/D), 4 (por-passo 2.5),
5 (ladder capybara, apos corrigir o dtype), 6 (W4A4 travados). Passada 3 morta (duplicada).

No Hub, todos com prova:
- NOVO `JoaoZaokk/Gemma-3-12B-it-W4A8-ConvRot` (fabrica; cond 0,043; render 6,75/0,894/0,087)
- `LTX-2.5-22B-distilled-W4A8-ConvRot/int8_ours/` (dois int8: com rotacao 4,19 = Lightricks 4,10; sem 8,23)
- `HunyuanVideo-1.5-720p-T2V-Quantized` (capybara W4A8 r: div 0,1439 vs 0,7072 do W4A4; byte-identico ao de 09-01)
- `Gemma-3-12B-it-Heretic-W4A8` (provas dos W4A4 + sidecars; os PESOS W4A4 subiram as 05:10 por decisao do
  dono, em `w4a4/`, rotulados no card: mudam a cena destravados, mantem travados)
- READMEs do 2.5, heretic, Hunyuan, 2.3 refeitos; Krea 2 (gated) e Qwen3-VL 4B ja tinham subido.

Achados que mudam leitura: destravar as travas do encoder custa 0,001 no W4A8 e 2x no W4A4; a
rotacao no int8 vale o mesmo que a largura do peso (2.5); os W4A4 do heretic rendem cena
coerente e ERRADA (dia por crepusculo) destravados e mantem a cena travados (16,4 / 27,4);
virada entre rel-L2 0,11 e 0,16. Empate do audio = saturacao (a 3 passos Q6_K +8,4 dB, W4A8 -1,1).

Ferramentas: `sampler_tempo_do_log.py` le HH:MM:SS; `quality_ladder.py`/`_dynamic_vram.py`
castam ao dtype de calculo (segundo modo da armadilha do lazy load; capybara BF16 x modelo fp16).

ABERTO: (1) monkeypatch 08-31 (destravar W4A8 = +0,18 no output cru) x flag 09-14 (0,001 no cond.
projetado) — escrito nos dois cards, nao reconciliado; (2) velocidade do capybara W4A8 (6,56 s/passo
numa corrida unica apos referencia de 15,5 GiB no processo — nao publicado); (3) `hv15_w4a8` sem
ladder em bench/ para por ao lado do capybara; (4) um prompt/uma semente por braco em tudo;
(5) ordem do render travado (convrot < smooth) inverte a do condicionamento — uma semente.

Estado da maquina: nenhum servidor 8190 vivo; GPU 0 livre; GPU 1 so com o cortex. Latentes do
capybara em `bench/capybara_w4a8/latents/` (nao versionados). Monitores parados. Scripts das
passadas em `.scratch/fila_gpu0_fecha{,2,3,4,5,6}.sh`, logs ao lado; uploads em
`.scratch/sobe_fecha.py` (`--item`, `--sem-readme`).

## Limpeza (05:17) — "so os originais + o melhor quant de cada modelo"

10 pesos apagados (89,79 GiB), cada um com sha256 local = LFS do Hub antes do remove
(`.scratch/limpeza_2026-09-14.log`, regra em `.scratch/limpeza_2026-09-14.py`). Ficam 14 builds
escolhidos (o menor usavel por modelo) + 3 que nao estao no Hub (LTX 2.3 W4A4, Qwen-Image 2512
W4A4, Wan 2.2 W4A4). Originais e terceiros intocados. O W4A8 do 2.3 agora so existe em
`W:\ltx-2.3\ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors` (a copia de P: caiu).
