# Resultado: `--enable-triton-backend` nos launchers + validação GPU da revisão (2026-09-29)

Critério: `criterio.md` (escrito antes). RTX 3090 só (`CUDA_VISIBLE_DEVICES=0`), lock por fase, porta 8199,
`--disable-pinned-memory` nos dois modos, `COMFYUI_MGPU_DISABLED=1` no servidor de medição. Dados: `off/`, `on/`
(`resultados.jsonl` por prompt_id, `contador.json` do registry do kitchen, `comfy.log`), `resultado_metricas.json`.

## C1 não quebra — passa
ON: 16/16 renders success (9 casos × frio/quente, sem Q2), 0 erro triton, 0 exceção. OFF: 16/16.

## C2 fidelidade ON × OFF — nenhuma igual; diferença explicada, não é defeito
Contador (ON): só duas ops mudam de backend: `dequantize_awq_w4a16` (9600 chamadas, eager -> triton, o objetivo) e
`apply_rope_split_half` com entrada fp32 (144 chamadas, eager -> triton; o cuda recusa fp32). As 144 são o rope do
text encoder, por isso TODOS os renders mudam (MS-SSIM ON×OFF de 0,99 no klein a 0,53 no Z-Image W4A4).
Teste direto (`rope_fp32.py`, sob lock): triton e eager fp32 têm o mesmo erro contra fp64 (2-4e-7) e diferem
entre si em 1 ulp (2-5e-7). O kernel não perde precisão; o ruído de 1 ulp no condicionamento é amplificado pela
amostragem. Visual (`folha_off_on.jpg`): mesma cena, variação de composição, sem artefato.
Consequência: com a flag, renders não são bit a bit comparáveis com referências feitas sem ela (regime casado).

## C3 regressão da revisão — passa (idêntico)
OFF × referência anterior à revisão: K1, K2, Q3, Q4, Q5 idênticos pixel a pixel (TE quantizado W4A8 pelo caminho
novo, costura do loader, lowbit pela costura). Q1 OFF idêntico à referência `w4a16_q4_1_nativo_ckstock2`
(kitchen de fábrica, eager), como esperado sem o triton.

## C4 desempenho — ganho real, abaixo do 1,5× estimado
Qwen 2.1 Q4_1 AWQ, 25 passos: OFF 0,70-0,73 it/s; ON 1,01-1,04 it/s (1,43×) = os 1,02 it/s de 27/09 com o import
direto do triton. Total quente 36,5 -> 25,6 s. Demais casos: amostragem ON dentro de ±5% do OFF.

## C5 unitários GPU — passa
`patches/tests/test_comfy_kitchen_awq.py` 8 passed (triton × eager); `test_lowbit.py` 16 ok (triton bit a bit).

## O que quebrava de verdade: os workers do multigpu-orchestrator
Nos launchers sem `COMFYUI_MGPU_DISABLED=1` (run_nvidia_gpu, 8190, 8190_flash, 8190_loopback, 8190_limpo,
fast_fp16, advanced/disable_api_nodes), o orquestrador sobe um worker por GPU com
`main.py --listen --port --cuda-device N --disable-auto-launch`, SEM `-s` e sem as flags do launcher, e manda os
prompts para ele. Ou seja: sage-attention, reserve-vram, fp16_accumulation etc. nunca valiam para a geração nesses
launchers, e `--enable-triton-backend` também não valeria. Correção: esses launchers definem
`COMFYUI_MGPU_WORKER_FLAGS` (mecanismo do próprio nó) com as flags de desempenho e `PYTHONNOUSERSITE=1`.
Prova (`confere_worker.ps1`): worker subiu com `--use-sage-attention --enable-triton-backend`, log do worker
"Enabling comfy-kitchen triton backend" e "Using sage attention", render executado no worker com sucesso.

Launchers alterados (originais em `bat_antes/`, `aplica_launchers.py`): os 7 acima + dual_component, ultra_image,
ultra_image - sem dynamic, ultra_video, void_cache_none (estes 5 já desligam o orquestrador: só a flag).
Fora: run_cpu.bat, ComfyLite.

## Limites e incidentes
- Q2 (Qwen BF16 14 GB do NAS) não rodou: é o caso que abortou o dynamic VRAM em 27/09 e prende a placa.
- LTX/vídeo não entrou na matriz; o contador mostra que rope/adaln do LTX têm implementação cuda (o triton só entra
  se o cuda recusar), mas não foi medido.
- Incidentes: (1) o primeiro contador importava o kitchen antes do main.py e derrubava a subida (corrigido: embrulho
  tardio); (2) a primeira rodada OFF foi para o worker sem flags e foi interrompida antes do Qwen/NAS;
  (3) `confere_worker.ps1` soltou o lock com o servidor de teste ainda vivo e ocioso na 3090 por ~12 min
  (taskkill do finally não derrubou; derrubado à mão, portas conferidas livres). Lição: soltar só depois de
  conferir a porta, como faz `tools/comfy_server.ps1`.
- `comfy_client.Entry.cache_hit` marcava renders rápidos (<5 s) como cache; corrigido para usar `execution_cached`
  × saídas do prompt (limiar só como fallback) + teste.
