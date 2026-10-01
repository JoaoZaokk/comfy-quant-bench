# Resultado — correção do harness (achados 1-10 + movimento estrutural)

Critério escrito antes: `criterio_harness.md`. GPU não usada, lock real não tocado
(`F:/GPU_BENCH.lock` ausente antes e depois), nenhum ComfyUI real subido. Todo Python com
`CUDA_VISIBLE_DEVICES=-1`, `python_embeded -s`. Servidores HTTP falsos (stdlib) em porta alta.

## Testes (medidos nesta sessão)

| teste | antes | depois |
|---|---|---|
| test_comfy_run_workflow | 36 PASS rc0 | 36 PASS rc0 |
| test_timing | 41 PASS rc0 | 41 PASS rc0 |
| test_ltx_studio | 71 PASS rc0 | 71 PASS rc0 |
| test_avaliar_backend | rc0 | rc0 |
| test_gpu_lock | rc0 | rc0 |
| **test_comfy_client** (novo) | — | 14 PASS rc0 |
| **test_bench_server_lock** (novo) | — | 6 PASS rc0 |
| **test_avaliar_despacho_lock** (novo) | — | 7 PASS rc0 |
| **teste_comfy_server.ps1** (novo, aqui em .scratch; mocks do lock) | — | 10 PASS em pwsh 7 e em Windows PowerShell 5.1 |

Logs: `antes_test_*.log`, `depois_test_*.log`, `depois_teste_comfy_server.log`.

## Equivalência medida (antes × depois, mesmos dados)
- `metricas_imagem.py --dir` em 5 PNGs sintéticos: saída **byte-idêntica** (diff vazio).
- `cruza_runtimes.py` (versão HEAD × nova) no mesmo par: saída idêntica.
- `monta_grade.py` layout antigo (HEAD × novo): PNG idêntico (`cmp`).
- `metricas_bateria.py` refeito sobre a bateria real `pesos_so` (logs + imagens existentes):
  **0 diferenças** contra `fase_pesos_so_metricas.json` salvo (todas as chaves antigas).
- `metricas_lowbit.py` refeito nas 4 fases (principal, residente, offload, gpu1): **0 diferenças**
  contra os `metricas_*.json` salvos.

## Defeito novo achado e medido (não estava na revisão)
`comfy_run_workflow.run_and_wait` **nunca consultava os workers do MultiGPU** quando `--server`
vinha sem esquema (o padrão, `127.0.0.1:8190`): `bases_irmas` devolvia `http://127.0.0.1:N` e
`_request` prefixava outro `http://` → `http://http://…`, exceção engolida pelo poll. Medido com
servidor falso: o cliente HEAD estourou `PollTimeout` com o resultado pronto no "worker"; o novo
acha (`test_worker_multigpu_e_consultado_de_fato`). Corrigido por `normaliza_base`.

## Por achado
1. **bench_server** — `_lock()` usa `gpu_lock.read_state(LOCK_PATH)`/`describe` (antes
   `F:/COMFY_PORTABLE/GPU_BENCH.lock` + chave `owner`: sempre "livre"). Conversão real toma
   `GpuLock("bench_server:<sub>")` por criação exclusiva antes do subprocesso e solta no `finally`
   depois de `proc.wait()`. Ocupado → recusa sem criar trabalho (teste).
2. **Polling sem prazo** — `run_and_wait` exige timeout > 0 (`ValueError` senão). roda_lowbit,
   roda_eros_2gpu, ltx_video, ltx25_queue, qwen_edit_test passam por ele. roda_eros_2gpu: exit
   0/1/5 (antes 0 em execução falha); `nvidia-smi` com timeout de 20 s.
3. **PowerShell** — `tools/comfy_server.ps1`: `Invoke-ComfyUnderLock` / `Invoke-ComfyServer` /
   `Complete-GpuLock`. Recusa porta já ocupada (não mata o ocupante), derruba com
   `ErrorActionPreference='Continue'` no escopo da função, `Release-GpuLock` só com a porta caída
   (ou servidor nunca subido); porta de pé → lock MANTIDO com aviso. Protocolo de gpu_lock.ps1
   inalterado. roda_bateria, roda_qwen21, roda_diag reescritos sobre ele (parse OK; **não
   executados** — subiriam ComfyUI). roda_lowbit.ps1 já tinha a correção; não mexido.
4. **Cliente único** — `tools/comfy_client.py` (só stdlib; teste confere `sys.modules` num processo
   limpo). `comfy_run_workflow.py` reexporta os mesmos nomes (871 → 632 linhas) e ganhou
   `--api-prompt` / `--lista` / `--saida` (JSONL). `comfy_client.py` também tem CLI própria para a
   VM. `portas_worker` (cópia em ltx_video e qwen_edit_test) removido: um método de descoberta.
5. **node_errors num 200** — `Comfy.submit` levanta `PromptRefused` (subclasse de `ValueError`, então
   o `return 2` de `main()` continua valendo).
6. **Tempo por posição** — metricas_bateria: fonte `*.jsonl` (junta por grafo, tempo =
   server_side_s, imagem = `files`) ou `log:ordem` que **recusa** contagem divergente (antes só
   avisava); `rsplit(":")` para log com letra de drive. metricas_lowbit: recusa se execuções sem
   erro ≠ "Prompt executed" do log. Registro JSONL: grafo, prompt_id, status, erro, server_side_s,
   wall, cache_hit, files, onde, placa/placas, args, comfyui_version, pytorch_version.
7. **`_00001_` fixo** — `metricas_imagem.imagem_unica`: várias execuções com pixels iguais → a de
   maior contador; diferentes → recusa. Usado em metricas_bateria, metricas_lowbit, folha_contato,
   monta_grade `--layout comfy`. Caso real existente (`nosso_int8/p0_s42` _00001_ e _00002_,
   sha256 iguais) passa.
8. **Métricas** — `metricas_imagem`: `medir`, `identica`, `sha_pixels`, `cria_lpips(nunca|so_cache|
   baixar)`, `imagem_unica`; `main` usa as mesmas. cruza_runtimes mantém "nunca baixa"
   (`so_cache`).
9. **Commit/bootstrap** — `_ram_guard`/`_conversion` não são meus (agente de conversão). Criei
   `tools/_comfy_boot.py` (`separa_flags`, `boot(flags, dinamico=)`), testado em CPU (flag aplicada,
   argv restaurado, torch não importado, segunda chamada reporta `ja_iniciado`). **Nenhum dos ~16
   chamadores migrado** — não são arquivos meus e cada um é ferramenta medida.
10. **Arquivo / doc** — `--warm` inexistente corrigido na doc. avaliar_despacho: recusa (exit 3) sem
    lock vivo (pid vivo + hb ≤ 55 s; `--dono` opcional exige o dono) quando algum checkpoint iria à
    GPU; grava `controles` (lock, device, CVD, torch/comfy-kitchen/triton-windows por metadata,
    HEAD e sujeira do ComfyUI, python) em cada laudo. avaliar.py: sem decomposição (não há corte
    óbvio e seguro); 999 linhas.

## Item extra: marcas do patch do text encoder em avaliar.py
Lidos os arquivos atuais: `def has_quantized_matmul` (ops.py:1732), `def use_quantized_matmul`
(ops.py:1737), `has_quantized_matmul(` (sd.py:303), `use_quantized_matmul(` (sd.py:328,497);
`sd1_clip.py` idêntico ao v0.37.4 (`git diff --quiet v0.37.4`). `_travas_do_encoder_soltas` agora
procura essas quatro marcas (incluí `use_quantized_matmul`, que é a trava 2). Medido: a versão HEAD
responde `False` (acusaria "travas presas" por engano) nesta árvore; a nova responde `True`.

## Probes arquivadas (`git mv` para tools/_arquivo/, todas rastreadas; nada apagado)
probe_bonsai_bits_por_peso, probe_bonsai_pack_gemlite, probe_conjunto_denso,
probe_encoder_no_difusor, probe_erro_por_tipo_de_camada, probe_pipeline_2x2,
probe_piso_de_ruido_do_prompt, probe_teto_groupsize, probe_vram_holder, probe_why_no_backoff.
Zero referências em tools/, docs/, .agent-reference/, AGENTS.md, W4A4_PROGRESS.md,
W4A4_HANDOFF.md, .scratch/ (grep -rIw). **Atenção:** todas são citadas FORA desse conjunto —
bench/bonsai_packs_resultado_2026-09-22.md, bench/criterio_bonsai_packs_2026-09-22.md,
bench/replicabilidade_fora_do_flux_2026-09-22.md, bench/varredura_ecossistema_2026-09-01.md,
bench/criterio_teto_zimage.md, HANDOFF_2026-08-30.md, HANDOFF_2026-09-01*.md. Esses caminhos
agora apontam para tools/_arquivo/. Reverter: `git mv tools/_arquivo/<x>.py tools/<x>.py`.
As renomeações estão **staged** (efeito do `git mv`); nada commitado.

## Não feito, e por quê
- `.scratch/qwen21_2026-09-26/encadeia*.ps1` (espera por sentinela `FIM` sem prazo): fora da
  posse. Os `encadeia*.ps1` do lowbit não têm espera por sentinela — nada a corrigir.
- `_ram_guard`/`_conversion` (commit): de outro agente.
- Migrar os ~16 bootstraps para `_comfy_boot`: arquivos não meus; exige rodar cada ferramenta.
- `.scratch/arc_2026-09-28/`: do coordenador (4 cópias do cliente sem prazo continuam lá).
- `build_bench_html.py`: nenhuma mudança necessária (a página lê `livre`/`dono`, mantidos).
- Docs em `.agent-reference/` que citem `portas_worker`/comandos: não são meus.

## Validações que exigem GPU / servidor real (não rodadas)
1. `comfy_run_workflow.py --api-prompt <grafo> --server 127.0.0.1:8190 --saida r.jsonl` com
   ComfyUI real: medir que `server_side_s` bate com "Prompt executed in" do log (diferença
   < 0,1 s) e que um reenvio idêntico sai com exit 5.
2. MultiGPU real (sem `COMFYUI_MGPU_DISABLED`): `ltx25_queue.py` com DisTorch2 — o registro deve
   dizer `resultado veio de http://127.0.0.1:<worker>`; antes estourava prazo.
3. Um grafo com uma saída inválida (nome de VAE inexistente num ramo secundário): exit 2/1 com
   `PromptRefused`, nenhum arquivo gravado para o ramo válido tido como sucesso.
4. `roda_bateria.ps1` (sob janela de GPU autorizada): conferir que `Release-GpuLock` só aparece
   depois de `Get-NetTCPConnection -LocalPort 8190 -State Listen` vazio, e que
   `bateria/resultados_<ordem>.jsonl` tem uma linha por grafo; `metricas_bateria.py` com o JSONL.
5. `bench_server.py --permitir-escrita`: conversão real com lock livre — o lock deve nomear
   `bench_server:<sub>` e o pid do servidor durante toda a conversão, e sumir ao fim.
6. `avaliar_despacho.py` sob `Assert-GpuLock`: laudo com bloco `controles` preenchido.
