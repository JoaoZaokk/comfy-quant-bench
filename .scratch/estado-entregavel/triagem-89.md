# Triagem barata dos arquivos rastreados

Nome do arquivo mantido (`triagem-89.md`) porque é o caminho que o ticket 02 pediu para
criar. **O número no nome está errado.** `git -C F:\COMFY_PORTABLE ls-files | wc -l` deu **93**
em 2026-08-21, não 88 nem 89 — contado agora, não copiado do título do ticket 02 (que também
estava errado e foi corrigido nele). Ver detalhamento abaixo.

```
$ git -C F:\COMFY_PORTABLE ls-files | wc -l
93
```

Repartição:

| grupo | contagem |
|---|---|
| raiz (docs + scripts soltos) | 14 |
| `calib/` (JSON de calibração, dados gerados) | 8 |
| `custom_nodes/comfy-quant-preflight/` | 3 |
| `tools/` (62 `.py` + 3 fixtures JSON + `benchmark_prompt.txt` + 2 `.ps1`) | 68 |
| **total** | **93** |

## Método

Barato de propósito: cabeçalho/docstring, imports, `def`s e sítios de chamada (`git grep -w
<nome_sem_extensão>` no repo inteiro) para cada arquivo. Onde `AUDITORIA_2026-08-18.md` já tinha
achado algo com linha exata, a classificação usou aquilo — mas **reconferida contra o código de
hoje**, porque a seção 6 daquele documento já registra uma leva grande de correções aplicadas no
mesmo dia. Tratar o achado do auditor como still-open sem reconferir teria classificado como
"suspeito" um bocado de arquivo que já foi consertado. Cada linha abaixo diz **leitura** ou
**leitura + comando** conforme o que realmente sustenta a classificação.

Legenda:
- **saudável** — nada a fazer aqui, agora.
- **suspeito** — motivo concreto, ticket próprio na tabela ao final.
- **morto** / **morto?** — sem sítio de chamada achado; `morto?` quando a dúvida (script chamado só
  na linha de comando, por exemplo) não pôde ser eliminada barato.

## Raiz (14)

| arquivo | classificação | motivo |
|---|---|---|
| `.gitattributes` | saudável | config, não código |
| `.gitignore` | saudável | config, não código |
| `AGENTS.md` | saudável | doc de política |
| `AUDITORIA_2026-08-18.md` | saudável | doc; é a própria fonte usada nesta triagem. Duas conclusões dela (item 19 `core_patch.py`, seção 5 baixa-confiança) seguem sem correção — refletido nos arquivos que elas apontam, não neste documento |
| `CLAUDE.md` | saudável | doc de instruções do projeto |
| `CORTIQ_LTX25_HANDOFF.md` | saudável | doc |
| `FBCACHE_FINDINGS.md` | saudável | doc; usado para cruzar cobertura dos `fbcache_*` |
| `GARIMPO_MODELOS.md` | saudável | doc |
| `UPSTREAM_REPORT_dtype_widget.md` | saudável | doc |
| `UPSTREAM_REPORT_w4a8_capture.md` | saudável | doc |
| `W4A4_HANDOFF.md` | saudável | doc |
| `W4A4_PROGRESS.md` | saudável | doc |
| `_check_accel.py` | saudável | sem sítio de chamada em outro arquivo, mas é comando documentado no `CLAUDE.md` ("Verify the attention accelerators..."). Chamado só pelo usuário na linha de comando não é morto |
| `inspect_quant.py` | saudável | 3 achados do auditor (`AUDITORIA:165,253`: sem validação de `header_size`, agulha `offset` com falso positivo, sem guarda de `argv`) — **todos corrigidos**, reconferido por leitura: `header_size <= 2 or > limit` (linha 19), comentário removendo `offset` da lista de agulhas (linha 34), `len(sys.argv) != 2` (linha 61) |

## `calib/` (8)

Todos dados gerados por `tools/calibrate_activations.py` / `tools/quant_mixed.py --save-analysis`,
não código. Nada aqui é "chamado" no sentido de import — são lidos por `--calibration` /
`--analysis`. Frescor de conteúdo (se ainda batem com os checkpoints atuais) é outro escopo, não
esta triagem.

| arquivo | classificação | motivo |
|---|---|---|
| `calib/xfer_beyond-reality-zimage-v2_native.analysis.json` | saudável | dado gerado |
| `calib/xfer_capybara.analysis.json` | saudável | dado gerado |
| `calib/xfer_hunyuan15.analysis.json` | saudável | dado gerado |
| `calib/xfer_recovered.analysis.json` | saudável | dado gerado |
| `calib/xfer_wan21_vace.analysis.json` | saudável | dado gerado |
| `calib/xfer_z_image_de_turbo_v1_bf16.analysis.json` | saudável | dado gerado |
| `calib/xfer_z_image_turbo_bf16.analysis.json` | saudável | dado gerado |
| `calib/zimage_v2_native.analysis.json` | saudável | dado gerado |

## `custom_nodes/comfy-quant-preflight/` (3)

| arquivo | classificação | motivo |
|---|---|---|
| `__init__.py` | saudável | carregado pelo boot do ComfyUI (`init_external_custom_nodes`); ver ticket 16 (resolved) para o contexto de "artefatos nossos dentro do ComfyUI" |
| `checks.py` | saudável | importado por `__init__.py`; cada check carrega a proveniência no próprio docstring (dois WARN dizem "not confirmed by execution", por desenho) |
| `test_checks.py` | saudável | roda sozinho (`python_embeded\python.exe -s ...test_checks.py`), CPU, sem GPU |

## `tools/` (68)

### Já cobertos por ticket existente — apontar, não duplicar

| arquivo | ticket(s) | nota |
|---|---|---|
| `comfy_run_workflow.py` | 03, 04, 06 | sendo editado agora por outro agente |
| `gpu_lock.ps1` | 01 | janela de GPU entre sessões |
| `quant_audit.py` | 12 | sendo lido agora por outro agente; o achado "não corrigido" do auditor (INT4 empacotado contado como I8, `AUDITORIA:95,343,346`) é do escopo do inventário — cabe em 12, não aqui |
| `gpu_lock_beat.ps1` | 01 | heartbeat do mesmo lock file; mesmo subsistema do 01, sem achado próprio |
| `test_comfy_run_workflow.py` | 03, 04, 05, 06 | suíte de teste do `comfy_run_workflow.py` (o próprio docstring diz "ticket 05") |
| `fixtures/golden_no_seed.json` | 03, 04, 05, 06 | fixture de `test_comfy_run_workflow.py` |
| `fixtures/golden_seed_12345.json` | 03, 04, 05, 06 | fixture de `test_comfy_run_workflow.py` |
| `fixtures/object_info_ltx25.json` | 03, 04, 05, 06 | fixture de `comfy_run_workflow.py --object-info-file` |

### Saudável (correção já aplicada 2026-08-18, reconferida hoje por leitura do código atual)

Cada linha cita o achado do auditor e onde a correção aparece agora. Isto é **leitura**, não
execução — nenhum destes rodou hoje.

| arquivo | achado original | reconferido |
|---|---|---|
| `_bench_guard.py` | módulo novo, substitui 2 cópias divergentes do guard NVML (`AUDITORIA:231-240`) | usado por 8 arquivos (`grep -rl "from _bench_guard import"`); falha fechado sem NVML |
| `_dynamic_vram.py` | — | sem achado; usado por `quality_ladder.py`, documentado no `CLAUDE.md` |
| `_ram_guard.py` | módulo novo, substitui `largest*3+2GiB` copiado (`AUDITORIA:236`) | usado por `quant_int8.py`, `quant_mixed.py`, `quant_w4a8.py` |
| `attn_bench.py` | comparação Sage/Flash contra SDPA que pode ser o próprio Flash (`AUDITORIA:209`) | seção 5 (baixa confiança) do próprio auditor: mecanismo certo, mas hipótese sem contraprova — não é bug confirmado |
| `attn_dtype_ab.py` | sem checagem de finitude; `.contiguous()` dentro do bloco cronometrado (`AUDITORIA:249,section 6`) | corrigido: usa `BenchGuard`, guarda de finitude presente |
| `benchmark_prompt.txt` | — | dado, não código |
| `calibrate_activations.py` | docstring prometia "zero sync"; media 98% de sync real (`AUDITORIA:267`) | docstring corrigido (mede o sync real, não promete zero) |
| `check_w4a8.py` | `main()` retornava 0 com todas as células `nan` (`AUDITORIA:79`) | linha 136: `return 1` com comentário "An all-nan table used to exit 0" |
| `compile_w4a4_fix_poc.py` | — | a reprodução que funciona (`torch.compile works, max_abs_diff 0.000000`, `W4A4_PROGRESS.md:357`); sem achado |
| `compile_w4a4_probe.py` | pergunta já respondida, resultado contradiz o registrado (`AUDITORIA:152,271`) | mantido de propósito, não apagado — apagar é irreversível, decisão registrada no próprio cabeçalho ("SUPERSEDED -- 2026-08-18. Do not trust this file's verdict") e em `AUDITORIA:271`. Não é `morto`: é superado com decisão escrita |
| `compile_w4a4_probe2.py` | mesma pergunta, mesmo motivo (`AUDITORIA:152,166`) | mesma decisão: mantido, cabeçalho próprio avisa |
| `dispatch_census.py` | — | o próprio arquivo já documenta a lacuna ("mecanismo provado por leitura, não confirmado que algo transponha neste checkout") — é o padrão de disciplina do projeto, não um achado novo |
| `fbcache_audit.py` | — | referenciado em `FBCACHE_FINDINGS.md:311` |
| `fbcache_branch_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:307` |
| `fbcache_catorder_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:305` |
| `fbcache_clone_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:309`, prova do bug 2 |
| `fbcache_kwargs_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:308` |
| `fbcache_probe.py` | — | referenciado em `FBCACHE_FINDINGS.md:311` |
| `fbcache_routing_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:304` |
| `fbcache_tuple_test.py` | — | tabela de cobertura em `FBCACHE_FINDINGS.md:306` |
| `fetch_ltx25.py` | — | sem achado; CLI standalone documentado (`python tools/fetch_ltx25.py`), chamado pelo usuário |
| `fetch_minimax_h3.py` | — | sem achado; CLI standalone documentado |
| `graph_capture_probe.py` | — | sem achado próprio; cita números de `w4a4_breakdown.py` que já são caveat no `CLAUDE.md` |
| `gpu_lock.py` | `__exit__` fazia `unlink()` sem checar pid; docstring citava `m_crossover` que nunca importava o módulo (`AUDITORIA:107,240`) | linha 76: checa `pid` antes de `unlink()`; `m_crossover` agora usa `BenchGuard`, que reexporta `GpuLock` (linha 29 de `_bench_guard.py`) — a cadeia de chamada existe, mesmo que indireta |
| `inspect_diffusion_arch.py` | — | sem achado |
| `ltx25_queue.py` | — | sem achado; CLI standalone |
| `m_crossover.py` | guard NVML fail-open fora do `try`; falha de M só impressa em `M=1` (`AUDITORIA:177,247`) | corrigido: usa `BenchGuard` (linha 240), reporta falha em todo M |
| `model_audit.py` | `PROJECT_OUTPUT` não reconhecia `_mixed`/`_native` (`AUDITORIA:168,262`) | regex atual inclui `mixed|native` (linha ~52-54) |
| `precheck_workflows.py` | `widgets_values` em dict iterava as chaves (`AUDITORIA:260`) | linha 58: `widgets.values() if isinstance(widgets, dict) else widgets`, com comentário citando os 3 nós VHS_* reais |
| `predict_promotion.py` | — | sem achado |
| `profile_transfer.py` | — | sem achado |
| `quality_battery.py` | 2 predicados reprovavam resposta correta, confirmado no interpretador (`AUDITORIA:330`); `kernel='native'` sem `kernel_impls` (item 18) | predicados reescritos (linhas 41-53, comentário explica o antes/depois); **reconfirmado agora**: `python_embeded\python.exe -s -c` deu `True`/`True` para os dois casos que antes davam `False`. `kernel_impls` agora sai no relatório (linha 181) |
| `quality_ladder.py` | — | sem achado |
| `quant_mixed.py` | preflight sem kwargs pulava validação; `--uncalibrated fail` ignorado em `--analysis`; `--budget` podia rebaixar camada nunca medida; `--analysis` sem checagem de coerência (`AUDITORIA:291-304`, seção "Grave") | todos os 4 confirmados corrigidos por leitura: `--uncalibrated=fail` checado fora do `if analysis is None` (linha 446-450), `--budget` exclui não-medidas do ranking (linhas 464-482), `--analysis` valida `source`/`shape`/`convrot_groupsize` (linhas 295-349) |
| `quant_w4a4.py` | `EXCLUSIONS` código morto; `--auto-detect` no-op; faltava guarda `.comfy_quant` inline (`AUDITORIA:264,276,280`) | os 3 confirmados corrigidos: `EXCLUSIONS` aplicado (linha 161), `--auto-detect` lido e avisa sobreposição (linhas 379-382), guarda inline presente (linhas 369-378) |
| `quant_w4a4_smooth.py` | sem preflight nenhum (`AUDITORIA:311`) | corrigido: importa `normal_comfy_backend` de `quant_w4a4` (linha 182), guarda `.comfy_quant` inline presente (linha 151-155) |
| `svdquant_probe.py` | — | sem achado; CLI standalone, documentado |
| `swap_to_nunchaku.py` | — | sem achado direto; a interação com `precheck_workflows.py` (dict de widgets) já está corrigida do lado do `precheck_workflows.py` |
| `synthetic_vs_real.py` | — | sem achado |
| `to_native.py` | — | sem achado; comportamento (bit-idêntico) checado contra `convert_diffusers_mmdit` conforme `CLAUDE.md` |
| `verify_w4a4.py` | `--kernel-smoke` não tinha caminho de falha; checagem de `weight_scale` só rodava com `--source`; nada detectava tensor extra na saída (`AUDITORIA:314,286`) | os 3 confirmados corrigidos: `return 1` após smoke (linhas 253-298), checagem de linhas movida para fora do `if source_header` (linha 121-124, comentário explica), detecção de tensor extra presente (linhas 149-156) |
| `w4a4_breakdown.py` | guard NVML duplicado, mas dentro do `try` (menos grave); número "host us" com ressalva | usa `BenchGuard` agora (linha 35); a ressalva do número já está no `CLAUDE.md` |
| `w4a8_fallback_sweep.py` | — | usa `BenchGuard` (linha 182), sem achado próprio |
| `weight_balance.py` | citado como a referência correta (mede crest pré e pós-rotação separadamente) | é o exemplo bom que `plot_weight_balance.py`/`activation_balance.py` deveriam ter seguido |
| `weight_dtype_probe.py` | — | sem sítio de chamada externo, mas CLI standalone com `argparse` + `if __name__ == "__main__"` — chamado só pelo usuário não é morto |

### Suspeito — graduado para ticket

| arquivo | motivo concreto | ticket |
|---|---|---|
| `tools/activation_balance.py` | crest medido em `x` (linha 85, pré-rotação), não em `rotated` (calculado na linha 83 mas não usado no crest) — mesmo defeito que `AUDITORIA:94` (seção 2.4) apontou e que segue sem correção | 18 |
| `tools/plot_weight_balance.py` | mesma classe: crest calculado sobre `w` (linha 94), enquanto `rotated` existe (linha 91) e é usado só para a régua de erro, não para o crest | 18 |
| `tools/core_patch.py` | `command_revert` (linha 144-157) sobrescreve o arquivo atual com `shutil.copy2` sem checar o hash do que está instalado agora — a checagem pós-restauração (linha 154-156) é tautológica, compara o restaurado com o próprio backup. `AUDITORIA:36` (item 19), não corrigido | 19 |
| `tools/gemma_chat.py` | `arg_names` do probe usa `"qweight"` (linha 105) onde a assinatura real de `convrot_w4a4_linear` é `qdata` (`tensor/w4a8_int8.py:93-96` segundo `AUDITORIA:174`) — efeito real não confirmado (pode ser só rótulo errado no dict), mas é leitura de assinatura divergente, não medição | 20 |
| `tools/diffusion_smoke.py` | conta chamadas ao despachante `convrot_w4a4_linear`, não ao backend que de fato rodou — `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1` muda o caminho sem mudar o nome do impl contado (`AUDITORIA:35`, item 18), sem `kernel_impls` no relatório | 20 |
| `tools/stage_probe.py` | mesmo defeito de `diffusion_smoke.py` (`AUDITORIA:35`, item 18) | 20 |
| `tools/convrot_ops_probe.py` | mesmo defeito (`AUDITORIA:35`, item 18) — rótulos já foram corrigidos, mas a contagem despachante-vs-backend não | 20 |
| `tools/te_smoke.py` | 6ª cópia de `instrument()` (`AUDITORIA:174` — o auditor citou 5, mas `te_smoke.py:65` também define `instrument()`); parte do mesmo cluster de duplicação que `_bench_guard.py`/`_ram_guard.py` já resolveram para NVML/RAM | 20 |
| `tools/nunchaku_compare.py` | `self.peak = max(self.peak, self._used())` (linha 89/137) só sobe — memória liberada durante o run nunca é descontada (`AUDITORIA:88`, seção 5, não corrigido); `--attention sage` (linha ~446) não verifica disponibilidade como `sparge`/`flash` fazem | 21 |
| `tools/svdq_to_bf16.py` | o gate de NaN (achado grave, `AUDITORIA:29`) **já está corrigido** (`not (zero_leak <= 1e-3)` + `torch.isfinite`, linhas 376-379). Mas 3 achados de baixa confiança da seção 5 seguem sem fechar: guards de shape do `split_fused` (linha 267, cenários construídos não observados), bias fundido mantém nome antigo no passthrough (linha 265), dtype BF16 hardcoded para norms (linha 252). Nenhum morde nos arquivos desta máquina hoje, mas nenhum foi fechado | 22 |
| `tools/test_svdq_verify.py` | suíte não chama `recover_weight` (a função real); reimplementa a sonda em `_recover`. O próprio arquivo documenta isso (linhas 250-263) — mas a lacuna segue aberta, e o item explícito "Não corrigido" do `AUDITORIA` (fim da seção 6) é exatamente este | 23 |
| `tools/quant_w4a8.py` | `as_bytes()` (linha 177) definido, zero sítios de chamada em `tools/` (`git grep -n "as_bytes" tools/` → só a própria definição); retorno do quantizador não validado além de um acordo tácito de shape (`AUDITORIA:193`, seção 5) | 24 |
| `tools/quant_int8.py` | sem preflight de backend mesmo quando `--device cuda` é passado (linha 113-114 só checa `torch.cuda.is_available()`, não resolve o backend nativo como `quant_w4a4.py`/`quant_w4a8.py` fazem); mesmo achado de contrato tácito do quantizador que `quant_w4a8.py` | 24 |
| `tools/fbcache_visual.py` | única ferramenta `fbcache_*` sem entrada na tabela de "Testes"/"Ferramentas" de `FBCACHE_FINDINGS.md` (comparar `FBCACHE_FINDINGS.md:302-312` com os 7 irmãos, todos listados) — não há evidência de que já rodou nem resultado registrado | 25 |
| `tools/hf_parallel_get.py` | construído para exatamente o problema que `fetch_ltx25.py`/`fetch_minimax_h3.py` descrevem (download resumível em paralelo de HF), mas nenhum dos dois importa este módulo — ambos usam `huggingface_hub.hf_hub_download` direto (single-connection, o problema que este arquivo existe para resolver). Zero sítios de chamada em todo o repo (`git grep -w hf_parallel_get` → nada) | 26 |

## Contagem final

93 arquivos, 93 linhas de classificação, cada arquivo em exatamente uma linha (14 raiz + 8 calib
+ 3 custom_nodes + 68 tools). Dentro de `tools/`: 8 já cobertos por ticket existente (01, 03, 04,
05, 06, 12, sem duplicar), 15 `suspeito` graduados em 9 tickets novos (18-26), 45 `saudável`.
0 `morto` confirmado, 0 `morto?` (todo candidato tinha `argparse`/`__main__` e cai na regra
"chamado só pelo usuário não é morto" — `hf_parallel_get.py`, ticket 26, é o mais perto da linha
e fica marcado para reavaliação).
