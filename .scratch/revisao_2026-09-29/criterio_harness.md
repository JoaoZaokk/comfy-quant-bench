# Critério — correção do harness (achados 1-10 + movimento estrutural)

Escrito ANTES de implementar. GPU proibida: nada aqui sobe ComfyUI, toma lock ou roda bateria.
Todo Python com `CUDA_VISIBLE_DEVICES=-1`, `python_embeded/python.exe -s`.

## Linha de base (antes)
`tools/test_comfy_run_workflow.py` 36 PASS, `test_timing` 41 PASS, `test_ltx_studio` 71 PASS,
`test_avaliar_backend` rc 0, `test_gpu_lock` rc 0. Logs: `antes_test_*.log`.

## Aceito quando
1. **bench_server**: `_lock()` lê `gpu_lock.read_state()` (F:/GPU_BENCH.lock, chave `dono`) e
   descreve com `describe()`; teste com arquivo de lock falso em tmp prova livre/ocupado. Conversão
   real roda dentro de `GpuLock(owner)`; lock ocupado → recusa sem iniciar subprocesso.
2. **comfy_client.py** só stdlib (`python3 -I` sem site-packages importa; checado com
   `-S -I` / varredura de imports). Contém `Comfy`, `run_and_wait` (sempre com prazo), `Entry`,
   `PollTimeout`, `bases_irmas`, gravação jsonl.
3. **submit()** levanta com `node_errors` não vazio em HTTP 200 (teste contra servidor HTTP falso).
4. **run_and_wait** estoura `PollTimeout` quando o /history nunca responde (servidor falso, prazo
   curto) e aceita callback por tick (para amostragem de GPU/RAM do roda_eros_2gpu).
5. Normalização de base: `127.0.0.1:8190` e `http://127.0.0.1:8190` viram o mesmo cliente; worker
   do MultiGPU achado por `bases_irmas` é consultado de fato (teste com servidor falso cujo
   /history só responde no "worker").
6. **Executor**: `comfy_run_workflow.py --api-prompt FILE` e `--lista ordem.txt --saida x.jsonl`;
   uma linha por grafo `{grafo, prompt_id, status, server_side_s, wall, cache_hit, files, placa,
   args}`; exit != 0 se qualquer grafo falhar/estourar. Goldens de `--dump-api` inalterados.
7. **Scripts de bateria** (roda_eros_2gpu, roda_lowbit, ltx_video, ltx25_queue, qwen_edit_test)
   usam `comfy_client`; exit != 0 em falha; nenhum laço sem prazo.
8. **Métricas**: `metricas_imagem.medir()/identica()/imagem_unica()`; main usa as mesmas funções
   (saída numérica idêntica em par de imagens sintéticas antes/depois). Chamadores
   (cruza_runtimes, metricas_bateria, metricas_lowbit, folha_contato) usam essas funções;
   nenhum `_00001_` fixo; junção log↔grafo recusa contagem divergente; modo jsonl junta por
   `grafo`/`files`.
9. **PowerShell**: `tools/comfy_server.ps1` `Invoke-ComfyUnderLock` — Assert-GpuLock, sobe,
   roda bloco, derruba com `ErrorActionPreference=Continue`, confirma porta caída antes de
   `Release-GpuLock`; se a porta não cai, NÃO solta e avisa. roda_bateria/roda_qwen21/roda_diag
   passam a usá-lo. Checado por parse (`[Parser]::ParseFile`) — não executado (subiria ComfyUI).
   encadeia*.ps1 do lowbit: espera por sentinela com prazo.
10. **avaliar_despacho**: recusa sem lock vivo (read_state + pid vivo + hb recente), grava
    controles (lock, device, CVD, versões por metadata, HEAD do ComfyUI) em cada laudo.
11. **Arquivo**: só probes com ZERO referência (grep -r em tools/, docs/, .agent-reference/,
    AGENTS.md, W4A4_PROGRESS.md, W4A4_HANDOFF.md, .scratch/), movidos com `git mv` para
    tools/_arquivo/. Nada apagado.
12. `--warm` corrigido na doc. avaliar.py só se decomposição óbvia (espera-se: não mexer).
13. Testes existentes: mesmos PASS depois; novos testes em `tools/test_comfy_client.py`.

## Não prova
Nada aqui prova comportamento contra ComfyUI real, MultiGPU real, ou GPU. Servidor falso prova
contrato HTTP, não semântica do ComfyUI.
