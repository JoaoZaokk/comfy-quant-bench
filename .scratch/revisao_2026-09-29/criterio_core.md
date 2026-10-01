# Critério — revisão core ComfyUI + comfy-kitchen + patches + preflight + lowbit (2026-09-29)

Escrito ANTES de editar. Sem GPU nesta rodada: tudo CPU com `CUDA_VISIBLE_DEVICES=-1`, `python_embeded/python.exe -s`.

## Baseline (antes)
- `pytest tests-unit/comfy_quant tests-unit/comfy_test/test_symmetric_patchifier.py` (ComfyUI) — contagem pass/fail/skip.
- `pytest custom_nodes/comfy-quant-preflight/test_checks.py` — contagem.
- `custom_nodes/comfy-lowbit-loader/test_lowbit.py` como script — contagem ok/FAIL/SKIP.
- Resultados guardados em `.scratch/revisao_2026-09-29/antes_core_*.log`.

## O que prova equivalência / correção, por item
1-2. Text encoder:
  - `comfy/sd1_clip.py` volta a ser idêntico ao v0.37.4 (git diff vazio no arquivo).
  - A decisão do upcast fp32 é tomada DEPOIS de `load_sd` e usa `MixedPrecisionOp.can_use_quantized_matmul` (predicado do core), sem `getattr(..., default)`.
  - Teste CPU: camada int8 com `full_precision_matrix_mult: true` → predicado False; mesma camada sem a flag → True
    (com formatos desabilitados = vazio); camada não quantizada numa MixedPrecisionOps → False.
  - `--disable-quantized-text-encoder` continua aceito (tools/ usam) e desliga o caminho novo por completo.
  - Não verificável sem GPU: kernel quantizado realmente despachado no encode e fidelidade do condicionamento → listado para o dono.
3-4. Preflight: checagem `check_inert_full_precision_flag`, `WEIGHT_ONLY_FORMATS` e os 2 testes removidos; header lido
   uma vez por (path, mtime_ns, size) (teste conta leituras); wrapper só olha nós cuja class_type está na tabela
   (teste com nó alheio que tem widget `ckpt_name`). Demais testes existentes continuam passando.
5. aimdo: fallback só em `cast_to_gathered`, captura só RuntimeError e só quando há destino em dispositivo,
   limpa o erro na placa do destino, avisa uma vez. Teste CPU: leitura falsa que lança → cópia normal feita, bytes iguais,
   1 aviso para 2 tensores, device passado ao descarte = device do destino; RuntimeError sem destino de dispositivo propaga.
6. comfy-kitchen: `dequantize_awq_w4a16` registrado em eager e triton via registry; `_awq_w4a16_dequant_then_matmul`
   usa o dispatch. Teste CPU: eager == fórmula de referência (torch.equal) e o resultado do dequant+matmul == caminho
   torch antigo. Paridade triton vs eager exige GPU → teste com opt-in, listado.
   Consequência a registrar: com o registry, o caminho triton só roda com `--enable-triton-backend` (ComfyUI desliga o
   backend triton por padrão).
7. Branch `local/0.37.4` no checkout ComfyUI, um commit por patch sobre 8ff6dc38 (v0.37.4); `patches/comfyui_*.patch`
   gerados por `git format-patch`; `tools/verifica_patches.py` reporta todos aplicados (`git apply --check -R`) e cada
   patch ComfyUI aplica sozinho sobre v0.37.4 limpo. Stash e branch awq-w4a16-format intocados (hash conferido antes/depois).
8. Patch AWQ inclui o teste do commit 579ad336 sem o skipTest morto; `group_size` lido com fallback em `params`.
9. W4A8 kitchen: só simplificar se a paridade numérica do fallback puder ser provada em CPU; senão não mexer.
Code judo: `QUANT_ALGOS[fmt]["params_from_state_dict"]` como costura única em `_load_quantized_module`;
   AWQ e lowbit passam por ela; `layout.py` do lowbit perde o monkeypatch. Teste: camada lowbit carregada via
   `mixed_precision_ops().Linear` em CPU tem qdata/scale/zero byte-idênticos à fonte e mesmo bits/group_size;
   AWQ carrega e salva igual ao teste do PR.

## Aceitação
Testes depois ≥ antes (mesmos passam, novos passam), exceto os 2 removidos de propósito. Nenhuma execução em GPU.
