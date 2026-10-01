# Revisão estrita de qualidade — 2026-09-29

Escopo: tudo que mexemos (core ComfyUI patchado + comfy-kitchen, conversores em tools/, harness de bench/avaliação,
custom nodes próprios + QAT klein, Arc). Leitura de código; 4 revisores em paralelo + Arc revisada diretamente.
Conferido por mim (leitura direta): P1, P2, P3, P5, A1–A3. O resto é achado de revisor com arquivo:linha citado, não reexecutado.

Incidente: o revisor de custom nodes rodou `custom_nodes/comfy-lowbit-loader/test_lowbit.py`; o teste
`test_triton_matches_torch_bit_for_bit` usou cuda:0 por segundos sem lock (CUDA_VISIBLE_DEVICES="" não esconde no Git Bash).
Sem modelo carregado. Falha na instrução do revisor; o teste também roda GPU sem opt-in.

## Prioridade 1 — defeitos reais (corrigir antes de refatorar)

- P1 `tools/bench_server.py:229-235` lê `F:/COMFY_PORTABLE/GPU_BENCH.lock` com chave `owner=`; lock real é
  `F:/GPU_BENCH.lock` com `dono=`. Sempre "livre"; trava da conversão real (l.335) nunca dispara; conversão roda sem lock.
- P2 `ComfyUI/comfy/sd.py:275` decide o upcast fp32 do TE (`text_encoder_has_quantized_math`) ANTES do `load_sd`;
  a flag por camada só é corrigida no load (`ops.py:1197-1201`). TE W8A16/W4A16 ou int8 sem suporte perde upcast sem
  ganhar kernel; yue2/minimax/sd.py:1029 disparam sem a CLI; `--disable-quantized-text-encoder` não restaura tudo.
- P3 `tools/qat_ternario_klein.py:554-562` engole qualquer exceção do HF como "começando do zero"; depois `envia` sobe
  checkpoint novo e faz `super_squash_history` → falha de rede apaga o checkpoint remoto avançado.
- P4 QAT: shards do professor nomeados por posição (`p{i:04d}_s{s}.pt`), prompt nunca conferido;
  `gera_dataset_klein.py:136-159` numera dentro de `faltam` → pula prompts novos após SIGKILL. Latente hoje.
- P5 `tools/svdq_to_bf16.py:548` `load_file` do modelo inteiro (e `load_reference` junta shards); também
  `ajusta_denso_diffusers.py:319`. Viola regra de streaming.
- P6 Sidecar `.quant.json` fora do commit atômico (`write_text` após `commit()` em w4a4:501, w4a8:394, int8:232,
  mixed:959, smooth:316). Queda entre os dois deixa modelo sem sidecar e rerun recusado.
- P7 Conversores recentes sem `Conversion.commit`: extrai_transformer.py:61, transplanta_klein.py:113-134 (TOCTOU,
  assert), grava_pesos_recuperados.py:174 (grava direto no nome final).
- P8 `quant_w4a4_smooth.py`: guard de RAM depois do acúmulo (l.312) e rotação ainda em BF16 (l.243; a correção FP32 de
  26/09 não chegou aqui). Trocar para FP32 muda bytes: decisão do dono.
- P9 `quant_w4a4.py`: dry-run retorna (l.439) antes de `refuse_unsafe` (l.463).
- P10 Retomada do QAT inconsistente: melhor.json/journal a cada avaliação, checkpoint a cada 25 min; ordem/RNG fora do ckpt.
- P11 `comfy-quant-preflight/checks.py:166-195` avisa que `full_precision_matrix_mult` é inerte — falso no core atual
  (base do W4A16/W8A16 do projeto); 2 testes fixam o erro.
- P12 comfy-kitchen patch AWQ: backend cuda importa `backends.triton.awq` direto → kernel Triton roda mesmo com
  `ck.registry.disable("triton")` do ComfyUI; `except ImportError` silencioso (~16× mais lento).
- P13 `patches/comfyui_awq_w4a16_format.patch` perdeu o teste que existe no commit do PR (579ad336).
- P14 `memory_management.py:71-85` fallback aimdo: `except RuntimeError` largo, só ramo hostbuf, sincroniza a placa errada.
- P15 Polling sem timeout / exit 0 em falha: roda_arc/edit/turbo.py, roda_lowbit.py, roda_eros_2gpu.py:86-91,
  encadeia*.ps1. `comfy_run_workflow.submit()` ignora `node_errors` em HTTP 200.
- P16 Métricas casam tempos por posição no log (metricas_bateria.py, metricas_lowbit.py) e leem `_00001_.png` fixo.
- P17 `roda_qwen21.ps1`, `roda_diag.ps1` sem a correção `'Continue'` antes do taskkill → podem soltar lock com servidor vivo.
- P18 `comfy-qwen21-nunchaku` aceita LoRA e descarta sem aviso; `comfy-void-stage-tools` cria contexto CUDA em todas as placas.
- P19 `quant_w4a4.detect_profile` sem o ramo LTX que o w4a8 tem → `--profile auto` em LTX falha.
- P20 `gera_dataset_klein.py:54-91` reescreve metadata.jsonl inteiro; erro passageiro/2 VMs perdem linhas.

## Arc (revisada diretamente)

- A1 Notas dos workflows de edição com tempos pré-medição: turbo4 "~50 s" (medido 24), turbo6 "55-80" (34), turbo8 "60-95" (43).
- A2 Comentário do patch zen culpa "Xe KMD"; causa real era vIOMMU (iommu=pt resolve). Patch aplicado == arquivo (md5).
- A3 `roda_edit.py` local diverge do rodado na Arc (falta gorro768, base25s5, 3º argumento).
- A4 Grafo zen+GGUF+VAE, SIGMAS, nomes de LoRA e cliente /prompt copiados em 5 scripts; tag por prefixo (`startswith("v01")`).
- A5 Edição turbo8: `ModelSamplingFlux` fixo 768×768 independentemente do formato da imagem 1.

## Movimentos estruturais (code judo), por área

1. Core ComfyUI: patch do TE reescrito sobre `can_use_quantized_matmul`/`use_quantized_matmul` já existentes
   (79 → ~10 linhas, decisão pós-load). Uma costura única em `_load_quantized_module` (tabela por formato) substitui o
   hunk awq/w4a8 em ops.py E o monkeypatch do lowbit. Branch local `local/0.37.x` com um commit por patch; `patches/`
   gerado por `format-patch`; verificador `git apply --check -R` para ComfyUI + kitchen.
2. Conversores: `_profiles.py` (Profile único; hoje 4 cópias divergentes) + `_formats.py` (`QuantFormat.tensors/
   quantize/layer_config`) usado por conversores E `verify_w4a4`; todos em `plan_lazy` streaming; sidecar dentro do
   `commit`. Apaga código morto (copy_range/read_header em w4a4/w4a8/mixed). Estimativa do revisor: −600 a −900 linhas.
3. Harness: `comfy_client.py` (Comfy/run_and_wait/Entry + node_errors) como executor único gravando jsonl por
   `prompt_id`; `metricas_imagem.medir()` exportado; `tools/comfy_server.ps1` para sobe+lock+teardown. Arquivar ~15
   probes sem referência em `tools/_arquivo/` (git mv, após conferir W4A4_PROGRESS e .scratch).
4. QAT: `lowbit_canon` (Quantizador imutável, pilhas_reais, empacotamento lowbit_affine, Shard por conteúdo) compartilhado
   com loader/construtores; decompor qat_ternario_klein.py (1065 l.) sem globais; `EstadoCorrida` no ckpt; `status.json`.

Detalhe completo de cada revisor: transcript da sessão de 2026-09-29.
