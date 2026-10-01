# Resultado — core ComfyUI + comfy-kitchen + patches + preflight + lowbit (2026-09-29)

Critério: `criterio_core.md` (escrito antes). Tudo em CPU com `CUDA_VISIBLE_DEVICES=-1`; nenhuma GPU usada.

## Testes (medidos nesta sessão)
| conjunto | antes | depois |
|---|---|---|
| ComfyUI `tests-unit/comfy_quant` + `test_symmetric_patchifier.py` (+ `test_cast_to_gathered_fallback.py` depois) | 12 passed | 25 passed (+1 AWQ do PR, +2 costura, +6 TE, +4 aimdo) |
| ComfyUI `tests-unit/comfy_quant` + `tests-unit/comfy_test` inteiro | não medido antes | 174 passed |
| preflight `test_checks.py` (pytest) | 40 passed | 40 passed (−2 removidos de propósito, +2 novos) |
| lowbit `test_lowbit.py` (script) | 11 ok, 1 SKIP | 14 ok (+3 costura), 1 SKIP (GPU, opt-in) |
| kitchen `patches/tests/test_comfy_kitchen_awq.py` | não existia | 6 passed, 2 skipped (GPU, opt-in) |
| `tools/verifica_patches.py` | não existia | 7 APLICADO, 4 comfyui_* aplicam sozinhos sobre v0.37.4, zen sem alvo; exit 0 |
Logs: `antes_core_*.log`, `depois_core_*.log`.

## Equivalência provada
- Kitchen: 0.2.35 limpo reconstruído e conferido contra o RECORD do wheel (sha256 de 5 arquivos). Limpo + w4a8 + awq ==
  instalado, byte a byte (cmp nos 5 arquivos).
- Dequant eager == cadeia torch antiga do CUDA (torch.equal); gemv eager inalterado (torch.equal); fallback M>256
  despacha para eager em CPU e dá o mesmo resultado (torch.equal).
- Lowbit pela costura: qdata/scale/zero byte-idênticos, bits/group corretos, dequant igual à referência, 1/2/4 bits.
- AWQ pela costura: teste do PR 579ad336 passa sem skip; group_size via `params` testado.
- TE: decisão depois do load_sd; camada `full_precision_matrix_mult` mantém upcast (falharia com o patch antigo);
  formato desabilitado mantém upcast; flag CLI mantém upcast; kernel só ligado dentro do encode.
- `sd1_clip.py` idêntico ao v0.37.4. Stash e branch awq-w4a16-format com o mesmo hash antes/depois.

## Não provado aqui (exige GPU)
Despacho real do kernel quantizado no encode, fidelidade do condicionamento, fallback aimdo em falha real de HtoD,
paridade Triton×eager do AWQ, velocidade AWQ com `--enable-triton-backend`, render lowbit/AWQ pela costura.

## Adendo — lowbit: fusão q/k/v em nomes diffusers (achado do agente do QAT)
- Causa: `formats.diffusers_flux2_to_bfl` agrupava to_q/to_k/to_v só pela letra; num arquivo já salvo como
  lowbit_affine em nomes diffusers, `weight`, `weight_scale`, `weight_zeros` e `comfy_quant` se sobrescreviam
  (o teste novo contra o `formats.py` antigo: `weight` fundido saiu com shape [81] = 3 × config de 27 bytes).
- Correção: agrupa por (módulo fundido, sufixo); cada sufixo concatena nas linhas; `comfy_quant` é um só (exige
  q/k/v iguais); camadas empacotadas pelos readers usam sufixo None (caminho LowBit inalterado); falta de q/k/v
  vira ValueError em vez de KeyError. `weight` denso: mesma concatenação de antes (teste dedicado).
- `test_lowbit.py`: antes 14 ok + 1 SKIP (GPU) → depois 16 ok + 1 SKIP (+2: sufixos de lowbit salvo, denso
  inalterado). Logs `antes_lowbit_qkv.log`, `depois_lowbit_qkv.log`. CPU, `CUDA_VISIBLE_DEVICES=-1`.
- Não provado aqui: carga real de um klein lowbit em nomes diffusers e render (GPU).

## Consequências e pendências
- AWQ Triton agora respeita o registry: sem `--enable-triton-backend` (nenhum launcher .bat usa) roda o eager,
  igual ao kitchen de fábrica; o ganho medido em 27/09 só volta com a flag.
- `tools/avaliar.py:729-733` procura marcas do patch antigo (`def quantized_text_encoder_math`, `sd1_clip.py`,
  `text_encoder_has_quantized_math(`) e agora vai dizer "travas presas". Fora da minha posse.
- `patches/tests/` e `patches/arquivo/` caem no `.gitignore` do root (`/patches/*`, só `*.patch` liberado).
- W4A8 kitchen (item 9) não simplificado: as funções são do backend CUDA (_C), sem paridade provável em CPU.
