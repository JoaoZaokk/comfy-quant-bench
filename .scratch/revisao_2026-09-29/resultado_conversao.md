# Resultado — refatoração dos conversores (achados 1–10), 2026-09-29

Critério escrito antes: `criterio_conversao.md`. Tudo em CPU (`CUDA_VISIBLE_DEVICES=-1`), sem GPU,
sem modelo carregado, sem reconversão de modelo real, nada escrito em `models/`. Nenhum commit feito.
Artefatos: `.scratch/revisao_2026-09-29/conv/` (prova, testes, logs); código ANTES congelado em
`.scratch/antes_tools_conv_20260929/` (cópia do working tree, igual ao HEAD a menos de fim de linha).

## O que mudou (estrutura)

- **`tools/_profiles.py` (novo, 358 l.)** — um registro de perfis. `MODULE_PATTERNS` (calibração),
  `FILE_PATTERNS` (chaves do checkpoint), `WEIGHT_PATTERNS` = arquivo + `\.weight$` (derivado, não
  copiado), `PROFILE_PATTERNS` (as MESMAS 6 chaves de antes dos conversores), `EXCLUSIONS`,
  `detect_profile` (a versão do w4a8, com o ramo LTX), `select_layers(header, perfil, accepts)`.
  Substitui 4 cópias (w4a4, w4a8, calibrate ×2) e 5 `selected_layers`. Provado: as regex derivadas
  são caractere a caractere as antigas (6 conversores, 9 módulo, 9 arquivo).
- **`tools/_formats.py` (novo, 300 l.)** — modelo explícito por formato: `ConvrotW4A4`, `AsymW4A8`,
  `Int8Tensorwise`, `AwqW4A16` (frozen dataclasses com `accepts/tensors/layer_config/quantize`),
  `plan_layer/plan_model` (entradas preguiçosas; o kernel roda dentro do laço de escrita),
  `quant_metadata`, `layer_configs`, `streaming_peak`, `QUANTIZER_INPUT`.
- **`_conversion.py`** — leitura canônica (`read_header/read_tensor/LazyTensors`), sidecar DENTRO do
  commit (`commit(..., sidecar=)`: `.partial` exclusivo + fsync, sidecar posto antes do modelo e
  removido se a troca do modelo falhar; saída/sidecar criados por outro processo após a recusa são
  recusados no ato), conferência de FORMA e DTYPE de todo tensor produzido contra o plano,
  `planned_size` exato, `plan_copy_from` (cópia de outro arquivo), `write_json_exclusive`
  (`write_sidecar` agora atômico e exclusivo), `save_file_order/metadata`, `header_bytes` com
  `metadata_last/ensure_ascii` só para reproduzir scripts legados. `guard_ram` confere RAM E commit.
- **`_ram_guard.py`** — `commit_free_bytes()/commit_gib()` via `GetPerformanceInfo`
  (PERFORMANCE_INFORMATION: `(CommitLimit-CommitTotal)*PageSize`) e `check_commit()`; usado por todo
  `Conversion.guard(accumulated=...)`. Saíram `w4a8_bytes/int8_bytes/convrot_w4a4_bytes` (só serviam ao
  acúmulo de duas passadas, que acabou). Medido nesta máquina: 58,9 GiB de commit livre no momento.
- **Todos os conversores transmitem** (`plan_lazy`): w4a4 (já transmitia), w4a8, int8, mixed, smooth,
  awq, te_residuos. Guardas de RAM/commit/disco rodam ANTES de qualquer quantização/calibração.
- **Achado a achado**:
  1. `svdq_to_bf16`: `load_file` da fonte → `LazyTensors`; `load_reference` → `_Reference` (lê só
     headers, puxa o tensor pedido). `ajusta_denso_diffusers`: `load_file` → leitura por faixa;
     escrita por `grava_saida` no núcleo (modelo + relatório `.json` num commit).
  2. `extrai_transformer`, `transplanta_klein`, `grava_pesos_recuperados`, `mistura_klein`: pelo
     núcleo (fim do "wb", do assert, do `save_file` direto no nome final, da cópia à mão).
     Funções ternárias (`BLOCO`, `pilhas_reais`, `cod_esc`, `rtn4`) intocadas.
  3. Sidecar no commit atômico em w4a4, w4a8, int8, mixed, smooth, awq, weight_only, te_residuos.
  4. Smooth: guardas antes da calibração; entrada do quantizador FP32 (decisão do dono), registrada no
     sidecar como `"quantizer_input": "float32"`.
  5. Código morto apagado (quant_w4a8: header_dtype/as_bytes/SAFETENSORS_DTYPE/copy_range/read_*;
     quant_w4a4: output_header/encoded_header/copy_range/write_tensor/read_tensor_range/estimate_output;
     quant_mixed: read_header/read_tensor/copy_range/SAFETENSORS_DTYPE/human_size; to_native:
     TORCH_DTYPES). Imports de quant_gguf, svdquant_probe, weight_balance, plot_weight_balance e
     quant_misto_w4a8_int8 apontam para `_conversion`/`_profiles` (lógica deles intocada). Nenhum
     módulo importa mais de `quant_w4a8` (só `test_native_probe`, que testa o próprio w4a8).
  6. Perfis em `_profiles`; `EXCLUSIONS` vale para todos os conversores; LTX autodetectado no w4a4.
  7. `verify_w4a4` usa os nomes de `_formats`, passou a conhecer `awq_w4a16` e `float8_e4m3fn`
     (estrutura + bytes da fonte; sem probe/smoke, dito no bloco de cobertura) e detecta SmoothQuant
     pelo `smoothquant_alpha` do próprio arquivo (sidecar vira segundo sinal).
  8. Duas passadas → streaming (acima).
  9. `quant_mixed.main` (≈510 l.) dividido em `load_analysis`, `load_calibration`, `measure_all`,
     `decide` (pura), `report`; arquivo 970 → 889 l. `decide` testado por tabela.
  10. w4a4: recusas antes do dry-run e do preflight; smooth: checagem de CUDA depois das recusas;
      docstring falsa do `convert.py` corrigida (+ subcomandos awq, weight-only, te-residuos).

Linhas (git numstat nos 27 arquivos existentes que toquei): **+1562 / −2000**; novos: `_formats.py`
300, `_profiles.py` 358, `test_formats_e2e.py` 366. Observação: os arquivos que reescrevi saíram com
fim de linha LF (o working tree tinha CRLF; o git já armazena LF, então o diff do git não acusa).

## Testes (antes → depois, mesmos 9 + 1 novo)

| teste | antes | depois |
|---|---|---|
| test_conversion_core | rc=0 (38/38) | rc=0 (38/38) |
| test_native_probe | rc=0 | rc=0 |
| test_quant_mixed_provenance | rc=0 (9/9) | rc=0 (9/9) |
| test_quant_mixed_sigma | rc=0 | rc=0 |
| test_smooth_guards | **rc=1** (6 falhas: "CUDA is unavailable" antes de qualquer guarda) | **rc=0** (6 OK, 1 pulado por falta de arquivo real) |
| test_svdq_verify | rc=0 | rc=0 |
| test_svdq_write_contract | rc=0 | rc=0 |
| test_verify_formats | rc=0 (13/13) | rc=0 (13/13, com awq e fp8 no caso "each format") |
| test_avaliar_backend | rc=0 | rc=0 |
| test_formats_e2e (novo) | — | rc=0 (10/10) |

O `test_smooth_guards` passou a rodar porque a checagem de CUDA desceu para depois das recusas
(achado 10), não porque alguma guarda afrouxou: os 6 casos exigem a mensagem de recusa específica.
Logs: `conv/testes_antes/`, `conv/testes_depois/`.

## Prova de equivalência (fontes sintéticas, comfy-kitchen eager REAL em CPU)

`conv/prova/prova.py` roda cada caso com o código antigo e com o novo; `comparacao.txt`:

- **Byte a byte idênticos (saída e sidecar normalizado)**: w4a4 (gemma, gemma cg64, ltx), w4a8
  (gemma, gemma sem codebook gs32, ltx), int8 (gemma convrot, gemma sem convrot, ltx), mixed
  (3 formatos + budget; somente-w4a4), awq, weight_only, te int8, te fp8, svdq (split + referência),
  svdq keep-fused, extrai, mistura, mistura rtn, transplanta (3 variantes). Sidecars com os mesmos
  bytes de fim de linha (CRLF) de antes. Normalizados só `conversion_seconds` e o prefixo do dir de saída;
  `comfy_version` também, porque o `git rev-parse` falhou no processo do código antigo ("unknown").
- **grava / grava_int8 / ajusta**: header parseado, offsets, ordem dos tensores e dados idênticos; só
  a ordem das chaves do `__metadata__` difere. Motivo medido: o `save_file` do safetensors 0.8 ordena o
  metadata por um HashMap de semente aleatória — 3 execuções, 3 ordens; e o próprio código ANTIGO deu
  dois sha256 diferentes em duas execuções do mesmo caso. O novo ordena as chaves (reprodutível).
  Isto corrige uma premissa do critério ("reproduzir o save_file byte a byte" não é possível nem para
  o código antigo). `ajusta` prova só a escrita (`prova_ajusta.py`: trecho antigo copiado x
  `grava_saida`): equivalente, e o relatório `.json` byte a byte igual.
- **w4a4_ltx_auto**: antes ValueError; depois converte, e a saída é byte a byte igual à do
  `--profile ltx_2_5` explícito (decisão do dono).
- **smooth (FP32, decisão do dono)**: (a) com o quantizador forçado a receber BF16 (`WRAP_BF16`), a
  saída do código novo é **idêntica** à do antigo, e o sidecar difere só na chave nova
  `quantizer_input`; (b) FP32 nativo, medido (`mede_smooth.txt`): header e metadata idênticos,
  tensores preservados e normas reescritas (lambda) idênticos byte a byte, **0,694% dos códigos int4
  diferentes** (8.186/1.179.648), escalas com erro relativo médio 2,52e-3 (máx 2,77e-3, 14 tensores).
  Sintético, eager em CPU: dá a ordem de grandeza, não o número do CUDA num Gemma real.
- **verify_w4a4 antigo x novo** nas saídas (`verifica_saidas.txt`): mesmos PASS em tudo que o antigo
  conhecia; awq e fp8 passam a PASS (antes "unexpected format"). Smooth SEM sidecar: antigo reprova
  (normas "preserved tensor bytes changed"), novo passa pelo `smoothquant_alpha` do arquivo.
- **Header planejado x arquivos REAIS do disco** (`plano_real.txt`, só headers): 8 de 8 idênticos byte a
  byte — mixed krea2_turbo_w4a4, qwen_image_edit_2511_w4a8, zimage-v2/deturbo/turbo_w4a4; w4a8
  qwen3vl_4b e ltx-2.5-22b-distilled (1440 camadas, header 1,53 MB); w4a4 qwen_2.5_vl_7b. Um pulado
  (fonte ausente). `verificar_migracao.py`: roda igual, par único ausente do disco (antes e depois).
- **Seleção nos headers reais** (`selecao_real_*.json`, 114 arquivos, todos os perfis x w4a4/w4a4 cg64/
  w4a8/int8/int8c/awq/mixed): **zero listas diferentes** — a `EXCLUSIONS` estendida a todos não cortou
  nada. Diferenças só: 9 arquivos LTX agora autodetectados pelo w4a4, e o texto do ValueError de
  autodetecção.

Ruído encontrado e NÃO introduzido: a tabela diagnóstica do `svdq --verify` (colunas do controle
embaralhado) muda de execução para execução porque `verify()` usa `torch.randn` sem gerador —
pré-existente, não entra nos bytes gravados.

## O que não fiz, e por quê

- `quant_audit.py`, `check_w4a8.py`, `activation_balance.py`: sem mudança. O `human_size` do
  quant_audit é formato de SAÍDA do inventário gerado (mexer mudaria o inventário); os outros dois são
  sondas de GPU sem duplicação do núcleo.
- `to_native.py`: só o `TORCH_DTYPES` morto saiu; a recusa condicional ao dry-run é decisão documentada
  no próprio arquivo (a de fonte quantizada vale no dry-run).
- `svdq_to_bf16` continua acumulando os pesos RECUPERADOS antes de escrever (o kernel só roda com CUDA e
  o fluxo valida tudo antes de escrever); só a leitura virou streaming (achado 1).
- `grava_pesos_recuperados` continua acumulando as camadas recuperadas (pequenas); só a escrita mudou.
- `transplanta_klein` agora faz uma passada por variante (antes escrevia as 3 numa passada) — custo de
  leitura 3x num script de CPU.
- Edge não coberto: `mistura_klein` com base de `__metadata__` explicitamente VAZIO (`{}`) passa a sair
  sem metadata (antes saía com as chaves `mistura_*`); nenhum arquivo real conhecido tem `{}`.
- Não editei W4A4_PROGRESS/HANDOFF/AGENTS/CLAUDE nem inventário (fora da posse).

## Validação que exige GPU (para a janela autorizada pelo dono)

Protocolo de lock da bancada; um conversor por vez; saída em pasta nova. Medida: sha256 da saída e
`verify_w4a4 --kernel-smoke` contra a fonte, comparando com um arquivo produzido pelo código antigo
(`.scratch/antes_tools_conv_20260929/`) na mesma placa, mesma fonte.

1. `python_embeded\python.exe -s tools\quant_w4a4.py --input <qwen_2.5_vl_7b bf16> --output <novo>` →
   sha256 igual ao `qwen_2.5_vl_7b_w4a4_convrot.safetensors` real (o header já bate; falta o dado CUDA).
2. `tools\quant_w4a8.py --input <ltx-2.5-22b-distilled bf16>` → sha256 igual ao
   `ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors` real; conferir pico de RAM/commit menor
   que o da versão de duas passadas (medida: pico de working set do processo).
3. `tools\quant_mixed.py --input zimage_turbo_bf16 --analysis <a mesma análise> ...` → sha256 igual a
   `zimage_turbo_w4a4.safetensors` real.
4. `tools\quant_int8.py --device cuda --convrot` numa fonte pequena, código antigo x novo → sha256.
5. Smooth FP32 num Gemma real: `--kernel-smoke` (rel-RMSE na faixa ~0,21 com lambda) e render pareado
   BF16-antigo x FP32-novo; fração de códigos diferentes medida no arquivo real.
6. `svdq_to_bf16` com nunchaku real: sha256 igual ao de uma recuperação antiga da mesma fonte, e pico de
   RAM menor (a fonte não é mais carregada inteira).
7. `awq`/`te_residuos` rodam em CPU: podem ser reconvertidos sem GPU se o dono quiser a prova em modelo
   real (não fiz: "não reconverta modelos reais").

> 2026-10-01: `.scratch/antes_tools_conv_20260929/` foi removida por ser idêntica ao commit `540ccef` (46/46 arquivos, conferido com `cmp`). Recuperar com `git show 540ccef:tools/<arquivo>`.
