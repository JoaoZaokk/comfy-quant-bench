# Auditoria da toolchain W4A4 — 2026-08-18

> **Nada aqui foi verificado por terceiro.** Cada achado veio com um campo `confianca` preenchido pelo próprio auditor que o escreveu — isso é auto-avaliação, não evidência. Trate todo item deste documento como **hipótese a confirmar**.
> **A ordem é por severidade *alegada*, não por risco comprovado.** O passo seguinte não é consertar: é rodar o comando ou fazer a leitura da coluna "como confirmar" e só então decidir.

Escopo: 127 achados brutos de 10 frentes; 126 alvos únicos (`comfy_kitchen/tensor/convrot_w4a4.py:237` foi reportado por dois auditores independentes). Distribuição por área: conversores clássicos 15, harness de medição 15, wrapper comfy-kitchen 14, kernels comfy-kitchen 14, verificação/inventário 13, benchmarks novos 13, integração ComfyUI 12, ferramentas antigas 12, SVDQ recovery 10, pipeline novo 9.

Restrição durante a auditoria: **GPU proibida** (lock ativo em `F:/GPU_BENCH.lock`, confirmado presente). Tudo abaixo foi levantado por leitura, grep e Python de CPU puro. Os itens marcados **[GPU]** não podem ser confirmados até o lock liberar.

Regra dura do projeto que organiza a lista inteira: *W4A4 significa execução nativa ConvRot CUDA*. Qualquer caminho que caia em eager/dequantizado sem gritar é severidade máxima por definição, não por gosto.

---

## 1. Tabela de prioridade

| # | Alvo | Por que importa (1 linha) | Como confirmar |
|---|---|---|---|
| 1 | `ComfyUI/comfy/ops.py:1373` — `_full_precision_mm` é no-op para `convrot_w4a4` / `asym_w4a8_int8` / `int8_tensorwise` | O flag que deveria forçar BF16 numa camada sensível não força nada: o peso continua `QuantizedTensor` e `F.linear` chama o kernel W4A4 assim mesmo. Atinge também `sd1_clip.py:114` e `gpt_oss.py:516`, que passam `full_precision_mm=True` para **todo** text encoder. | CPU. `mixed_precision_ops({'mixed_ops':True}, torch.bfloat16, full_precision_mm=True, disabled=set())`, `Linear(256,64,bias=False)` carregado com `comfy_quant={'format':'convrot_w4a4'}`; checar `type(cast_bias_weight(lin, x_bf16)[0]).__name__` — se sair `QuantizedTensor` com `lin._full_precision_mm == True`, confirmado. |
| 2 | `ComfyUI/comfy/ops.py:431` — entrada com dtype ≠ `compute_dtype` dequantiza o peso | Um único nó upstream emitindo fp32 tira o modelo inteiro do caminho ConvRot, sem log. Lógica duplicada em `post_cast` (`ops.py:300-306`). | CPU. Mesma `Linear` do item 1: `cast_bias_weight(lin, torch.randn(4,256,dtype=torch.float32))` deve devolver `Tensor`, e com entrada bf16 deve devolver `QuantizedTensor`. |
| 3 | `comfy_kitchen/tensor/base.py:365` — fallback universal para dequantização em `logger.debug` | Qualquer `aten` op não registrado (`mul`, `add`, `slice`, `split`, `baddbmm`) dequantiza o peso W4A4 para BF16 no meio do forward. O logger herda WARNING, então a linha nunca aparece. | Leitura + 1 linha: rodar um workflow real com `logging.getLogger('comfy_kitchen.tensor.base').setLevel(logging.DEBUG)` e contar. Sem isso, `grep -n "_DISPATCH_TABLE\|_LAYOUT_DISPATCH_TABLE" comfy_kitchen/tensor/base.py` e comparar a lista de ops registrados com os ops que o modelo usa. |
| 4 | `comfy_kitchen/tensor/convrot_w4a4.py:237` — peso `transposed` → `F.linear(x, weight.dequantize(), bias)` **(2 auditores)** | É literalmente "INT4 weight-only + GEMM BF16", o resultado que o CLAUDE.md proíbe. `_handle_convrot_w4a4_t` (`:195`) só inverte a flag, então qualquer `aten.t` antes do `linear` cai aqui. Os irmãos `mm`/`addmm` fazem o oposto (exigem `transposed=True` e usam o kernel). | Leitura direta das linhas 195-241. Contagem real: monkeypatch em `_handle_convrot_w4a4_linear` contando entradas no ramo `transposed` durante um forward. **[GPU]** para o forward; o mecanismo é confirmável só por leitura. |
| 5 | `comfy_kitchen/registry.py:218` e `:202` — queda silenciosa cuda→eager; `use_backend('cuda')` não é pin | `get_capable_backend` guarda o motivo num dict local e continua para o próximo backend; só `logger.debug` registra. `use_backend` falha a validação e **cai de volta na prioridade** em vez de levantar. Triton está desabilitado por padrão (`comfy/quant_ops.py:65`), então o próximo é eager. | Leitura de `registry.py:202-232` e `:266`. Reproduzível em CPU importando `constraints.py` isolado e alimentando um kwargs que reprova (ex.: `x` 1-D, que o cuda rejeita por `MinDims(2)` e o eager aceita). |
| 6 | `comfy_kitchen/backends/cuda/__init__.py:2261` e `:2213` — `w4a8_int8_linear` volta para eager sem log; e o ramo chunked pode fazer GEMM com ativação **não inicializada** | `:2261` retorna `eager_w4a8_int8_linear` quando `cutlass_int8_dequant` devolve falsy — o preflight das ferramentas não enxerga isso, porque ele só olha qual implementação o *registry* resolve. `:2213`: se `w4a8_codebook_linear_chunked` devolver `used=False`, `xq`/`xs` (alocados com `torch.empty`, `:2155-2156`) chegam ao GEMM com lixo — o ramo irmão `:2188-2212` prova a intenção ao chamar `quantize_int8_rowwise_convrot` antes. | Leitura de `:2150-2265`. O `:2213` é confirmável só por leitura do fluxo de controle; o `:2261` idem. Reprodução exige shape que o CUTLASS recuse. **[GPU]** |
| 7 | `tools/verify_w4a4.py:246` e `:154` — `--kernel-smoke` não pode reprovar nada; o payload quantizado nunca é verificado | Não existe branch que retorne 1 depois do smoke: backend eager, RMSE nan, tudo sai `exit 0` com "Structural verification: PASS". E `validate_structure`/`validate_preserved_bytes` iteram só `source_header`, pulando os tensores quantizados — pesos I8 todos zero e escalas 0.0 passam. | CPU, já demonstrado pelo auditor: montar safetensors sintético com `a.weight` I8 zerado + `a.weight_scale` F32 zero → as duas funções devolvem `[]`. Repetir com o arquivo truncado em 528 bytes. Ler `:240-252` para o exit code. |
| 8 | `tools/quant_w4a4_smooth.py:173` — não roda preflight de backend nenhum | Produz exatamente o mesmo formato (`convrot_w4a4`) que `quant_w4a4.py` recusa produzir sem CUDA, e o eager gera saída **estruturalmente idêntica** (`_pack_int4_row_major` + scales `[rows]` f32). Nem `verify_w4a4` nem `inspect_quant` distinguem, e o sidecar não grava o campo `backend` que `quant_w4a4.py:416` grava. | `grep -n "normal_comfy_backend" tools/*.py` → confirmado: aparece em `quant_mixed.py`, `quant_w4a4.py`, `quant_w4a8.py`, `verify_w4a4.py`. **Ausente** em `quant_w4a4_smooth.py` e `quant_int8.py`. |
| 9 | `ComfyUI/comfy/ops.py:1377` — qualquer LoRA / `weight_function` derruba a camada para BF16 | `ModelPatcher` popula `weight_function` (`model_patcher.py:1938`); com ela presente `cast_bias_weight` entra no ramo `:431`, chama `dequantize_convrot_w4a4_weight` (desfaz a rotação Hadamard) e faz `F.linear` BF16. Sem uma linha de log. | CPU: setar `lin.weight_function = [lambda t: t]` na `Linear` do item 1 e checar que `cast_bias_weight` devolve `Tensor`. |
| 10 | `ComfyUI/comfy/model_detection.py:1512` — `convert_diffusers_mmdit` descarta ou renomeia errado as chaves de quantização | Os mapas de `comfy/utils.py` só contêm `.weight`/`.bias`; `.comfy_quant`, `.weight_scale`, `.weight_s_rel` etc. somem para Flux/SD3/PixArt/AuraFlow. O único passthrough (Z-Image, `:1512-1514`) copia a chave com o **nome diffusers** enquanto o peso é renomeado — marcador órfão exatamente nas camadas que o mapa renomeia. Esta é a família do silent-failure de naming diffusers que já mordeu hoje. | CPU: injetar `context_refiner.0.attention.to_q.comfy_quant` no header real de `beyond-reality-zimage-v2_bf16.safetensors`, rodar `convert_diffusers_mmdit(sd,'')`, e checar se existe `context_refiner.0.attention.qkv.comfy_quant` na saída. |
| 11 | `comfy_kitchen/backends/cuda/__init__.py:300` — W4A4 vira W4A8 fora de sm8x, mas o registry anuncia a partir de sm75 | `_cuda_device_supports_native_int4_mma` retorna `major == 8` (igualdade exata): em sm75/sm90/sm100 a ativação passa a ter 8 bits. `FunctionConstraints` declara `min_compute_capability=(7,5)`, então o preflight aprova igual em todas. Um número de qualidade medido em outra placa não transfere. | Leitura de `:293-300` + `:3537-3558`. Cruzável com `_check_accel.py` para a CC local. A consequência (RMSE diferente por placa) é **[GPU]** e só numa placa não-sm8x. |
| 12 | `tools/svdq_to_bf16.py:457` — `zero_leak > 1e-3` falha aberta em NaN | Em Python, `float('nan') > 1e-3` é `False`. Para `index >= --verify` (128 das 136 camadas do Z-Image com o default 8) essa é a **única** checagem; `torch.isfinite` só roda dentro de `check_recovery`, que só roda na amostra. Escreve 11,2 GiB e sai 0. | Uma linha no interpretador embutido: `.\python_embeded\python.exe -s -c "print(float('nan') > 1e-3)"`. O resto é leitura de `:446-460`. |
| 13 | Guard de RAM `largest*3 + 2 GiB` copiado para desenhos acumuladores — **5 arquivos** | A heurística é correta só em `quant_w4a4.py` (streaming, um tensor por vez). `quant_int8.py:148` exige 2,375 GiB e acumula 19,144 GiB no LTX-2.5 (8x); `quant_w4a8.py:257` exige 2,375 e acumula 10,777 (4,5x); `quant_mixed.py:413` exige 2,25 e acumula 2,92 já no **menor** modelo do projeto; `quant_w4a4_smooth.py:176` e `svdq_to_bf16.py:522` não checam nada (5,084 GiB e pico de 14,57 GiB). Terceiro silent-failure de hoje foi exatamente "guard de VRAM lendo o número errado". | CPU, sem carregar tensor: parsear o header da fonte e somar `N*K + N*4` (int8) / `N*(K//2) + N*(K//group) + N*4 + 64` (w4a8) sobre as camadas que o profile seleciona; comparar com `largest*3 + 2 GiB`. |
| 14 | `tools/nunchaku_compare.py:646` + `:501` — `quant_modules` nunca aparece no relatório; handle NVML indexado por ordinal CUDA | O contador que existe para provar que o loader não dequantizou fica só no stdout de um processo já terminado — `report()` (`:215-321`) nunca lê `quant_modules` nem `loader`. Separadamente, `nvmlDeviceGetHandleByIndex(torch.cuda.current_device())` mistura dois espaços de numeração: sob `CUDA_VISIBLE_DEVICES=1` mede a VRAM do 3090 enquanto a carga roda no 3080 Ti. | Leitura: `grep -n "quant_modules" tools/nunchaku_compare.py` mostra escrita em `:646` e nenhuma leitura em `report()`. O NVML é confirmável por leitura de `:97` e `:501`; a divergência real de índices exige rodar com `CUDA_VISIBLE_DEVICES` **[GPU]**. |
| 15 | `tools/gpu_lock.py:63` + `tools/m_crossover.py:91` — o lock não protege quem precisa e apaga o lock alheio | `grep -rl "GpuLock\|gpu_lock" tools/` retorna **apenas** `gpu_lock.py` e `w4a4_breakdown.py` — confirmado. O exemplo da docstring (`:12`) nomeia `m_crossover`, que nunca importa o módulo. E `__exit__` faz `unlink()` sem comparar pid/owner, enquanto a mensagem de busy (`:55-58`) ensina o usuário a apagar o arquivo à mão. | Já confirmado: `grep -rln "gpu_lock\|GpuLock" tools/` → 2 arquivos. Leitura de `:42-70` para o `__exit__`. |
| 16 | `tools/test_svdq_verify.py:100` — a suíte fica 100% verde com w3/w1 trocado, `.T` removido e q/k/v invertido | `recover_weight` (`svdq_to_bf16.py:198`) **nunca** é chamada: o teste reimplementa a sonda em `_recover` com mecânica diferente. `split_fused` (`:249`), cuja própria docstring diz que a ordem errada muda o erro relativo de 0,0995 para 1,4199, não tem teste nenhum. O auditor rodou as três mutações simultâneas e obteve PASS/exit 0. | CPU, reprodutível: copiar `svdq_to_bf16.py` para o scratchpad, aplicar as 3 mutações, rodar `.\python_embeded\python.exe -s tools\test_svdq_verify.py` contra o módulo mutado. |
| 17 | Probe de backend nativo em **8 cópias divergentes**, e 6 cópias de `instrument()` | Três definições diferentes de "nativo pronto": `verify_w4a4.py:64-73` só `convrot_w4a4_linear`; `quant_w4a4.py:93-103` exige os dois ops; `quant_audit.py:386-409` só o quantizador. Três agulhas diferentes: `"comfy_kitchen.backends.cuda"` (m_crossover, w4a4_breakdown), `".backends.cuda."` com ponto final (quant_audit), `"cuda"` (check_w4a8:65). `quant_mixed.py:97` chama `get_implementation(n)` **sem kwargs**, e `registry.py:246` documenta que kwargs vazio pula a validação de constraints inteira. O CLAUDE.md afirma que "verify_w4a4.py runs the same check" — é falso. | Já confirmado por leitura lado a lado das 4 variantes. `grep -rln "def instrument" tools/` → 6 arquivos (`compile_w4a4_probe`, `convrot_ops_probe`, `diffusion_smoke`, `gemma_chat`, `stage_probe`, `te_smoke` — o auditor citou 5, faltou `te_smoke.py`). |
| 18 | `tools/diffusion_smoke.py:173` — o booleano `native` conta chamadas ao **despachante**, não ao backend | `convrot_w4a4_linear` no `tensor/` é só o dispatcher; contá-lo não distingue cuda de eager, nem W4A4 de W4A8 (`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1` muda o caminho sem mudar o nome do impl). Mesmo defeito em `stage_probe.py:163`, `convrot_ops_probe.py:116`, `compile_w4a4_probe.py:157` e `quality_battery.py:212` — esta última imprime `kernel='native'` na tabela e nunca mostra `kernel_impls`. | Leitura de `:170-176` + `comfy_kitchen/tensor/convrot_w4a4.py:80-110`. Confirmável em CPU: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1` e verificar que o nome do impl reportado não muda. |
| 19 | `tools/core_patch.py:153` — `revert` sobrescreve arquivo atualizado do ComfyUI sem checar hash | O docstring (`:3-4`) promete exatamente essa detecção. Ela existe só em `command_backup` (`:83-89`). O check pós-revert (`:155`) é tautológico: compara o que acabou de restaurar com o hash do próprio backup. Um `revert` depois do update 0.29→0.33 devolve um `comfy/ops.py` de quatro versões atrás e imprime "Reverted". | Leitura de `command_revert` (`:144-158`) — não há `sha256(path)` antes do `shutil.copy2`. |
| 20 | `tools/quant_mixed.py:354` + `:370` + `:246` — `--uncalibrated fail` vira bf16 em silêncio; `--budget` rebaixa camadas nunca medidas; `--analysis` pula toda checagem de coerência | Três buracos no mesmo pipeline de decisão por camada. `'fail'` só é checado dentro de `if analysis is None` (`:290`); a fatia de tamanho fixo `sorted(promoted,key=gain)[:len-allowed]` alcança as chaves `inf` que o próprio comentário diz proteger (simulado: 170 camadas, 12 não calibradas, `--budget 0.05` → rebaixa L0..L3); e no ramo `--analysis` nada compara `source`, `group_size`, `convrot_groupsize` nem `shape` com o checkpoint. | CPU: rodar `--analysis calib/zimage_v2_native.analysis.json --convrot-groupsize 64` sobre a fonte real e ver se sai sem recusa. E simular o `sorted(...)[:n]` com uma lista contendo `inf`. |

---

## 2. Silent failures suspeitos

Este projeto já teve três hoje: **fp16 estourando para inf virando nan**, **naming diffusers fazendo camada carregar sem escala**, **guard de VRAM lendo o número errado**. Os três se repetem literalmente nesta auditoria, o que é o argumento mais forte para não descartar a categoria.

### 2.1 O cluster central: "a garantia de execução nativa não garante nada"

O projeto tem uma cadeia de guardas para provar que o W4A4 roda no kernel CUDA. A auditoria alega que **cada elo da cadeia tem um furo diferente**, e que os furos compõem:

| Elo | Furo alegado | Onde |
|---|---|---|
| Conversor recusa converter sem CUDA | Dois conversores não têm preflight nenhum | `quant_w4a4_smooth.py:173`, `quant_int8.py:98` |
| Preflight verifica os ops certos | `quant_w4a8` só verifica o quantizador, `verify_w4a4` só o linear, `quant_audit` só o quantizador | `quant_w4a8.py:113`, `verify_w4a4.py:73`, `quant_audit.py:386` |
| Preflight valida constraints reais | `quant_mixed` chama sem kwargs → validação pulada por inteiro (`registry.py:246`) | `quant_mixed.py:97` |
| Registry escolhe cuda | Reprovar constraint cai para eager em `logger.debug`; `use_backend('cuda')` também cai | `registry.py:218`, `:202` |
| Implementação cuda executa cuda | `w4a8_int8_linear` chama `eager_w4a8_int8_linear` por dentro quando o CUTLASS recusa | `cuda/__init__.py:2261` |
| Dispatch do QuantizedTensor usa o kernel | Op não registrado → dequantiza tudo, log em DEBUG; peso `transposed` → `F.linear` BF16 | `tensor/base.py:365`, `tensor/convrot_w4a4.py:237` |
| ComfyUI não desliga o caminho | `weight_function` (LoRA) e input dtype ≠ compute dtype dequantizam | `ops.py:1377`, `ops.py:431` |
| verify reprova o que passou errado | `--kernel-smoke` não tem branch de falha; payload quantizado nunca é lido | `verify_w4a4.py:246`, `:154` |
| Benchmark denuncia | `nunchaku_compare` não reporta `quant_modules`; `diffusion_smoke` conta o dispatcher | `nunchaku_compare.py:646`, `diffusion_smoke.py:173` |

**Suposição que sustenta o cluster inteiro** e precisa ser derrubada ou confirmada primeiro: que o backend eager de fato declara as mesmas capabilities e produz saída estruturalmente indistinguível. O auditor cita `backends/eager/__init__.py:37`, `:364` e `backends/eager/convrot_w4a4.py:125/166/174-190`. É uma leitura de 20 minutos e decide o peso de metade desta seção.

### 2.2 Resultado numericamente errado sem erro

- **`ops.py:1373`** — `full_precision_matrix_mult: true` no `comfy_quant` de uma camada é ignorado para os três formatos do projeto. Um conversor que marcou adaLN/proj_out como "roda em BF16" está errado sobre o próprio arquivo.
- **`model_detection.py:1512`** — marcador `.comfy_quant` órfão faz `_load_quantized_module` (`ops.py:1142`) cair em `layer_conf is None` e carregar o container INT8 como se fosse peso. Para `int8_tensorwise` o shape bate, então **não há erro de shape**: os inteiros −127..127 viram BF16 sem escala, ruído ~100x fora de escala.
- **`ops.py:1203`** — `weight_correction` do `asym_w4a8_int8` é gravado no save (`w4a8_int8.py:257-258`) e **nunca lido** no load: o termo de correção assimétrica some e o peso decodifica com viés sistemático. Que o buraco é conhecido está no próprio projeto: `quant_w4a8.py:280` e `quant_mixed.py:443` levantam `SystemExit('symmetric=True returned a correction tensor; ComfyUI would drop it')`.
- **`ops.py:1131`** — `state_dict.pop('weight')` antes do `super_load` desliga a validação de tamanho do torch para toda camada de um modelo MixedPrecisionOps; `QuantizedTensor.dequantize()` fatia `full[:orig_shape]` em silêncio (`tensor/base.py:288-291`), e `weight.to(dtype=storage_t)` sobre um float **trunca** em vez de reinterpretar bits (peso ~0.02 → 0).
- **`cuda/__init__.py:2213`** — GEMM com `xq`/`xs` vindos de `torch.empty` não escrito.
- **`convrot_w4a4.py:253`** — `addmm` descarta `beta`/`alpha` e aceita addend 2-D como se fosse bias vetorial. O layout irmão `w4a8_int8.py:355-358` tem exatamente esse guard; das 4 cópias do handler (`convrot_w4a4:253`, `svdquant_w4a4:576`, `awq_w4a16:201`, `w4a8_int8:352`) só uma foi endurecida.
- **`svdq_to_bf16.py:408`** — `--limit` escreve um arquivo com as camadas não recuperadas ainda em INT4 cru, mas remove `quantization_config` e grava `dequantized_from`. Nada no nome nem no log distingue de uma conversão completa.
- **`quant_int8.py:185`** — nenhuma validação de dtype/shape do retorno do quantizador; `TensorWiseINT8Layout.Params._validate_tensor_fields` é literalmente `pass` (`tensor/int8.py:80-81`), então um scale `[N]` 1-D em vez de `[N,1]` faria broadcast ao longo de K em vez de por linha.

### 2.3 Ferramenta que aprova o que devia reprovar

- **`verify_w4a4.py:246`** — nenhum caminho retorna 1 depois do smoke; `relative_rmse`/`max_abs_error` nunca são comparados com nada; a implementação é resolvida (`:203`) e depois chamada por um **segundo dispatch independente** (`:204`), então o backend reportado não é necessariamente o executado.
- **`verify_w4a4.py:121`** — a checagem de linhas do `weight_scale` está indentada dentro do `if source_header is not None`. Mover a fonte para outro drive faz a checagem **desaparecer** em vez de degradar — e a fonte Gemma vive em `D:/ComfyUI-Models/`, mount opcional.
- **`verify_w4a4.py:123`** — nada percorre `model_header` procurando tensores que não existem na fonte; um `GARBAGE.leftover` embutido passa PASS/PASS.
- **`check_w4a8.py:76`** — `main()` retorna 0 mesmo com todas as células `nan`.
- **`m_crossover.py:158`** — falha de um caminho só é explicada em `M=1`; nos demais vira `nan` mudo, e a linha 166 escolhe `min()` entre o que sobrou, imprimindo um vencedor sobre um campo incompleto.
- **`attn_dtype_ab.py:132`** — erro medido como `mean(abs(.))` sem checar finitude: um backend que devolve NaN só em bf16 imprime tempos e veredito normais com `nan` na última coluna.
- **`svdq_to_bf16.py:448`** — `--verify 0` remove o único gate real e o resumo não diz que nada foi checado.
- **`precheck_workflows.py:54`** — `widgets_values` em formato dict faz o `for` iterar as **chaves**; o nó passa sem nenhum valor examinado. Confirmado no diretório real: 556 nós com lista, 3 com dict (VHS_*). Ainda não morde, mas `swap_to_nunchaku.py` tocando um workflow VHS já basta.

### 2.4 Medição que mede outra coisa

- **`nunchaku_compare.py:346`** — `load_s` do loader nunchaku inclui `spec.loader.exec_module` do pacote ComfyUI-nunchaku inteiro (init de contexto CUDA, parse de `models.yaml`, imports de flux/qwenimage/zimage + extensão nativa + diffusers/transformers), dentro do cronômetro. O ramo comfy não paga nada disso — os imports de comfy já aconteceram em `main()` fora do cronômetro. Numa sessão real o import acontece no boot.
- **`nunchaku_compare.py:115`** — `peak` é inicializado com o baseline e só sobe; memória liberada durante o run vira desconto até `0.00 GiB`, e `report()` trata `0.0` como falsy, então nem a anotação aparece.
- **`nunchaku_compare.py:125`** — o único guard da subtração de baseline (`other_processes`) é inerte por construção neste host: o próprio comentário (`:118-122`) documenta que o WDDM não atribui memória por processo, então `attributed` fica vazio e `other_processes` é forçado a 0. O aviso de `:512` nunca dispara.
- **`nunchaku_compare.py:621`** — `torch_peak_gib` e `peak_gib` medem janelas de tempo diferentes (a do torch é resetada depois do warm-up, a do NVML inclui load+warm-up) e são impressos lado a lado como "of which torch alloc", com `:693-696` interpretando o gap como interferência externa.
- **`w4a4_breakdown.py:105`** — a coluna `host us` mede sincronização serializada + resíduo do CUPTI, não despacho de host. Os próprios números registrados divergem: mesmo peso 3840x3840, mesma placa, `bf16 M=1` = 0,057 ms no `m_crossover` contra 89,7 us no breakdown; `w4a4 M=1` = 0,122 ms contra 196,5 us. Em `M=128` concordam (97 vs 93 us). Usando o wall limpo do m_crossover, o "host" de M=1 cai de ~161 us para ~87 us — quem implementar CUDA graph esperando o número maior vai medir bem menos.
- **`attn_dtype_ab.py:82`** — `flash_attn` e `xformers` pagam três `.contiguous()` de transposição **dentro** do bloco cronometrado (48 MiB por chamada na forma B=2,H=16,S=4096,D=64); `sdpa`, `sageattention` e `spargeattn` não. Como o custo é igual em fp16 e bf16, ele **encolhe** a penalidade relativa do bf16 — exatamente a hipótese que o arquivo existe para testar.
- **`calibrate_activations.py:329`** — o latente é montado para VAE de imagem 8x (`side = size//8`, shape 4-D sem eixo temporal), mas `--profile` aceita `ltx_2_5` e `hunyuan_video_15`. Para zimage confere e é provável (1024→128→patch 2→4096 tokens, e a calibração gravada mostra 131072/32 = 4096 linhas por chamada em `noise_refiner`).
- **`plot_weight_balance.py:163`** e **`activation_balance.py:85`** — as duas medem crest no tensor **pré-rotação** e atribuem o número ao caminho que roda. A escala real é absmax **por linha do peso rotacionado** (`backends/eager/convrot_w4a4.py:112`). A ferramenta irmã `weight_balance.py:82-100` já calcula as duas coisas separadamente — `plot_weight_balance` reimplementou metade e escolheu a errada. A anotação "the W4A4 scale reaches to here" e a linha de referência "Gaussiana ~ 4" (`:180`, que é a expectativa de uma **linha** de 3072, `sqrt(2·ln 6144)=4,18`, contra crests de matriz inteira onde o valor certo seria `sqrt(2·ln 1,9e7)=5,79`) apoiam uma conclusão causal construída sobre o tensor errado.
- **`quant_audit.py:21`** — INT4 empacotado dois-por-byte em I8 é inventariado como I8. Dado real: `svdq-int4_r32-z-image-turbo.safetensors`, 3,36 GiB, `{BF16: 963, I8: 136}` → `current_precision = "mixed I8/BF16"`. A coluna `Current` do `quantization_inventory.md` — documento que alimenta o ranking do `W4A4_PROGRESS.md` — reporta o dobro dos bits reais.

### 2.5 Afirmação que os dados não sustentam

- **`quant_w4a4.py:423`** — `EXCLUSIONS` **nunca é aplicado**: a única referência no arquivo é o campo `excluded_patterns` do manifesto. A exclusão é efeito colateral do allowlist não casar. Todo `.quant.json` do projeto documenta um mecanismo que não existe, e o CLAUDE.md lista quatro arquiteturas (Flux, Hunyuan, SeedVR2, Z-Image) que ainda precisam de receita própria — quem escrever um `PROFILE_PATTERNS` mais frouxo confiando na rede de segurança não a tem.
- **`check_w4a8.py:45`** — comentário e conclusão citam "per-group scale set by an outlier", mecanismo que o kernel ConvRot **remove** (rotação Hadamard sobre grupos de 256 antes de quantizar, escala **por linha**). Os números registrados confirmam: 3840x3840 gaussiana 0,2230 → pós-ativação 0,2234 (+0,18%); 11520x3840 0,2231 → 0,2233 (+0,09%). Só 3840x10240 se move, e ali o próprio K explica 0,0140 dos 0,0234.
- **`cuda/__init__.py:1932`** — "Real model hidden dims avoid that band anyway" para `5120 < K < 8192`. O inventário desta instalação contém K = 6144, 6912 e 7680.
- **`cuda/__init__.py:1261`** — o ramo chamado de "W4A4" no docstring quantiza a ativação em INT8 cheio. E `quantize_int4_rowwise_convrot64_to_int8` (`:743`), documentada como exatamente a função que preservaria A4 dentro do container INT8, é **código morto** (grep no pacote e em `ComfyUI/comfy`: só a definição e o `__all__`).
- **`svdq_to_bf16.py:54`** — "The probe is exact, not approximate ... with no error at all". A única evidência on-kernel é homogeneidade (`layer(8I)/8 == layer(I)`), que o próprio docstring de `verify()` admite não distinguir nada. O `oscales` do nunchaku é BF16 para INT4, não float32; medido em CPU sobre 200k amax aleatórios, `7*bfloat16(amax/7)` nunca reproduz `amax` (erro relativo médio 0,0014, máximo 0,0039). O teste que travaria isso (`test_identity_probe_is_exact`) roda o `FakeW4A4` em float32, onde o erro estruturalmente não pode aparecer.
- **`calibrate_activations.py:152`** — o docstring de `LayerStats` promete zero sync por chamada ("~7600 calls"), e `reservoir.offer()` faz `.to(device='cpu')` em ~98,3% delas (com os números da calibração real: `layers.0`, 159744 linhas em 32 chamadas, capacidade 128 → `(154752/159744)**128 = 0,017` de chance de não copiar, na **última** chamada).
- **`nunchaku_compare.py:476`** — "The encoded conditioning is saved into the result file ... exactly the same tensors". O que é salvo é `cond_shape` e `cond_norm = round(float(norm), 4)`. E `report()` (`:299-302`) usa isso para afirmar que os dois runs usaram o mesmo condicionamento — enquanto o **negativo** não é gravado nem verificado, e com `cfg > 1` ele participa de cada passo.
- **`ops.py:1666`** — o log "Native ops" imprime `QUANT_ALGOS.keys() - disabled`, e `disabled` só é alimentado por fp8/nvfp4/mxfp8. Não existe `supports_convrot_w4a4_compute` em `model_management.py`. Então `convrot_w4a4`, `asym_w4a8_int8` e `int8_tensorwise` aparecem como "Native ops" em qualquer hardware. `QuantizedLayout.supports_fast_matmul()` existe e é consultado por `MoEExperts._expert_linear_impl` (`ops.py:1531`) mas **nunca** por `Linear.forward` — duas cópias da mesma decisão que discordam.
- **`gpu_lock.py:12`** — a docstring exemplifica `with GpuLock("m_crossover")`, e `m_crossover` é justamente quem não importa o módulo. Confirmado por grep.
- **`quality_battery.py:38`** — dois dos oito predicados reprovam respostas corretas, contradizendo o docstring ("a right answer inside a rambling one counts as right"). Verificado no interpretador embutido: `'The first 8 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19.'` → `False` (o `8` de "first 8" entra no `re.sub(r'[^\d,]','',t)`), e `'3.9 is larger than 3.11.'` → `False` (`'3.11'` sobrevive ao `.replace('3.9','')`). O viés pune verbosidade, que é exatamente o que quantização agressiva muda.

### 2.6 Razão invertida (o erro que já foi corrigido aqui antes)

Dois lugares reintroduzem a razão que "3,41x mais leve" existe para eliminar:

- **`nunchaku_compare.py:287`** — duas linhas depois de usar `row(lower_is_better)` para "seconds / sampling pass", imprime o mesmo fato como razão crua `a/b` sob o rótulo `speedup (A/B)`. B 24% mais lento vira `speedup 0,81x`.
- **`check_w4a8.py:76`** — cabeçalho `W4A8 better by` com `ratio = e4/e8` sem checar o sinal: se o W4A8 for pior, imprime `0,73x` debaixo de um título que já afirma a direção.

---

## 3. Fork do comfy_kitchen

**Fato de custo levantado nesta sessão:** `python_embeded/Lib/site-packages/comfy_kitchen/backends/cuda/` contém apenas `.cuh` (`dtype_dispatch.cuh`, `float_utils.cuh`, `utils.cuh`) — **nenhum `.cu`**. O que está instalado é o `_C.abi3.pyd` compilado. Portanto qualquer item da coluna "exige `.cu`" pressupõe **obter o fonte upstream do comfy-kitchen 0.2.23 e montar uma toolchain CUDA 13 no Windows compatível com torch 2.13+cu130**. Isso não é "editar um arquivo": é um build paralelo, e trocar o `.pyd` instalado colide com a regra de não mass-upgrade. Nenhum ganho abaixo justifica isso sozinho; só o conjunto justifica, e só depois de medido.

### 3.1 Só Python — sem recompilar nada

Custos alegados por chamada, medidos pelos auditores neste interpretador (CPU, só dispatcher) contra um total de 109 us de CPU por forward W4A4 em M=1:

| Mudança | Onde | Ganho alegado | Risco |
|---|---|---|---|
| Resolver a implementação uma vez no load, com `backend='cuda'` explícito | `tensor/convrot_w4a4.py:92` | 6,69 us/chamada (a construção do dict é 0,16 us; o resto é `validate_function_call` iterando 7 `ParamConstraint`) | Baixo. Mata o overhead **e** o fallback silencioso do item 5 da tabela, porque `get_implementation(..., backend='cuda')` levanta `NoCapableBackendError` (`registry.py:266`) |
| Usar `_int8_weight_scale_arg` (que já existe, `:474-478`, e já é usado nos 3 caminhos INT8) nas escalas do int4 | `cuda/__init__.py:1096-1097` | 3,472 us → 0,477 us, x2 escalas ≈ 6 us/chamada | Muito baixo. Uma linha cada. É duplicação divergente dentro do mesmo arquivo |
| Fast-path 2D: pular 5 dos 7 `reshape` e o slice final | `cuda/__init__.py:723`, `740`, `1096-1097`, `1251`, `1315` | ~12-14 us dos 109 us (`reshape(-1,K)` em 2D contíguo medido em 2,228 us; `out[:m]` em 2,607 us; bate com 17us/7 = 2,43 us do profile) | Baixo, mas precisa de teste: o slice `out[: x2d.shape[0]]` é no-op **garantido só no caminho nativo** |
| Pool de scratch para `q_2d`/`scales_2d` (2 das 3 alocações não escapam) | `cuda/__init__.py:728-729` | ~6-7 us/chamada (`torch.empty((1,2048), int8)` medido em 2,145 us) | Médio. Precisa ser keyed por `(device, dtype, m, k, stream_ptr)` para não aliasar entre streams no MultiGPU. **Nunca poolar `output`** |
| Mover `torch.empty` da 1095 para depois dos dois early-returns | `cuda/__init__.py:1095` | Elimina churn de `[M,N]` em out_dtype em Turing/Ada-fallback/Hopper (8 MB por camada por forward para 1024x4096 bf16) | Nenhum no sm86 (lá o buffer é usado). É correção de código morto para as outras placas |
| `w4a8_int8_linear` tentar `quantize_int8_rowwise_convrot64` antes do `convrot` | `cuda/__init__.py:2189`, `:2216` | Desconhecido — a evidência é que os **outros dois** call sites preferem o convrot64, e 28% do tempo em M=5856 é quantização de ativação | **[GPU]** para medir. Confiança do auditor: média |
| Preencher `_prefer_legacy_int4_kernel` com valores medidos no 3090 | `cuda/__init__.py:303-317` | Hoje o threshold é `base * SM_count // 142`, extrapolado da L40S/RTX 6000 Ada. No 3090 (82 SM) dá 454.136 para k>1024; no Gemma 3 12B `gate/up_proj` (n=15360) `m_tile*n = 491.520` já em M=1, então o decode de token único usa o GEMM CUTLASS — na placa de referência usaria o kernel a mão | **[GPU]**. 8 linhas de Python, zero recompilação. É o ganho mais barato do fork *se* a varredura confirmar |
| Preencher `_prefer_cublas_int8_fallback` (hoje `return False` com 4 parâmetros ignorados) | `cuda/__init__.py:480-486` | Gancho de tuning pronto; `COMFY_KITCHEN_DISABLE_CUTLASS=1` (`:1525`) já permite forçar o outro caminho para comparar | **[GPU]** |
| Corrigir `_int4_int8_weight_chunk_cols` (dois ramos idênticos, `m` não afeta nada) | `cuda/__init__.py:818-823` | Memória e reuso de L2: com n=15360 e m=5856 o acumulador é ~91 MiB por chamada | **[GPU]** para escolher o valor |
| Guards de correção em Python | `convrot_w4a4.py:253` (beta/alpha/addend 2-D), `eager/convrot_w4a4.py:166` (acumular em fp32), `:184` (`raise` em vez de ignorar `linear_dtype='int8'`), `cuda:3547` (estreitar constraint de `wscales` para float32), `base.py:365` (WARNING + modo estrito por env) | Não é performance: converte degradação silenciosa em erro alto | Baixo. É a maior parte do valor do fork |
| Embrulhar os 5 argumentos em `_wrap_for_dlpack` | `cuda/__init__.py:979-987` | Único dos 35 call sites de `_C.*` sem o wrapper; a **mesma** função é chamada corretamente em `:2052` | Nenhum aqui (caminho só alcançável em sm75) |

### 3.2 Exige `.cu` novo — e portanto build upstream

| Mudança | Ganho alegado | Custo honesto |
|---|---|---|
| Fundir quantização de ativação no prólogo do GEMM (2 launches → 1) | 59,5 us de `cudaLaunchKernel` num forward cujo trabalho de GPU em M=1 dura muito menos. Não há símbolo em `_C` que faça rotação+quantização+GEMM junto (o auditor levantou os 44 símbolos referenciados; existe `int8_linear_m1`, **não** existe `int4_linear_m1`) | Kernel novo: prólogo lendo x em BF16, FHT por grupo de 256, quantização em registrador/shared antes do MMA. É o inverso do que o SVDQuant faz. Semanas, não dias |
| Hoistar `cudaFuncSetAttribute` para `once_flag` por (kernel, device) | 1,6 us de host por forward de camada quantizada; ~6,4 ms de CPU por geração a 200 lineares × 20 steps | Uma linha no lançador — **se** você tiver o lançador. Confiança do auditor sobre qual dos 2 kernels chama: média |
| Instanciação `m16n8k64 s4` para sm90+ | Faz W4A4 ser W4A4 fora de sm8x (hoje `major == 8` exato, item 11 da tabela) | Só vale se houver hardware sm90+ no horizonte. Neste host, nenhum |

### 3.3 O meio-termo: CUDA Graph

Sem tocar em CUDA: capturar a sequência da camada em graph. No replay não há chamada de API de host, então `cudaFuncSetAttribute` e os 59,5 us de `cudaLaunchKernel` somem juntos, e boa parte do overhead Python também. O código já é capture-safe por construção — `_wrap_for_dlpack` usa `stream=-1` exatamente para não quebrar captura fora do default stream (`cuda/__init__.py:489-505`).

Custo real: exige shapes estáticas (decode M=1 é o caso ideal) e buffers de entrada/saída persistentes — o que **amarra com o item de pooling** e reintroduz o risco de uma camada sobrescrever a saída ainda em uso por outra. E a expectativa de ganho precisa ser recalibrada antes de começar: veja `w4a4_breakdown.py:105` na seção 2.4 — o número de "host" que motivaria esse trabalho está inflado por ~2x segundo os próprios logs do projeto.

---

## 4. Lixo

| Arquivo | Razão | Verificado? |
|---|---|---|
| `tools/compile_w4a4_probe.py`, `tools/compile_w4a4_probe2.py` | A pergunta que fazem ("torch.compile sobrevive a um QuantizedTensor ConvRot?") está respondida no docstring de `compile_w4a4_fix_poc.py:3-6` e o resultado está registrado em `W4A4_PROGRESS.md` ("torch.compile works, max_abs_diff 0.000000", re-verificado sob torch 2.13). Rodar `probe2.py` hoje imprime `[FAIL]` por backend e contradiz o resultado já medido. As três duplicam `read_header`/`load_tensor`/`build_module`/`DTYPES` quase byte a byte | `grep -n "compile_w4a4_probe" W4A4_PROGRESS.md W4A4_HANDOFF.md` → **zero ocorrências**. Nada os cita. Apagar é seguro por esse critério |
| `quant_w4a4.py:62` — flag `--auto-detect` | `args.auto_detect` nunca é lido; a detecção só olha `args.profile == "auto"`, cujo default já é `auto`. No-op no caso comum, mentira quando combinado com `--profile gemma` | Leitura + grep no arquivo |
| `quant_w4a8.py:177` — `as_bytes()` | Escrita exatamente para bfloat16 (`.view(torch.int16)`) e float8_e5m2 (`.view(torch.uint8)`); **zero chamadas** em `tools/*.py`. O caminho real (`:320` + `:347`) cobre só float8_e4m3fn. Ou usar, ou apagar — mas apagar deixa o buraco de bf16/e5m2 aberto, então a correção é usar | Grep |
| `cuda/__init__.py:743` — `quantize_int4_rowwise_convrot64_to_int8` | Código morto (grep no pacote e em `ComfyUI/comfy`: só a definição e o `__all__`). **Não apagar**: é a função que preservaria A4 no ramo de fallback que hoje faz A8 (`:1261`). É lixo no sentido inverso — deveria estar sendo chamada | Grep |
| `inspect_quant.py` | Quarta cópia do parser de header safetensors (as outras: `verify_w4a4.py:31`, `quant_w4a4.py:106`, `quant_audit.py:135`), e a única sem validação de `header_size` — só `quant_audit:141` rejeita `<=2` ou `> min(filesize-8, 1 GiB)`. Além disso a heurística de `quant_keys` inclui a substring `offset`, que não aparece em nome de tensor (aparece em `data_offsets` do header), e falta guarda em `sys.argv[1]` | Confirmado por leitura das 4 cópias |
| `cuda/__init__.py:480-486` — `_prefer_cublas_int8_fallback` | Gancho morto (`return False`, 4 parâmetros ignorados), consultado em dois pontos quentes. **Não apagar**: preencher (ver 3.1) | Leitura |
| `cuda/__init__.py:818-823` — ramo `m <= 128` de `_int4_int8_weight_chunk_cols` | Dois `return` literalmente idênticos; o parâmetro `m` não afeta o resultado. Alguém pretendia diferenciar e a distinção se perdeu | Leitura |
| `model_audit.py:52` — `PROJECT_OUTPUT` | Não conhece os sufixos que os conversores atuais produzem. `QUANT_MARKERS`, no **mesmo arquivo** (`:43-49`), já lista `_w4a8_mixed`, `_int8_convrot`, `_mixed`, `_convrot`. E `quant_mixed.py:217` escreve `{stem}_mixed.safetensors`. Resultado: saída do projeto cai na categoria "maiores arquivos que nenhum workflow menciona" (a lista que o usuário lê procurando o que apagar) em vez da categoria "regenerável" | Leitura das duas listas |
| `convrot_ops_probe.py:132` — rótulos | Citam `sd1_clip.py:114` e `sd.py:262`, linhas da 0.29.0. Na 0.33.0 instalada os alvos são `sd.py:269`, `sd1_clip.py:213` e `ops.py:431-434`. A ferramenta continua rodando; só os rótulos mentem sobre onde olhar | Leitura cruzada com o checkout atual |

Duplicação estrutural que sustenta metade da seção 4, e que é achado por si:

- **Probe de backend nativo: 8 cópias, 3 definições de "pronto", 3 agulhas diferentes.** Confirmado por leitura lado a lado. Consolidar em `tools/_convrot_backend.py`.
- **`instrument()`: 6 cópias** (`compile_w4a4_probe`, `convrot_ops_probe`, `diffusion_smoke`, `gemma_chat`, `stage_probe`, `te_smoke` — o auditor citou 5; `te_smoke.py:65` é a sexta, e assim como `gemma_chat.py:84` para de sondar quando `len(impls)` chega a 4). Já divergiram: `gemma_chat.py:106` usa `arg_names` com `'qweight'` onde a assinatura real é `qdata` (`tensor/w4a8_int8.py:93-96`), e `constraints.py:186-187` retorna `ok()` quando o valor é `None` — ou seja a constraint que decide o backend nunca é avaliada nesse probe. `diffusion_smoke.py:78` tem o nome certo.
- **Parser de header safetensors: 4 cópias**, 1 com validação de bounds.
- **Guard de `.comfy_quant` inline: 4 cópias, 2 com o guard.** `quant_w4a8.py:220-223` e `quant_int8.py:119-122` têm; `quant_w4a4.py:346` e `quant_w4a4_smooth.py:148` não. O cenário é real: `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` tem metadata `['gemma_config','format']` (sem `_quantization_metadata`) e 328 marcadores `.comfy_quant` inline. Hoje a sorte segura (0 camadas selecionadas, com a mensagem enganosa "Profile 'gemma' selected no compatible layers"); a sorte acaba num checkpoint parcialmente quantizado — o que `quant_mixed.py` produz.
- **Guard NVML: 2 cópias em 4 instrumentos, ambas fail-open.** Em `m_crossover.py:91-106` o teste do teto fica **fora** do `try`, então uma exceção depois de `nvmlInit()` vaza o handle (nunca chama `nvmlShutdown`) e o script segue como se a placa estivesse vazia. Em `w4a4_breakdown.py:130-140` o teste fica dentro. `check_w4a8.py` e `attn_dtype_ab.py` não têm cópia nenhuma.
- **Handler `addmm` de layout: 4 cópias, 1 endurecida** (`w4a8_int8.py:355-358`).
- **Ordem de concatenação do SVDQ em 2 tabelas** (`svdq_to_bf16.py:160` e `:265`): a que escreve o arquivo é a que nunca é verificada — `match_reference` só roda com `--reference` (opcional) e compara a matriz **fundida**, antes do split.

---

## 5. Achados de baixa confiança

Separados porque o próprio auditor marcou `confianca: media` ou registrou explicitamente o que não conseguiu demonstrar. Não entram na fila de conserto até alguém fechar a lacuna nomeada.

| Achado | O que falta para virar fato |
|---|---|
| `svdq_to_bf16.py:54` — "a sonda é exata, sem erro nenhum" é falso por ~1e-3 relativo (arredondamento BF16 da escala de ativação) | O auditor mediu em CPU que `7*bfloat16(amax/7)` nunca reproduz `amax` (erro médio 0,0014, máx 0,0039), mas isso pressupõe que o `oscales` do nunchaku para INT4 é BF16 — leitura de `ops/quantize.py` e `models/linear.py`, não execução. **Confirmação definitiva pede GPU**: quantizar uma W BF16 conhecida com o próprio quantizador do nunchaku e comparar o `recovered` contra o dequant em fp32 |
| `svdq_to_bf16.py:267` — guards de shape do `split_fused` não verificam a fusão | Os dois cenários (GQA com q=2048/k=512/v=512 dando out=3072 divisível por 3; `FeedForward` diffusers com `activation_fn='gelu'` tendo `net.0.proj` como projeção única) são construídos, não observados. Nos arquivos Z-Image daqui as shapes conferem (`to_qkv` out=11520=3·3840, `net.0.proj` out=20480=2·10240) — o que significa que hoje **não morde** |
| `svdq_to_bf16.py:265` — bias fundido fica com o nome antigo no passthrough | O próprio auditor mediu: os 136 stems quantizados dos dois arquivos Z-Image aqui têm **zero** bias. O Qwen-Image tem `transformer_blocks.0.attn.to_qkv.bias` [9216], mas escapa porque o padrão testado é `attention.to_qkv` e o Qwen usa `attn.to_qkv`. Latente, não ativo |
| `svdq_to_bf16.py:252` — dtype BF16 hardcoded para as norms | Confirmado que num Gemma 12B real as norms são BF16, então hoje coincide. O cenário (Gemma-3 com norms F32, comum em exports diretos do HF) não foi observado num arquivo desta máquina |
| `quant_w4a8.py:177` / `quant_int8.py:185` — `as_bytes` morto e falta de validação do retorno do quantizador | O mecanismo está provado por leitura, mas o gatilho é uma mudança futura do comfy-kitchen (devolver `s_channel` em bf16, ou scale `[N]` 1-D). Hoje funciona por acordo tácito: `quantize_int8_rowwise` devolve scale `[N,1]`, e `int8_linear` faz `weight_scale.reshape(-1)` (`eager/quantization.py:1005-1010`). Nada no conversor amarra esse contrato |
| `precheck_workflows.py:50` — heurística de UUID isenta node types com hífen | Levantados os 131 node types dos workflows atuais: os 15 com hífen são **todos** UUID hoje. Latente |
| `eager/convrot_w4a4.py:166` — acumular o produto INT4 em out_dtype pode estourar fp16 | Com K=15360 e valores em [−7,7] a soma pode chegar a 752.640 contra 65.504 de máximo em fp16. Mas isso exige correlação acima do típico **após** a rotação de Hadamard, e o auditor não conseguiu demonstrar numericamente sem GPU. Mesmo assim, é o padrão exato do primeiro silent-failure de hoje |
| `tensor/convrot_w4a4.py:237` — quem transpõe? | O mecanismo está provado por leitura (`_handle_convrot_w4a4_t:194-206` liga a flag; `:237` dequantiza). O que **não** foi localizado é o chamador concreto que transpõe neste checkout. Um auditor marcou média por isso; o outro marcou alta olhando só o mecanismo. Resolver com o monkeypatch-contador antes de escalar |
| `registry.py:31` — `_compute_capability` é `cached_property` global sobre `current_device()` | As duas GPUs deste host são 8.6, então não morde hoje. O risco é MultiGPU heterogêneo — e o projeto tem `Video-LTX2_MultiGPU.app.json` |
| `cuda/__init__.py:979` — único call site sem `_wrap_for_dlpack` | Só alcançável em sm75. Não reproduzível neste host (3090/3080 Ti são sm86) |
| `cuda/__init__.py:2216` — `w4a8_int8_linear` nunca tenta o `convrot64` | A evidência é ordem de preferência nos outros dois call sites, não medição. "Se o convrot64 não fosse mais rápido, os outros dois não o teriam como primeira escolha" é inferência sobre intenção do autor |
| `cuda/__init__.py:731` — `cudaFuncSetAttribute` por forward | Alta confiança em que não é corrigível em Python (os 44 símbolos de `_C` são todos lançadores, nenhum é entry point de configuração). Média sobre **qual** dos 2 kernels chama |
| `m_crossover.py:59` — ordem crescente de M mede o regime pequeno com a placa fria | Hipótese sem contraprova: ninguém rodou a varredura em ordem invertida. O mesmo padrão está em `attn_dtype_ab.py:129` (fp16 sempre antes de bf16), e o docstring daquele arquivo (`:54-57`) já registra um artefato de ordem/estado diagnosticado como pressão de alocador — também sem contraprova invertida. **[GPU]** para fechar |
| `nunchaku_compare.py:115` — `peak` só sobe, memória liberada vira desconto até 0.00 | O mecanismo (`self.baseline = self.peak = self._used()`, sem rastreio de mínimo) é leitura direta. A premissa contestada é se o CLIP realmente ainda está residente no baseline: `:492` faz `del clip` + `empty_cache` sem `gc.collect()`, e o grafo do ComfyUI tem ciclos (`model_management.py:796` usa `weakref.finalize`). Não medido |
| `nunchaku_compare.py:446` — `--attention sage` é o único backend sem verificação | Confirmado que os ramos `sparge` (`:412`) e `flash` (`:450`) checam disponibilidade e retornam 1, e o `sage` (`:446`) só imprime texto. O que é hipótese é o cenário: `attention.py:646` desvia para `attention_pytorch` com mask, e `:676-680` captura exceção por chamada. Não observado acontecendo |
| `calibrate_activations.py:175` — crest medido num prefixo posicional fixo | Alta confiança no mecanismo (`head = magnitude[:crest_rows]`, sempre as mesmas 64 primeiras linhas), média na magnitude: o auditor não conseguiu separar as duas populações (caption vs imagem) só pelos agregados gravados. Não afeta decisão de formato — `quant_mixed` só imprime `crest_p99`, nunca decide com ele. Afeta a correlação de posto +0,10 citada no docstring |
| `ops.py:1530` — `MoEExperts` quantiza a ativação com o layout de **peso** | O caminho está claro no código (`from_float` chama `quantize()`, escrita para pesos com rotação offline; depois `_handle_convrot_w4a4_linear` dequantiza a entrada). Não foi executado. Precisa de um checkpoint MoE em `convrot_w4a4`, que este projeto não tem |
| `ops.py:1576` — `Embedding` aceita só fp8 e int8_tensorwise | Os conversores deste projeto excluem embeddings por profile (`quant_w4a4.py:36`), então só dispara com checkpoint de terceiros |
| `ops.py:1131` — `pop` do peso desliga a validação de shape do torch | Mecanismo provado por leitura; o gatilho exige metadata inconsistente com os tensores (`convert_old_quants`, `utils.py:1457-1459`, injeta marcador para todo layer do `_quantization_metadata` sem verificar se o peso existe ou já está quantizado) |
| `model_patcher.py:968` — orçamento de offload conta buffer BF16 completo para peso quantizado | O mecanismo (`QuantizedTensor.numel()` é o numel lógico) é leitura direta, e daria 4x para convrot_w4a4. O impacto sobre a ordenação depende de como `_load_list` consome `module_offload_mem`, que não foi executado |
| `attn_bench.py:77` — compara Sage/Flash contra um SDPA que pode ser o próprio Flash | O mecanismo é certo (`F.scaled_dot_product_attention` sem `sdpa_kernel` fixo). O que é hipótese é **em quais** das quatro shapes o torch de fato despacha para flash. Além disso o arquivo roda só em fp16 e com head_dim 64, enquanto o HV15 instalado é 16 heads / head_dim 128 |
| `check_w4a8.py:65` — agulha `"cuda"` mais frouxa que a dos irmãos | Confirmado por leitura das 3 variantes. Hoje nenhum backend alternativo tem `cuda` no caminho, então o guard funciona. O cenário (`comfy_kitchen.backends.cuda_dequant_fallback`) é construído |
| `precheck_workflows.py:28` / `:50` | `known_files` como set global é confirmado por leitura; o cenário concreto (VAELoader apontando para nome de checkpoint) não foi observado num workflow real desta instalação |

---

## Apêndice: coisas confirmadas nesta sessão (não são achados, são fatos de contexto)

- `F:/GPU_BENCH.lock` existe agora. Nenhum item **[GPU]** pode avançar até liberar.
- `grep -rln "gpu_lock\|GpuLock" tools/` → `gpu_lock.py`, `w4a4_breakdown.py`. Nada mais.
- `grep -rln "normal_comfy_backend" tools/` → `quant_mixed.py`, `quant_w4a4.py`, `quant_w4a8.py`, `verify_w4a4.py`. `quant_audit.py` tem probe próprio, com agulha `".backends.cuda."` e só o quantizador.
- `grep -rln "def instrument" tools/` → 6 arquivos, não 5.
- `comfy_kitchen/backends/cuda/` tem só `.cuh`; não há `.cu` para forkar localmente.
- `grep -n "compile_w4a4_probe" W4A4_PROGRESS.md W4A4_HANDOFF.md` → nenhuma ocorrência.

---

## 6. Correções aplicadas — 2026-08-18, sem GPU

Ordem pedida: severidade fraca → grave. Tudo verificado por leitura ou teste de CPU; nenhum item
que precise de GPU foi tocado.

### Módulos novos, que matam duplicação em vez de corrigir cópia por cópia

| Arquivo | Substitui |
|---|---|
| `tools/_bench_guard.py` | as 2 cópias divergentes do guard NVML espalhadas por 4 instrumentos |
| `tools/_ram_guard.py` | o `largest*3 + 2 GiB` copiado para 3 desenhos que acumulam |

O `_bench_guard` agora **falha fechado**: sem NVML ele recusa em vez de assumir placa ociosa.
Testado simulando `ImportError` em `pynvml`. E os quatro instrumentos passaram a pegar o lock —
antes só `w4a4_breakdown` pegava, apesar de a docstring do `gpu_lock` usar `m_crossover` como
exemplo.

### Fraca

- **Razão invertida em 2 lugares** (`nunchaku_compare.py` `speedup (A/B)`, `check_w4a8.py`
  `W4A8 better by`). Os dois afirmavam a direção no rótulo e imprimiam a razão crua.
- **`m_crossover`**: falha de caminho só era impressa em `M=1`; virava `nan` mudo e o veredito
  ainda nomeava vencedor. Agora reporta em todo M e marca `(de N)` quando o campo perdeu entrante.
- **`attn_dtype_ab`**: sem checagem de finitude (um backend que quebra só em bf16 imprimia veredito
  normal com `nan`), e os três `.contiguous()` do flash/xformers estavam **dentro** do bloco
  cronometrado — custo idêntico nos dois dtypes, que encolhia justamente a penalidade que o
  arquivo mede. Hoisted para fora.
- **`inspect_quant.py`**: sem validação de `header_size` (única das 4 cópias sem), sem guarda de
  `argv`, e a agulha `offset` que só pode dar falso positivo (aparece em `data_offsets`, campo do
  header, nunca em nome de tensor).
- **`quality_battery`**: dois predicados reprovavam resposta correta. `"The first 8 prime numbers
  are: 2, 3, 5, 7..."` virava `8,2,3,5,7,...` e falhava o `startswith`; `"3.9 is larger than
  3.11."` falhava porque o `3.11` comparado sobrevivia ao replace. O viés punia verbosidade, que
  é o que quantização agressiva muda. Testados 7 casos em CPU.
- **`precheck_workflows`**: `widgets_values` em dict fazia o `for` iterar as chaves; nó passava sem
  um valor examinado. 3 nós VHS_* no diretório real.
- **`model_audit`**: `PROJECT_OUTPUT` não conhecia `_mixed` nem `_native`, então a saída do
  conversor de hoje caía na lista "arquivos grandes que nenhum workflow menciona".
- **`quant_w4a4 --auto-detect`**: era no-op, e mentira quando combinado com `--profile gemma`.
  Agora força de verdade e avisa quando sobrepõe.
- **`convrot_ops_probe`**: rótulos apontavam para linhas da 0.29.0.
- **Docstrings que afirmavam o que o código não faz**: `calibrate_activations` prometia zero sync
  por chamada (sincroniza em ~98%, calculado com os números da calibração real);
  `svdq_to_bf16` afirmava sonda "exata, sem erro nenhum" (medido em CPU: escala em BF16 dá erro
  relativo médio 0,0014, máx 0,0039, e só 6 de 200000 valores reproduzem exato).
- **`compile_w4a4_probe.py` / `probe2.py`** marcados como superados, **não apagados** — apagar é
  irreversível e não é consertar erro.

### Média

- **`EXCLUSIONS` era código morto** em `quant_w4a4.py`: a única referência era o campo
  `excluded_patterns` do manifesto, então todo `.quant.json` do projeto documenta uma rede que
  nunca existiu. Aplicado. Verificado que **não muda nada hoje** (gemma 336 → 336) e que com um
  perfil frouxo ele barra 2 camadas que o allowlist deixaria passar.
- **Guard `.comfy_quant` inline** faltava em `quant_w4a4.py` e `quant_w4a4_smooth.py`. Cenário
  real nesta máquina: `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` tem 328
  marcadores inline e nenhum `_quantization_metadata`.
- **Guard de RAM**: `largest*3 + 2 GiB` é correto no desenho streaming do `quant_w4a4` e foi
  copiado para três desenhos de duas passadas. Provado em CPU: com 12 GiB livres o guard antigo
  **passa** contra uma acumulação de 19,14 GiB (LTX-2.5 int8). O novo recusa com o número real.
- **`verify_w4a4`**: a checagem de linhas do `weight_scale` estava indentada dentro do
  `if source_header is not None` — mover a fonte para um drive não montado fazia a checagem
  **sumir** em vez de degradar. E nada percorria o header de saída procurando tensor que a fonte
  não tem; um `GARBAGE.leftover` embutido passava PASS/PASS.

### Grave

- **`quant_mixed`, preflight sem kwargs.** `registry.get_implementation`'s own docstring:
  *"kwargs: Kwargs for constraint validation (empty/None skips validation)"*. O probe perguntava
  qual backend seria escolhido **ignorando toda constraint**, enquanto a conversão real passa
  tensores reais onde uma constraint reprovada cai para eager com um `logger.debug`. O guard podia
  passar com o trabalho rodando dequantizado — exatamente o que ele existe para impedir.
  Reescrito com tensores reais para os 4 ops.
- **`quant_mixed`, `--uncalibrated fail` era ignorado** no ramo `--analysis`: virava bf16 em
  silêncio, o oposto do que a flag pede.
- **`quant_mixed`, `--budget` rebaixava camadas nunca medidas.** O `gain=inf` parecia proteger,
  mas o corte é fatia de tamanho fixo e alcança as chaves `inf` quando o orçamento aperta. Agora
  as não medidas ficam fora do ranking e o excesso é avisado em vez de silenciosamente resolvido.
- **`quant_mixed`, `--analysis` sem checagem de coerência.** Testado: análise contra outro modelo
  agora recusa; com `--convrot-groupsize` diferente do medido, recusa; caso coerente segue dando
  115/55, sem regressão.
- **`svdq_to_bf16`, gate de NaN falhava aberto.** `float('nan') > 1e-3` é `False`, e para as 128
  de 136 camadas além de `--verify` esse era o **único** check. Trocado por
  `not (zero_leak <= 1e-3)` mais `torch.isfinite`. Tabela de verdade conferida em CPU: idêntico em
  todo valor finito, barra `nan`.
- **`quant_w4a4_smooth` não tinha preflight nenhum** e escreve o mesmo `convrot_w4a4` que o
  `quant_w4a4` recusa produzir sem CUDA — e a saída do eager é estruturalmente indistinguível.
  Preflight importado de `quant_w4a4` (não uma nona cópia), e o sidecar passou a gravar `backend`.
- **`verify_w4a4 --kernel-smoke` não podia reprovar nada.** Não havia caminho que retornasse 1
  depois do smoke. Agora checa backend CUDA, RMSE finito e um teto de liveness. O teto ficou em
  **0,9, não 1,0**: saída toda zero pontua exatamente 1,0, então 1,0 deixaria passar a falha mais
  óbvia — furo encontrado pelo próprio teste da correção.

### Não corrigido, e por quê

`test_svdq_verify.py` fica verde com 3 mutações (w3/w1 trocado, `.T` removido, q/k/v invertido)
porque reimplementa a sonda em vez de chamar `recover_weight`. Consertar é reescrever a suíte
contra o módulo real, não um ajuste — e a confirmação da mutação `.T` pede GPU.
`quant_audit` contando INT4 empacotado como I8 exige decidir como representar "I8 contendo INT4"
no inventário, que muda o schema do `quantization_inventory.json`.
Todo item marcado **[GPU]** na seção 1 continua aberto.
