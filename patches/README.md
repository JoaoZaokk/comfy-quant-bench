# Patches locais

Mudanças nossas em código que **não** é deste repositório: o checkout do ComfyUI, o `comfy-kitchen` instalado
no `python_embeded` e clones de custom nodes de terceiros. Uma atualização desses códigos (Manager,
`git pull`, `update_comfyui.bat`, `pip install -U`) apaga a mudança sem avisar. O arquivo vivo continua no lugar
original; aqui fica a cópia versionada para reaplicar.

## Conferir

```bash
python_embeded/python.exe -s tools/verifica_patches.py
```

A ferramenta só lê. Para cada patch roda `git apply --check -R` no alvo e informa um de três estados:
- `APLICADO`: o patch está no alvo;
- `AUSENTE (aplica limpo)`: o alvo foi atualizado e o patch sumiu, mas dá para reaplicar;
- `DIVERGENTE`: o código mudou e o patch precisa ser refeito à mão.

Os `comfyui_*` também são testados sozinhos sobre a tag base (`--base v0.37.4`). Rode a ferramenta depois de
qualquer atualização.

## Reaplicar

Rode sempre o `--check` primeiro e, se passar, o mesmo comando sem `--check`. Todos os comandos partem de
`F:\COMFY_PORTABLE`.

| Patch | Alvo | Como aplicar |
|---|---|---|
| `comfyui_*.patch` | `ComfyUI/` | `cd ComfyUI && git apply ../patches/<arquivo>` (os que começam com `From <hash>` também aceitam `git am`, que recria o commit) |
| `comfy_kitchen_*.patch` | `comfy_kitchen` em site-packages | `git apply -p1 --directory=python_embeded/Lib/site-packages patches/<arquivo>` |
| `dasiwa_*.patch` | `ComfyUI-DaSiWa-Nodes` | `git apply --directory=ComfyUI/custom_nodes/ComfyUI-DaSiWa-Nodes patches/<arquivo>` |
| demais nós de terceiros | o próprio clone (tabela abaixo) | `cd ComfyUI/custom_nodes/<clone> && git apply ../../../patches/<arquivo>` |
| `zen_image_edit_*.patch` | Arc (`ssh arc`), `~/ComfyUI/custom_nodes/zen-image-edit-comfyui` | `git apply` dentro do clone, na Arc |

Atenção ao rodar o `comfy_kitchen` de dentro de site-packages: o `git apply` pula os arquivos **em silêncio**,
porque site-packages fica dentro do repositório raiz. Use o comando da tabela.

Ao atualizar um patch, gere-o de novo com `git diff` **pelo bash**. O `Out-File` do PowerShell grava o patch com
CRLF, e ele deixa de aplicar. O `.gitattributes` marca `*.patch -text` para o Git guardar os bytes exatos.

## O que cada um faz

### ComfyUI (base v0.37.4, branch local `local/0.37.4`)
| Patch | Mudança |
|---|---|
| `comfyui_symmetric_patchifier_cpu_scalars.patch` | coordenadas do patchifier do LTX montadas sem tensores no host |
| `comfyui_aimdo_hostbuf_fallback.patch` | cai na cópia normal quando a leitura de arquivo do aimdo para a GPU falha (DiT grande vindo do NAS) |
| `comfyui_awq_w4a16_format.patch` | formato de quant `awq_w4a16` (layout AWQ do comfy-kitchen) |
| `comfyui_text_encoder_quantized_math.patch` | text encoder quantizado roda com o matmul quantizado |
| `comfyui_extra_paths_timeout.patch` | pula, na subida, a seção do `extra_model_paths.yaml` que não responde em 10 s (NAS ocupado) |
| `comfyui_sage_pv_accum_env.patch` | `COMFY_SAGE_PV_ACCUM=fp16+fp32` (ou `fp16`/`fp32`) e `COMFY_SAGE_SMOOTH_K=0/1` escolhem o acumulador P·V e o K-smoothing do SageAttention em SM80/86 chamando o kernel direto; sem a variável nada muda. Os acumuladores fp16 transbordam (imagem preta) quando |V| é grande e a atenção plana: `fp16+fp32` soma 32 chaves em fp16 (inf acima de |V| ≈ 2047, medido), `fp16` soma a linha inteira. O patch escala cada coluna de V por uma potência de 2 (|V| ≤ 1, exato, sem sincronizar) e reescala a saída; `fp16` puro cai em `fp16+fp32` acima de 65 504 chaves. Também deixa de repetir K/V 4× para GQA no caminho Sage (o kernel aceita 12 cabeças de K/V para 48 de Q, saída idêntica). O launcher passa `fp16+fp32`. `COMFY_SAGE_SMOOTH_K=0` (o que o ComfyUI pede e o `sageattn` ignora — reportado em [thu-ml/SageAttention#404](https://github.com/thu-ml/SageAttention/issues/404)) é 15× mais exato no Nunchaku Qwen-Image-Edit e pior no Krea2; o launcher passa 0 desde 02/10 (pedido do dono). `COMFY_SAGE_HYBRID=0.1` (experimento, opt-in): cabeças de K quase constante vão para SDPA exato e o resto fica no Sage — Qwen Edit 30,5 → 33,5 dB e Krea2 25,5 → 26,9 dB do SDPA, +0 a +2 % de custo (`resultado_modelos.md` §12). Medido 2026-10-01/02 em Qwen 2.1, Krea2, Z-Image, LTX 2.5, H3 e Nunchaku (`.scratch/quantfunc_2026-10-01/otimizacao/`, `resultado_modelos.md` §7–8) |
| `comfyui_krea2_fused_norm_gqa.patch` | Krea2: `RMSNorm + modulação` em um kernel `rms_adaln` do comfy-kitchen (peso da norma dobrado na escala), resíduo com gate em `addcmul`, e K/V sem `repeat_interleave` (GQA nativo). KSampler int8 8,1 → 7,5 s, W4A4 5,8 → 5,1 s, edição 24,5 → 23,4 s; muda a imagem tanto quanto trocar a precisão da atenção e mantém a distância ao SDPA (`resultado_modelos.md` §1.3). Aplicar/reverter também por `.scratch/quantfunc_2026-10-01/otimizacao/aplica_krea2_fusao.py` |

### comfy-kitchen 0.2.35
A ordem importa: o `awq` vai por cima do `w4a8`.

| Patch | Mudança |
|---|---|
| `comfy_kitchen_w4a8_dequant_fused.patch` | dequant fundido do W4A8 |
| `comfy_kitchen_awq_w4a16_triton.patch` | dequant AWQ W4A16 no registro de ops (eager + Triton); testes em `tests/test_comfy_kitchen_awq.py` |
| `comfy_kitchen_swiglu_w4a4_fused.patch` | `convrot_w4a4_linear(..., input_act="swiglu")`: SwiGLU fundido no quantizador ConvRot int4 pelo módulo companheiro `_swiglu_quant` (`tools/ck_swiglu`, mesmo kernel do ck carregando `silu(gate)*up`); sem o módulo aplica a ativação e quantiza como antes. Aplicar depois do `awq` |

### Custom nodes de terceiros
| Patch | Clone | Mudança |
|---|---|---|
| `nunchaku_eager_linear_dynamic_vram.patch` | `ComfyUI-nunchaku` | loaders Z-Image/Qwen-Image funcionam com dynamic VRAM; `ZImageModelPatcher` aceita o `fast_disk` do 0.37.4 (medido em 2026-10-01, `.scratch/dynamic_2026-10-01/resultado.md`) |
| `nunchaku_models_qwenimage_preexistente.patch` | `ComfyUI-nunchaku` | mudança anterior em `models/qwenimage.py` (gate/modulação por `timestep_zero_index`), exportada em 2026-10-01 só para preservar |
| `anomalous_model_browser_modal_hidden.patch` | `Anomalous_Model_Browser` | diálogo de configurações escondido de fato. Antes, o frontend achava que havia modal aberto e matava todos os atalhos (Ctrl+Z, Ctrl+S) |
| `dasiwa_system_monitor_skip_bad_drive.patch` | `ComfyUI-DaSiWa-Nodes` | o monitor de sistema pula drive ilegível em vez de falhar |
| `fishspeech_s2wrapper_local.patch` | `ComfyUI-FishSpeechS2Wrapper` | device/compile por variável de ambiente; inclui o arquivo novo `fish_api_server_clamped.py` |
| `ltx2_multigpu_local.patch` | `ComfyUI-LTX2-MultiGPU` | repassa a escolha de device do text encoder ao ComfyUI-MultiGPU (antes ia para cuda:0) |
| `ltxvideo_pyramid_blending_local.patch` | `ComfyUI-LTXVideo` | `pad` trocado por `F.pad` |
| `multigpu_local.patch` | `ComfyUI-MultiGPU` | export DLPack do comfy_kitchen usa o device real do tensor |
| `diffueraser_local.patch` | `ComfyUI_DiffuEraser` | sem xformers, cai na atenção SDPA em vez de quebrar |
| `zen_image_edit_offload_encoder.patch` | zen-image-edit (Arc) | offload do encoder |

Os patches de nó guardam só o diff. Ao lado do arquivo alterado costuma haver uma cópia `*.orig_AAAAMMDD` do
original, fora do Git.

### `arquivo/`
Tentativas que não deram certo, guardadas como registro. Não aplicar.

| Patch | O que era |
|---|---|
| `comfyui-gguf_xpu_staged_copy.FALHOU.patch` | GGUF: cópia em estágios para XPU; falhou |
| `comfyui_qwen21_fused_qkv_profile.SEM_GANHO.patch` | Qwen-Image-2.1: `fused_convrot_qkv` (uma quantização de ativação ConvRot para `to_q/k/v`) e perfil por `QWEN21_PROFILE`. Imagem idêntica, wall time igual; revertido em 2026-10-01 (`.scratch/quantfunc_2026-10-01/paper_investigacao.md`, §8–9). Reaplicar só para repetir o perfil |

Adendo 03/10 (fase 4): `comfyui_sage_pv_accum_env.patch` passou a tocar também `comfy/model_base.py` — política por
modelo (`BaseModel.sage_attention`, Krea2 = per_warp + smooth_k, `COMFY_SAGE_MODEL_POLICY=0` desliga) — e
`comfyui_swiglu_triton_env.patch` ganhou o fold W4A4 SwiGLU em `linear_input_act` (`COMFY_W4A4_INPUT_ACT=0` desliga), que
depende de `comfy_kitchen_swiglu_w4a4_fused.patch` e do módulo `_swiglu_quant` (`tools/ck_swiglu/build.bat --install`;
rebuild após atualizar o torch). Testes: `patches/tests/test_comfyui_fase4_dispatch.py`. Medições em
`.scratch/quantfunc_2026-10-01/otimizacao/resultado_fase4.md`.

Adendo 02/10 (fase 2): `comfyui_sage_pv_accum_env.patch` ganhou o valor `COMFY_SAGE_PV_ACCUM=fp16sv` (acumulador fp16 com a
subtração da média de V dentro do kernel sm80, `accum_f16_fuse_v_mean`, por cima da escala por coluna; cai em `fp16+fp32`
acima de 32 752 chaves). Medido no kernel: iguala o fp32 onde o `fp16` puro errava 10–75× (componente contínua) e é 10–12 %
mais rápido que o `fp16+fp32`; ponta a ponta em `.scratch/quantfunc_2026-10-01/otimizacao/resultado_modelos.md` §14.
