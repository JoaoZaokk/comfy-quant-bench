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

### comfy-kitchen 0.2.35
A ordem importa: o `awq` vai por cima do `w4a8`.

| Patch | Mudança |
|---|---|
| `comfy_kitchen_w4a8_dequant_fused.patch` | dequant fundido do W4A8 |
| `comfy_kitchen_awq_w4a16_triton.patch` | dequant AWQ W4A16 no registro de ops (eager + Triton); testes em `tests/test_comfy_kitchen_awq.py` |

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
Tentativas que não deram certo, guardadas como registro. Não aplicar. Exemplo:
`comfyui-gguf_xpu_staged_copy.FALHOU.patch`.
