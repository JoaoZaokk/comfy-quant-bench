# ComfyUI Ultra Optimization - 2026-08-26

## Resultado atual

- GLM/SGLang descarregado; o container `glm46v-quality` ficou parado e a RTX 3090 voltou a ~38 MiB antes de iniciar o ComfyUI.
- ComfyUI validado em `http://127.0.0.1:8190` com RTX 3090 + RTX 3080 Ti, SageAttention, pinned memory e async offload de 2 streams.
- Nunchaku usa `--disable-dynamic-vram`; `--fast` global não é usado.
- Frontend `Nodes 2.0` desligado e preview do sampler definido como `none`.
- 59 workflows auditados: zero erro de JSON; os três arquivos com nodes ausentes são templates remotos/API. Os workflows locais otimizados não ganharam novos nodes ausentes.
- Modelos e workflows originais não foram apagados, movidos ou sobrescritos.

## Launchers recomendados

### Imagem, Qwen Edit, Z-Image e Nunchaku

`run_nvidia_gpu_8190_ultra_image.bat`

- SageAttention;
- DynamicVRAM desligado para compatibilidade com os loaders Nunchaku;
- RAM-pressure cache para reaproveitar loaders e condicionamentos entre edições;
- previews desligados;
- um processo enxerga as duas GPUs e os nodes escolhem `cuda:0`/`cuda:1`.

### LTX, VOID e vídeos grandes

`run_nvidia_gpu_8190_ultra_video.bat`

- mesma base segura;
- `--cache-none` para impedir que intermediários antigos consumam toda a RAM;
- CUDA graphs continuam permitidos; cada workflow/node compatível decide se captura.

Não usar globalmente: `--fast`, `--high-ram`, `--fast-disk`, `--disable-pinned-memory`, `--gpu-only` ou `--highvram`. Nesta instalação, eles são incompatíveis com Nunchaku, aumentam risco de OOM ou já mediram pior no workflow VOID.

## CUDA Graph e torch.compile

Foi aplicado ao core local o patch estreito do PR upstream ComfyUI #15559 em `comfy/ldm/lightricks/symmetric_patchifier.py`. Ele remove duas construções `torch.tensor(python_data, device=cuda)` do hot path LTX, preservando os valores das coordenadas e eliminando as cópias H2D que bloqueavam captura.

Validação local:

```text
PASS: LTX patchifier hot path is capture-safe and preserves coordinate values
```

Script reproduzível: `tools/verify_ltx_cuda_graph_patch.py`.

Perfis SeedVR2 criados sem alterar o original:

- `SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_INDUCTOR_DIT.json`: recomendado; compila apenas o DiT com Inductor e deixa o VAE fora do compile.
- `SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_FAST_Q8_CUDAGRAPHS_DIT_EXPERIMENTAL.json`: captura explícita para comparação; usar com tamanho e batch fixos. Primeira execução é aquecimento/compilação, então comparar a segunda e a terceira.

Nunchaku não recebeu node `torch.compile`: o loader já usa kernels SVDQuant próprios, e combinar isso com o caminho `--fast`/lazy Linear reproduz o erro `NoneType.weight.dtype`.

LTX 2.5 não recebeu um node de CUDA Graph forçado. O patch remove o bloqueio confirmado, mas captura end-to-end continua dependente do modelo, formato, shapes e caminhos de sampler. Forçar sem benchmark seria transformar uma otimização em crash/OOM.

## Auditoria de workflows

Relatório final: `docs/workflow-audit-2026-08-26-ultra-final.json`.

Resumo:

```text
files: 59
remote_or_api: 10
with_missing_nodes: 3
with_missing_models: 8
with_accelerator_nodes: 29
parse_errors: 0
```

Os models ausentes restantes pertencem principalmente a variantes nativas ou templates que apontam para nomes não instalados. As variantes `*_MODELREF_FIXED.json` e `*_OPTIMIZED.json` são as cópias locais utilizáveis; os originais foram preservados.

## Avisos que continuam, mas não bloqueiam a base

- AnimateDiff: não há motion model instalado. Só é problema ao usar AnimateDiff.
- QwenVL GGUF: `llama_cpp` ausente, portanto apenas os dois subnodes GGUF não carregam. Os demais nodes do pacote carregam e nenhum workflow salvo referencia QwenVL.
- LTX2-MultiGPU: a integração automática com ComfyUI-MultiGPU cai para modo standalone, mas registra os quatro nodes e detecta `cpu`, `cuda:0` e `cuda:1`. Os workflows preferidos usam os loaders ComfyUI-MultiGPU/DisTorch2 diretamente.
- Nunchaku `minimal mode`: falta apenas `nunchaku_versions.json`; Nunchaku 1.2.1 e os loaders continuam registrados.
- VoxCPM: o handler antigo não existe no PromptServer atual; o node usa seu fallback de envio de configuração e carrega.
- Warnings de API frontend legada pertencem a extensões antigas. Não são traceback de inferência.

## Fontes técnicas consultadas

- ComfyUI PR #15559 e issue #15550: hot-path LTX e CUDA Graph.
- ComfyUI issues/discussions sobre DynamicVRAM, SageAttention, cache e LTX.
- ComfyUI-nunchaku issue #810: falha de lazy Linear com `--fast`.
- Ajuda CLI da versão local ComfyUI 0.33.0, que é a autoridade para os flags realmente instalados.

## Reversão

- Para voltar ao renderer novo, definir `"Comfy.VueNodes.Enabled": true` em `ComfyUI/user/default/comfy.settings.json`.
- Para voltar aos previews, definir `"Comfy.Execution.PreviewMethod": "auto"`.
- Para reverter o patch LTX, restaurar apenas `ComfyUI/comfy/ldm/lightricks/symmetric_patchifier.py` pelo checkout interno do ComfyUI e remover `ComfyUI/tests-unit/comfy_test/test_symmetric_patchifier.py`.
- Os launchers e workflows novos podem ser ignorados; nenhum launcher antigo ou workflow original foi substituído.
