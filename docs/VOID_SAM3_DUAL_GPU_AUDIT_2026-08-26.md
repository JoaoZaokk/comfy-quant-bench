# Auditoria VOID/SAM3 dual-GPU — 2026-08-26

## Resultado

O workflow otimizado executou de ponta a ponta no ComfyUI real e produziu um MP4 válido de 5 segundos, 120 frames, 720x1280, H.264 + AAC. A inspeção do frame central confirmou a remoção do `clideo.com`, sem alteração visível da personagem/bota e sem um quadrado aparente no carpete naquele frame.

O JSON original foi preservado:

- original: `C:\Users\joaoz\Downloads\VOID_FIXED_WATERMARK_3090_3080Ti.json`
- SHA-256 original: `6795B688048573C882A3190AB4117F31E079E4D28116F652A92BE9726B43EDF7`
- novo: `C:\Users\joaoz\Downloads\VOID_FIXED_WATERMARK_3090_3080Ti_OPTIMIZED.json`
- SHA-256 novo: `495965D1415A0801968BD89A172268768E5BD6107860149CF465978D051C4692`

## Ambiente confirmado

- ComfyUI 0.33.0, commit `c1739380`
- Python embarcado 3.13.12
- PyTorch 2.13.0+cu130, CUDA 13.0
- GPU 0: RTX 3090 24 GB
- GPU 1: RTX 3080 Ti 12 GB
- SageAttention efetivamente selecionado no log (`Using sage attention`)
- `comfy_kitchen` CUDA disponível; Triton instalado, mas desativado e sem evidência de ganho para este caminho VOID

## Causas encontradas

1. O mapeamento exposto do subgraph estava trocado: o campo de Pass 2 recebia o nome do T5 e o campo CLIP recebia o T5 FP16. Foi corrigido para `void_pass2.safetensors` e `t5xxl_fp8_e4m3fn_scaled.safetensors`.
2. O fluxo antigo decodificava o vídeo completo de aproximadamente 20 segundos antes de aplicar o recorte temporal de 5 segundos. No teste parcial isso levou a `54.36 GiB` privados e praticamente zero RAM disponível antes de a 3090 começar o sampling.
3. Source, Pass 1 e composições full-frame permaneciam referenciados simultaneamente. Agora o pipeline conserva o crop de 256x256 e só decodifica/recompõe o source no final.
4. O cache RAM-pressure padrão retinha a árvore pesada de loaders. A rodada comparativa chegou a `64.06 GiB` privados e apenas `1.59 GiB` disponíveis; foi interrompida com segurança antes do Pass 1.
5. `SelectCLIPDevice` clonava o T5 inteiro. O loader final usa `CLIPLoaderMultiGPU` diretamente em `cuda:1`, e o selector ficou bypassado.
6. O ComfyUI-MultiGPU tentava carregar `libcudart.so` no Windows e depois fazia CPU staging por camada quando não havia P2P. O host tem P2P indisponível nas duas direções. O patch local passou a carregar o `cudart64_*.dll` do PyTorch e a executar o DLPack no device real do tensor, sem o staging CPU -> GPU incorreto.

## Arquitetura final

- T5 FP8 scaled: carregado diretamente na GPU 1; usado pelos dois encodes; descarregado antes do sampling.
- CogVideoX VAE: GPU 1.
- VOID Pass 1: GPU 0; descarregado após o decode do Pass 1.
- RAFT: GPU 0; descarregado após materializar o warped noise. Não foi movido para GPU 1 porque o loader nativo não expõe device e isso exigiria uma cópia cross-GPU do batch temporal sem P2P.
- VOID Pass 2: GPU 0; descarregado antes do decode final.
- Recompose: source lazy de 5 segundos decodificado apenas após os modelos ficarem fora de `loaded_models`; paste do crop feito in-place.
- SAM3 temporal e auto-crop: desativados para este watermark fixo. A definição do subgraph foi mantida para reutilização futura.

## Prova dos unloads

Medições do run final (`runtime_barriers.jsonl`):

| boundary | loaded models antes | GPU 0 reserved antes | loaded models depois | GPU 0 reserved depois |
|---|---|---:|---|---:|
| Pass 1 | CogVideoX + VAE | 12,650,020,864 B | somente VAE | 100,663,296 B |
| RAFT | RAFT + VAE | 1,811,939,328 B | somente VAE | 167,772,160 B |
| Pass 2 | CogVideoX + VAE | 12,448,694,272 B | somente VAE | 100,663,296 B |
| recomposição final | nenhum | 100,663,296 B | nenhum | 100,663,296 B |

`unload_model_and_clones` primeiro remove a residência CUDA. Como o launcher usa `--cache-none` e a barreira não propaga o ModelPatcher, a referência do estágio torna-se descartável quando a dependência termina. Isso é diferente de apenas escolher `offload_device=cpu` e manter o modelo ocioso no cache.

## Benchmarks de 5 segundos

Todos usam crop/mask/seed iguais, Pass 1 = 24, Pass 2 = 18 e CFG = 6.

| variante | resultado | tempo total | pico privado | RAM disponível mínima | GPU0 pico | GPU1 pico |
|---|---|---:|---:|---:|---:|---:|
| grafo antes do lazy slice, cache-none | interrompido antes do sampling | n/a | 54.36 GiB | ~0 GiB | 0.53 GiB | 3.19 GiB |
| otimizado, cache-none + Sage | sucesso | 150.61 s | 52.44 GiB | 8.34 GiB | 12.33 GiB | 8.95 GiB |
| otimizado + fast-disk | sucesso | 161.08 s | 49.16 GiB | 5.94 GiB | 12.33 GiB | 9.27 GiB |
| cache RAM-pressure padrão | interrompido por pressão | 57.27 s até abortar | 64.06 GiB | 1.59 GiB | 5.09 GiB | 8.54 GiB |
| final, T5 direto em cuda:1 | sucesso | 148.617 s | 53.47 GiB | 8.23 GiB | 12.33 GiB | 9.54 GiB |
| final + `--disable-pinned-memory` | sucesso | 158.282 s | 50.46 GiB | 7.80 GiB | 12.33 GiB | 9.19 GiB |

Pass 1 levou aproximadamente 40 segundos e Pass 2 aproximadamente 30 segundos; o restante é carga/conditioning, VAE, RAFT, recomposição e encode.

O run final foi ligeiramente mais rápido e elimina o deep clone do T5. A diferença de aproximadamente 1 GiB no pico privado entre os dois runs cache-none não sustenta afirmar melhora de RAM pelo loader direto; o ganho confirmado é semântico (sem clone) e de tempo. O alvo ideal de menos de 40–45 GB **não foi atingido**. O maior saldo restante está nos pesos/model objects e no allocator do processo, não nos frames full-frame.

## Launcher recomendado

`F:\COMFY_PORTABLE\run_nvidia_gpu_8190_void_cache_none.bat`

Flags:

- `--use-sage-attention`: confirmado ativo no log.
- `--disable-dynamic-vram`: preserva a compatibilidade exigida pelos workflows Nunchaku desta instalação.
- `--cache-none`: evita a retenção que levou o processo a 64 GiB.
- `--listen 127.0.0.1 --port 8190`: acesso somente local.
- sem `--fast-disk`: economizou cerca de 3.3 GiB privados, mas ficou 10.47 s mais lento e não melhorou a folga mínima do sistema nessa comparação.
- sem `--disable-pinned-memory`: economizou 3.01 GiB privados, mas ficou 9.665 s (6.5%) mais lento, aumentou ligeiramente o pico RSS (35.45 contra 34.85 GiB) e não melhorou a folga mínima do sistema. O launcher prioriza velocidade.
- `COMFYUI_MGPU_DISABLED=1`: desativa o orquestrador job-level do plugin; os Select/Load device nodes usados no workflow continuam ativos.

## Arquivos criados/alterados

- `custom_nodes/comfy-void-stage-tools/__init__.py`: lazy temporal slice, recomposição in-place e barreiras explícitas de unload/auditoria.
- `ComfyUI/custom_nodes/comfy-void-stage-tools/__init__.py`: stub de carregamento persistente.
- `tools/build_void_optimized_workflow.py`: construtor reproduzível sem sobrescrever o original.
- `tools/monitor_void_runtime.py`: telemetria por processo, sistema e NVML.
- `run_nvidia_gpu_8190_void_cache_none.bat`: launcher recomendado.
- `ComfyUI/custom_nodes/ComfyUI-MultiGPU/p2p_registry.py`: CUDA runtime compatível com Windows.
- `ComfyUI/custom_nodes/ComfyUI-MultiGPU/__init__.py`: DLPack guard pelo device real do tensor.

## Validação da saída

- MP4: `F:\COMFY_PORTABLE\ComfyUI\output\video\VOID_direct_t5_bench5s_00001_.mp4`
- SHA-256: `C39EC288C2773FB16177FAC785FE8E881A0260F433D9E8F987705B34CBD16D54`
- duração: 5.000 s
- vídeo: H.264, 720x1280, 120 frames
- áudio: AAC, 236 frames
- output visualmente inspecionado no frame de 2.5 s
- A/B sem pinned memory: `F:\COMFY_PORTABLE\ComfyUI\output\video\VOID_direct_t5_bench5s_00002_.mp4`, 5.000 s, H.264 720x1280/120 frames + AAC/236 frames.

## Não verificado nesta rodada

- VAE GPU 0 versus GPU 1 em benchmark A/B isolado.
- RAFT na GPU 1, porque isso exige um loader/device route adicional e a máquina não possui P2P entre as GPUs.
- SageAttention versus SDPA/xFormers com o mesmo seed; Sage foi apenas confirmado como backend ativo.
- redução de steps para 16/10 ou 16/12; o baseline de qualidade 24/18 foi mantido.
- ausência de flicker em todos os 120 frames por análise temporal automática; houve inspeção do vídeo/frame e integridade do arquivo, não uma métrica óptica completa.
