# Resultado: Qwen-Image 2.1 leve na Arc A750 (2026-09-28)

Critérios: `criterio.md` (render) e `criterio_driver.md` (driver). Root na VM via `qm guest exec 101` no Proxmox.

## Causa raiz

A VM 101 tem `machine: q35,viommu=intel` e o guest punha a GPU num domínio IOMMU traduzido: cada mapeamento de memória
paginável passava pelo IOMMU emulado do QEMU. Isso dava cópia host->XPU de 60 MB/s, travas acima de ~128 MB e resets do
motor de cópia. `iommu=pt` no guest (domínio `identity`) resolve nos dois drivers. O `job_timeout_ms` do bcs só sobe
até 10 s no Xe (teto compilado); com 10 s continuou travando (X1).

## Xe vs i915 (mesmo script `bench_driver.sh`, render p2s42 com GGUF sem patch e zen com patch)

| config | cópia paginável 256 MB | pinada 256 MB | render p2s42 | s/it | resets | llama.cpp tg |
|---|---|---|---|---|---|---|
| X1 Xe, IOMMU traduzido, bcs 10 s | trava | 9,2 GB/s | falhou (DEVICE_LOST) | - | 3 | 11,1-11,5 tok/s |
| X2 Xe + iommu=pt | 6,6 GB/s | 9,3 GB/s | completou, **imagem lixo** (roxo com manchas) | 3,20 | 0 | 13,1 |
| X3 Xe + iommu=pt + reserve-vram 1 | 6,7 GB/s | 9,2 GB/s | completou, **imagem lixo de novo** | 2,96 | 0 | 13,1-13,2 |
| I1 i915 + iommu=pt | 7,3 GB/s | 8,8 GB/s | completou, imagem correta | 3,17 | 0 | 13,2 |

Escolha: **i915 + iommu=pt** (Xe calcula errado em silêncio, 2/2). Persistido no GRUB (`GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt"`,
original em `/etc/default/grub.antes-20260928`). GuC carregado 70.36 (kernel recomenda 70.53).

Um GPU HANG no i915 (18:15) quando DiT+VAE ficaram residentes e o ComfyUI achava ter 7,54 GB (outros processos ocupam
~0,4-0,8 GB). Corrigido com `--reserve-vram 1.0` (drop-in `/etc/systemd/system/comfyui.service.d/reserva-vram.conf`).

## Bateria de 12 no i915 (A1-A5 do criterio.md)

- A1: 12/12 sem erro, 0 hang/reset.
- A2: VRAM pela contabilidade do ComfyUI: DiT 4547 MB carregado, 183 MB descarregados antes do VAE (644 MB) em cada
  imagem, dentro de 6,5 GB úteis (7,54 - 1,0 reservado). O orçamento Vulkan não enxerga outros processos no i915, então
  não há medição externa. Encoder zen: 0 MiB residente depois do encode (teste isolado). **Parcial.**
- A3: 3,18-3,51 s/it (0,28-0,31 it/s); imagem quente 85,5-98 s (média ~90 s), primeira 200 s. **Confirmada** (faixa prevista 60-100 s).
- A4: MS-SSIM contra `zen_v12` (3090, DiT BF16 + zen) 0,928 média, 0,844 mín; contra `w4a16_q4_1` 0,835; contra `bf16`
  (TE nativo) 0,819. **Confirmada.**
- A5: 12/12 aderentes (`folha_i915_s42.jpg`, `folha_i915_s7.jpg`: em cima 3090 zen_v12, embaixo Arc). Mesmos defeitos da
  referência: linha de lixo no neon s42, contagem de frutas em s7. **Confirmada.**

Patches finais: só `patches/zen_image_edit_offload_encoder.patch` (o ComfyUI-GGUF ficou original;
`patches/comfyui_gguf_xpu_staged_copy.FALHOU.patch` era contorno do IOMMU e não é mais necessário).

Uso: o llama.cpp (`spark-x25-heretic`) ocupa ~6,9 GB da placa quando carregado; para gerar imagem é preciso pará-lo.

## LoRAs de poucos passos (2026-09-28, noite)

Baixadas na Arc (`/mnt/comfy-models/loras`, tamanhos iguais aos do HF): Viggle v0.2.1 r128 (0,68 GB), Viggle v0.1 4-step
r64 (0,34 GB), Turbo8 (1,36 GB). Nó `viggle_turbo.py` da Viggle em `custom_nodes/` (LoRA sem mesclar + sigmas).
Fun-Acc 4-step da Alibaba descartada: não é LoRA comum (4 cabeças, exige sampler próprio).

Seed 42, p2 (pôster) e p1 (pescador), 1024². Amostragem por passo 4,4-5,6 s com LoRA (3,2 s sem).

| config | amostragem | total quente | visual |
|---|---|---|---|
| base 25 passos | 80 s | ~90 s | referência |
| Viggle v0.2.1, 4 passos | 21 s | 31 s | **quebrado**: texto embaralhado, fantasma |
| Viggle v0.1 r64, 4 passos | 21 s | 28-31 s | bom; mais estilizado |
| Viggle v0.2.1, 6 passos | 31 s | 34-48 s | bom, mais perto do base |
| Viggle v0.2.1, 8 passos | 38 s | 49 s | bom |
| Turbo8, 8 passos | 44 s | 46-53 s | bom; xícara maior, um resto de xícara ao lado do S |

Folhas: `folha_turbo.jpg`, `folha_4passos.jpg`. 0 GPU hang. Workflows salvos em `~/ComfyUI/user/default/workflows/`:
turbo 4 passos (Viggle v0.1), 6 passos (Viggle v0.2.1), 8 passos (Turbo8); cada um aberto na interface e executado
com sucesso a partir do arquivo salvo (`folha_turbo_ui.jpg`). Avaliação só visual, 2 prompts; sem MS-SSIM.

## Edição (2026-09-28, noite)

Entradas: pescador (p1 s42) e frasco (p4 s42) da bateria i915. Casos: gorro vermelho (1 referência) e frasco na mão
(2 referências). Configs: base 25 passos, Viggle v0.1 (4), Viggle v0.2.1 (6), Turbo8 (8).

- Referências em 1024: só o de 4 passos pôs o gorro; base/6/8 ignoraram a instrução e saíram com aspecto "HDR queimado".
  `ModelSamplingAuraFlow` shift 5 (o do autor do zen) não corrige. Em 768 as 4 variantes editam certo, mesma pessoa e
  fundo (`folha_edicao.jpg`, `folha_edicao2.jpg`). Causa não isolada (DiT Q4_1 + zen em 1024); workflows fixados em 768.
- 2 referências em 1024 estouram a VRAM (12,3k tokens); em 768 passam nas 4 variantes.
- Patch do zen ajustado: a reserva pedida ao ComfyUI cresce com as referências (0,5 GiB + 1,2 GiB por imagem; medido
  +1,1 GiB com 1 imagem e +1,66 GiB com 2).
- RAM da VM: o ComfyUI foi morto pelo OOM killer (9,2 GB anon, VM de 15,5 GB sem swap) ao alternar LoRAs.
  Adicionado swap de 8 GB em /mnt/comfy-models/swapfile (fstab, swappiness 10) e `--cache-ram 4`; `--reserve-vram`
  subiu para 1,6 (LoRA da Viggle fica fora da contabilidade do ComfyUI). Depois disso: 0 OOM, 0 hang; ~3 GB em swap.
- Uma trava não explicada (thread a 100% por 19 min no primeiro job, com dois clientes enfileirando ao mesmo tempo);
  não reproduziu depois do restart.
- Conversão API -> UI: o frontend chama a entrada dinâmica de `images.image_1`; `loadApiJson` não liga `image_1`.
  A primeira versão dos workflows gerava outra pessoa (referência ignorada); religado à mão e reverificado.

Workflows salvos: edição 25 passos (~92 s), turbo 4 (~24 s), turbo 6 (~34 s), turbo 8 (~43 s), rodados na interface a
partir do arquivo, imagem 1 ativa e imagem 2 desativada (`folha_edicao_ui.jpg`: mesma pessoa, gorro, fundo mantido).
No workflow "leve" o `resolution` passou para 768 (só afeta referências).
