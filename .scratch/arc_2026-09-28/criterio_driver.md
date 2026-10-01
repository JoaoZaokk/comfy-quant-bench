# Critério: Xe vs i915 (e limite do motor de cópia) na Arc A750 — 2026-09-28

Escrito antes das medições. Pedido do dono: "faz tudo que precisa, faz os testes, tira o limite do motor de copia, ve qual
funciona melhor, Xe ou i915". Root na VM via `qm guest exec 101` no Proxmox (sudo da VM pede senha).

Achado antes de medir: a VM 101 tem `machine: q35,viommu=intel` e o guest põe a GPU num domínio IOMMU `DMA` (traduzido).
Hipótese: cada mapeamento de memória paginável vira trap no QEMU, por isso 60 MB/s paginável contra 9 GB/s pinado.

## Configurações (em ordem)

- X1: Xe, cmdline atual, `job_timeout_max`/`job_timeout_ms` do bcs subidos (não persiste no boot).
- X2: Xe + `iommu=pt` no guest (reboot).
- I1: i915 (tira os dois force_probe) + a opção de IOMMU que vencer entre X1 e X2 (reboot).
- Se sobrar dúvida, I0: i915 sem `iommu=pt`.

## Medidas por configuração (mesmos scripts)

M1. Cópia host->XPU no torch: paginável 64 e 256 MB, pinada 256 MB (MB/s; trava = timeout de 60 s).
M2. Render p2s42 no ComfyUI (GGUF sem patch, zen com patch): sucesso, it/s do KSampler, tempo total; resets no kernel.
M3. llama.cpp Vulkan (container `spark-x25-heretic`): tokens/s de geração num pedido fixo de 128 tokens.

## Previsões

P1. X1 (só timeout): sem resets, mas paginável continua ~60-115 MB/s; render pode completar, com carga lenta (~1 min).
P2. X2 (iommu=pt): paginável sobe para >= 1 GB/s, render completa sem patch no GGUF.
P3. i915 com iommu=pt: igual ou melhor que X2 em M1/M2; M3 parecido (Vulkan anv funciona nos dois).
Escolha: a configuração que passa M2 sem resets com maior it/s, desde que M3 não piore mais de 10%.
Ao final: deixar a escolhida persistente, containers religados.
