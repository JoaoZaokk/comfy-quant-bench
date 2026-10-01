# Regras rígidas: host, WSL, VRAM compartilhada

> Referência preservada do CLAUDE.md original, linhas 146–260, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Hard rules

- **Only** `.\python_embeded\python.exe`. Never global Python, pip, or Conda. Always pass `-s` when running scripts directly so user site-packages stay out.
- **Never** delete, move, overwrite, or requantize an original model. Outputs go beside their source with a `_w4a4_convrot` suffix; the converter refuses to overwrite anything.
- **Do not mass-upgrade** Torch / CUDA / ComfyUI / comfy-kitchen. Inspect installed versions and local APIs before proposing any package change. `_pip_freeze_before_w4a4_cu130_20260816.txt` is the current known-good freeze baseline.
- **WSL on this host runs the owner's containers. Do not stop, restart, or reconfigure it.** The running distro is `docker-desktop`; `Ubuntu` was *Stopped* and holds no project, so this rule was never about the Qwen/DeepSeek jobs it used to name. `wsl --shutdown` takes every container down with it. `vmmemWSL` eats ~28 GB RAM and can starve conversions (see *Memory gotchas*).

  **The ERP left this machine, on purpose, and the migration is done.** This rule used to say "eleven live containers — the `baselinker-erp-offline` stack (`zhao-*` / `baselinker-*`), a postgres, and a WhatsApp gateway", measured 2026-08-21. Measured 2026-08-30 in **both** docker contexts (`default` and `desktop-linux` — same engine, so one `docker ps` was not enough to claim absence): **five** containers, and **no `zhao-*` or `baselinker-*` at all**, not even stopped. Confirmed by the owner the same day: **the ERP is on Proxmox now, and MacroLog migrated too.** What still runs here is the **rollback copy**, which he does not expect to need. `F:\zhao-hub-data` (6.7 GB, 16258 files) is the old data, kept.

  So the rule's original reason is gone, but the rule stands until the owner says otherwise, because `wsl --shutdown` is still destructive to what *is* running:

  ```
  recensa             s40911120/recensa         (healthy, up 24h)
  gestao-db           postgres:16-alpine        (healthy, up 37h)  <- the MacroLog rollback copy
  macrozao-api        app_api-api               (up 37h)
  nostalgic_torvalds  cloudflare/cloudflared    (up 37h)
```

**That list was measured 2026-09-20 and it is NOT the one this file carried until then.** The old
one named `glm-w4` (`qwen38-pp-dspark`) as the container holding the 3090 and `macrozao` as
healthy. Neither is true: **`glm-w4` does not exist** -- today's GPU container is `glm46v`
(`vllm/vllm-openai`), **Exited 3 days ago**, and the 3090 measured **49 MiB used at 0%**, i.e.
free; `macrozao` is `Exited (137) 2 weeks ago`. Of **20** containers in `docker ps -a`, **4** run.
Every paragraph below that says "while `glm-w4` held the 3090" is still a valid *measurement of
2026-08-30*; it is not a statement about today's machine.

```
  ```

  **Consequence worth knowing: the `.vhdx` compaction is no longer blocked by the ERP.** `Optimize-VHD` needs `wsl --shutdown` because all distros share one VM — that was unthinkable with the ERP here and is now a scheduling question, not a hazard. Still the owner's call, and still never `--allow-unsafe`.

  **But the numbers this paragraph carried were wrong, and so was the mechanism.** It said "~140 GB sits in `docker_data.vhdx` (229 GB, ~90 GB of build cache with zero active entries) and `ext4.vhdx` (63 GB with 22 GB used)". Measured 2026-09-20:

```
  docker_data.vhdx   318.86 GiB   (was 229 -- it grew ~90 GiB)
  ext4.vhdx (Ubuntu)  39.61 GiB   (was 63)
  docker system df   TOTAL  ACTIVE   SIZE      RECLAIMABLE
    Images              54      15   191.1 GB   101.6 GB (53%)
    Local Volumes       23       9    56.77      44.62   (78%)
    Build Cache        161       0    63.3        8.798
    Containers          20       4     2.36       2.359  (99%)
                                                 157.4 GB
```

  191.1 + 2.36 + 56.77 + 63.3 = **313.5 GB of real content inside a 318.86 GiB file**. **The vhdx is not inflated — it is genuinely full**, so compacting *before* pruning returns almost nothing. The order is **prune first, compact second**, and the big item is now **images** (101.6 GB reclaimable), not the build cache the old text blamed. Two cautions: `docker volume prune` targets 44.62 GB across 14 inactive volumes and one of them may hold the MacroLog rollback or the rakazo postgres data, so that one goes volume by volume, never in bulk; and `wsl --shutdown` drops the four running containers including `gestao-db`.

  **Also measured the same day, and it makes every commit-charge number in this file worse:** the pagefile is **35.07 GiB**, not 61, so the commit limit is **98.72 GiB**, not 124.8 — and free commit was **24.27 GiB**. A 38 GiB safetensors needs 76 GiB of commit through ComfyUI's normal reader. Re-read the pagefile before trusting any commit arithmetic here; it is system-managed and moves.

  **And two distro facts this file had wrong:** `Ubuntu` is **Running** (this file says it was *Stopped* and "holds no project"), with its `ext4.vhdx` written the same day; and `NVIDIA-Workbench` **exists, Stopped** — the memory `topologia-da-maquina-e-o-erp` recorded it as `wsl --unregister`ed and that is false.

- **On a shared card, `torch.cuda.mem_get_info` reports memory that is real but belongs to somebody else.** Measured 2026-08-30 while the `glm-w4` container (inside WSL2) held the 3090: torch reported **23332 MiB free** where `nvidia-smi` reported **8362 MiB** — a 15 GiB disagreement on the same card in the same second. Same direction on the 3080 Ti, 41x, stable over six interleaved rounds.

  **The obvious reading of that gap is wrong, and this file carried the wrong one for a few hours.** It said the number is phantom and ComfyUI "will decide a model fits when it does not". It fits. Probed by allocating 1 GiB at a time on `cuda:0` with the tenant present (`scratchpad/probe_wddm_evict.py`, under `Assert-GpuLock`): **22 GiB allocated, zero failures, 14166 MiB past what the driver called free.** WDDM evicts the neighbour's pages to system RAM on demand, so torch's number is honest about what this process can get.

  What it is *not* honest about is the price, and the probe shows the eviction happening:

  ```
   held MiB  alloc  torch free  smi free  smi used
      10240     OK         977      1002     23574
      11264     OK       12068      1002     23574   <- 11 GiB reappears: the neighbour was evicted
      18432     OK           0      1012     23564
  ```

  The neighbour's resident VRAM fell from 16214 to 13054 MiB and it was still paging back minutes later. It survived — the training kept running at 100% — but a 4-hour job silently trading VRAM for system RAM is a real cost that nothing in either tool reports.

  So the rule is not "distrust the number". It is: **`get_free_memory()` answers "can I get this?", never "is the card free?"** — and it is ComfyUI's own fit test (`comfy/model_management.py`, in `:978`, `:1096-1098`, `:1210-1213`). On a card with a tenant, ComfyUI will load successfully *by taking memory from whatever else is running*. Cross-check NVML before starting work on a card someone else is using, and read `/system_stats` as a statement about the reader, not about the hardware.

  Not covered: the neighbour's throughput was never measured, only its resident memory; Linux, TCC-mode cards, and a native-Windows neighbour are all untested.

- **`--reserve-vram` does bind, and the way this bench proved otherwise for twenty minutes is the lesson.** The flag's help says it reserves vram *"for use by your OS/other software"*. Reading the code, the reserve never enters `get_free_memory()` — it is added to the **demand** side — so recomputing the comparison by hand (`tools/probe_reserve_vram.py`) said `loads? YES` at every value up to 20 GiB and the conclusion written here was "the guard cannot bind". **That conclusion was wrong.** Measured end to end on 2026-08-30 with a real 15881 MiB checkpoint, a real `comfy.sd.load_diffusion_model`, and the real `load_models_gpu`, while `glm-w4` held the 3090:

  ```
   --reserve-vram   model    really free   landed on the card   neighbour
        (default)  15881 MiB   10924 MiB          +6672 MiB     unchanged
            8 GiB  15881 MiB   10924 MiB             +0 MiB     unchanged
           20 GiB  15881 MiB   10924 MiB             +0 MiB     unchanged
  ```

  The flag changes the outcome from a partial 6.7 GiB residency to nothing at all, and at the default it stayed **under** what the card actually had — no eviction, the neighbour ended exactly where it started.

  **What made the hand-recomputation wrong was an axis held, not an axis varied.** The first end-to-end attempt passed `force_full_load=True`, which is not what a sampler uses; under it every arm loaded and the neighbour *was* evicted, which looked like confirmation. The real path streams weights under a budget that the reserve genuinely shrinks. Two separate self-inflicted wounds in one probe: modelling a guard instead of calling it, and then calling it with the one argument that disables it. See the memory `teste-varia-o-eixo-errado`.

  **`--gpu-only` and `--highvram` bypass the reserve.** Same model, same card, same tenant, one arm per flag:

  ```
                           arm   really free    landed   placed
                       default     10924 MiB     +6672   cuda:0
                    --gpu-only     10924 MiB    +10169   cuda:0
                    --highvram     10999 MiB    +10252   cuda:0
   --gpu-only --reserve-vram 8     10999 MiB    +10207   cuda:0
  ```

  The last row is the finding: with `--gpu-only`, asking for 8 GiB of reserve lands **+10207** against **+10169** without it — the flag is simply not consulted. On the default path the same reserve took residency to zero. So `--reserve-vram` protects only while nothing else has overridden the placement policy, and the two flags people reach for to "make it use the GPU" are exactly the two that switch it off.

  **It does not back off, and "ComfyUI evicted nobody" was a measurement error.** That claim came from watching the card's `memory.used` rise by only ~10.2 GiB for a 15.9 GiB model and concluding ComfyUI had stopped early. It had not. Reproduced 2026-08-30 against a **synthetic tenant** — a separate native-Windows process holding 13 GiB with `torch.empty` and then sleeping, which unlike a live training does not touch its pages and page them back:

  ```
  --gpu-only, tenant holding 13790 MiB
    smi before                13790 MiB
    smi after                 23428 MiB      delta only +9638
    ComfyUI's own accounting  15881 MiB resident, no exception
    tenant afterwards          7402 MiB      <- it lost 6.2 GiB
  ```

  `7318 + 15881 = 23199`, against the 23428 measured. The full model landed. **The delta looked small because the other occupant was shrinking at the same time** — I attributed to ComfyUI a number that was the sum of two moving quantities, which is `ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` with the arms being two processes instead of two code paths.

  So WDDM evicts, and ComfyUI does ask it to. Against the live `glm-w4` this was invisible precisely because that training was at 100% and kept faulting its pages back in — its *resident* number barely moved while it was actually thrashing against the loader. **A neighbour that looks unharmed by resident VRAM may be the worst case, not the best.**

  The default path is different and is fully explained by arithmetic, not by any backoff: `MIN_WEIGHT_MEMORY_RATIO` is **0.0 on NVIDIA** (`comfy/model_management.py:454-457`), so the budget collapses to `lowvram_model_memory = max(0, free − (model_size + reserve))`. Measured `6750 MiB` against `23332 − (15881 + 700) = 6751`. Note the shape of that formula: it subtracts the **whole model** from free and loads the remainder, so a bigger model puts *less* on the GPU, and any reserve at or above `free − model_size` puts nothing there at all. That is what the `+0 MiB` rows above are.

  **Why `--gpu-only` differs is also not a policy decision:** under it, `comfy.sd.load_diffusion_model` writes weights straight to the GPU while reading the file (`smi_after_read` was already 23428 MiB, before `load_models_gpu` was ever called), so `load_models_gpu` and its reserve arrive after the memory is spent.

  One thing this closed by accident: the synthetic tenant is a **native Windows process**, not WSL, and `get_free_memory` over-reported identically (23332 MiB with 13567 MiB held). The inflation is not a WSL artifact.

  **Not covered:** no sampling was run, so this is placement, not a completed generation — inference-time allocation is a separate budget and untested here. The neighbour's throughput was never measured, only its resident VRAM. Re-run both with `tools/probe_reserve_vram.py` (arithmetic) and `tools/probe_reserve_e2e.py` (real load).
