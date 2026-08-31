# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`F:\COMFY_PORTABLE` is a **live ComfyUI Portable installation**, not a clean source project. It is currently the working bench for the **ConvRot W4A4 quantization project** (converting local high-precision checkpoints to native 4-bit weight / 4-bit activation format).

**Three separate Git repositories overlap here, and confusing them is easy.** The root *is* a repo — on `master`; **run `git ls-files | wc -l` for the count, do not quote one from here.** (This file used to say the root was not a repo, which stopped being true once `tools/` was committed.) The number is deliberately absent: it was written as 88, corrected to 89, and was 93 one commit later — three staleness events in a single session, the last one caused by the commit that fixed the previous one. It is an **allowlist**: `.gitignore` ignores `/*` and re-includes tracked paths one by one, because models alone are ~600 GiB. `ComfyUI/` is a *separate* checkout (version **0.33.0**, `v0.33.0-19-gc1739380`, 2026-08-18 -- it was 0.29.0/`42d2aa55` earlier in this project's life) and is **not** tracked by the root repo. `ComfyUI/custom_nodes/` holds dozens of packages — **count them, do not quote a number from here**, with `ls -A ComfyUI/custom_nodes | wc -l` and `find ComfyUI/custom_nodes -mindepth 1 -maxdepth 1 -type d | wc -l`. This paragraph said "66 entries, 64 directories, 63 packages" at 06:00 on 2026-08-21 and `ls -A` gave **67 entries, 65 directories** the same afternoon — the third staleness event on this one number, which is why it now goes the way the tracked-file count went. It aged a fourth time: measured 2026-08-30, `ls -A` gives **93 entries, 91 directories** — the package count grew by a third in nine days, so any number written here is wrong before it is read. Two of the entries are loose files (`example_node.py.example`, `websocket_image_save.py`), one directory is `__pycache__`, one directory is literally named `.disabled`, and **four** entries end in `.disabled` and are hidden from plain `ls`: subtract those before calling anything a package count. **Roughly half have a `.git` of their own** — **42** measured 2026-08-30 (this line said 31), with `find ComfyUI/custom_nodes -maxdepth 2 -name .git | wc -l`. An earlier version of this very paragraph claimed each entry was its own repo; that is false for half of them, which matters the moment someone tries to `git -C <package>` anything. Only **four** files under `custom_nodes/` are ours and tracked by the root, in **two** packages: three in `custom_nodes/comfy-quant-preflight/` and one in `custom_nodes/comfy-void-stage-tools/` (tracked 2026-08-29; this line said "three files, all in comfy-quant-preflight" until then). Count it, do not quote it: `git ls-files custom_nodes`.

Count these before quoting them. Both the file count and the node count were wrong in this file until they were recounted — see the memory `escrevo-mais-rapido-do-que-confiro`.

Also read [AGENTS.md](AGENTS.md) (root policy) and [ComfyUI/AGENTS.md](ComfyUI/AGENTS.md) (upstream engineering style — mandatory before editing anything under `ComfyUI/`).

## Hard rules

- **Only** `.\python_embeded\python.exe`. Never global Python, pip, or Conda. Always pass `-s` when running scripts directly so user site-packages stay out.
- **Never** delete, move, overwrite, or requantize an original model. Outputs go beside their source with a `_w4a4_convrot` suffix; the converter refuses to overwrite anything.
- **Do not mass-upgrade** Torch / CUDA / ComfyUI / comfy-kitchen. Inspect installed versions and local APIs before proposing any package change. `_pip_freeze_before_w4a4_cu130_20260816.txt` is the current known-good freeze baseline.
- **WSL on this host runs the owner's containers. Do not stop, restart, or reconfigure it.** The running distro is `docker-desktop`; `Ubuntu` was *Stopped* and holds no project, so this rule was never about the Qwen/DeepSeek jobs it used to name. `wsl --shutdown` takes every container down with it. `vmmemWSL` eats ~28 GB RAM and can starve conversions (see *Memory gotchas*).

  **The ERP left this machine, on purpose, and the migration is done.** This rule used to say "eleven live containers — the `baselinker-erp-offline` stack (`zhao-*` / `baselinker-*`), a postgres, and a WhatsApp gateway", measured 2026-08-21. Measured 2026-08-30 in **both** docker contexts (`default` and `desktop-linux` — same engine, so one `docker ps` was not enough to claim absence): **five** containers, and **no `zhao-*` or `baselinker-*` at all**, not even stopped. Confirmed by the owner the same day: **the ERP is on Proxmox now, and MacroLog migrated too.** What still runs here is the **rollback copy**, which he does not expect to need. `F:\zhao-hub-data` (6.7 GB, 16258 files) is the old data, kept.

  So the rule's original reason is gone, but the rule stands until the owner says otherwise, because `wsl --shutdown` is still destructive to what *is* running:

  ```
  glm-w4              qwen38-pp-dspark:0.27.1   <- the owner's own GPU work; holds the 3090
  macrozao            tributario-macrozao       (healthy)  } project `tributario`
  gestao-db           postgres:16-alpine        (healthy)  } the MacroLog rollback copy
  macrozao-api        app_api-api
  nostalgic_torvalds  cloudflare/cloudflared
  ```

  **Consequence worth knowing: the `.vhdx` compaction is no longer blocked by the ERP.** ~140 GB sits in `docker_data.vhdx` (229 GB, ~90 GB of build cache with zero active entries) and `ext4.vhdx` (63 GB with 22 GB used). `Optimize-VHD` needs `wsl --shutdown` because all distros share one VM — that was unthinkable with the ERP here and is now a scheduling question, not a hazard. Still the owner's call, and still never `--allow-unsafe`.

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
- **W4A4 means native ConvRot CUDA execution**, not weight-only INT4 followed by BF16 GEMM. Any change that lets the work fall back to eager/dequantized math defeats the entire project.

  **But this rule names two options where the hardware offers three, and the third one wins on accuracy.** Measured 2026-08-30 on the 3090, varying one axis — `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` (`comfy_kitchen/backends/cuda/__init__.py:212`), which forces the same `convrot_w4a4_linear` down the other branch. `_cuda_device_supports_native_int4_mma` is `major == 8` (`:293`), so Ampere and Ada reach the `m16n8k64 s4` MMA and **Hopper and Blackwell are routed to the INT8 branch deliberately**. The native branch is not merely selected here — it executes: outputs differ from the fallback in 6/6 synthetic cases (`tools/probe_int4_mma_dispatch.py`).

  The third option is **INT4 weight × INT8 activation on tensor cores**, which is neither "native INT4 MMA" nor "dequantized BF16 math". On **real** Z-Image activations and real weights — 24 layers spanning crest 4.4 to 43.0, from `calib/xfer_z_image_turbo_bf16.calib.pt` (`tools/probe_int4_vs_int8_real_acts.py`):

  ```
  media rel-RMSE   nativo 1,28e-1   int8 8,59e-2
  nativo ganha em 0/24 camadas      -> int8 e 1,49x mais fiel
  Spearman(crest, nativo-menos-int8) = +0,247
  ```

  Zero of twenty-four. And this was the run meant to *rescue* the native path: the earlier synthetic measurement used gaussian input, which has no outliers, and outliers are what the rotation exists to suppress — so real activations were expected to narrow the gap. They widened it slightly, 1.4x to 1.49x. Crest does not explain it either (+0.247, weak, and this repo already measured crest against W4A4 error at +0.10).

  Speed is the other half and it points the other way: native is **1.41x and 1.67x faster at M=1024**, and **1.3x to 1.74x slower at M=1** — the `m_crossover` curve shape again.

  So "anything that is not native INT4 defeats the project" is contradicted by measurement, and the two largest public ConvRot distributors — `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` (791k downloads, `w4a4_int4mm_layers: 0`) and `joeygambino/...surgical_int8_convrot` (`target_dtype: int8_tensorwise`) — ship exactly the third option. That reads as a deliberate trade, not a shortcut.

  **Executed 2026-08-31, with the models on the GPU** — `tools/probe_quant_dispatch.py --forward-only` records the device of the weights, the implementation the registry resolved, and the `linear_dtype` that reached the dispatcher, then varies `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`:

  ```
  checkpoint                                 linear_dtype  impl     flag muda?  ramo
  zimage-v2-w4a4                 (nosso)         int4      cuda        SIM      nativo int4
  LTX25-distilled-DiT-comfy-w4a4 (riftcast)      int4      cuda        SIM      nativo int4
  MiniMax_H3_FL2VA         (Abiray, 791k dl)     int8      cuda        NAO      INT8, por instrucao
  ```

  The Abiray file **does run quantized math on the CUDA backend**; it is simply insensitive to the int4 fallback flag because those layers never take that branch. Its own summary contradicts itself, too — `convrot_w4a4_mixed` carries `"linear_dtype": "int4"` at the top and `"w4a4_int4mm_layers": 0` three keys below, while all 117 per-layer `comfy_quant` tensors say `"int8"`. Read the layers, not the summary. And `LTX25-distilled-DiT-comfy-w4a4` (`quantized_by: riftcast/ltx25-quant-lab`, 1440 layers, `linear_dtype` absent) is a **second public W4A4 that really executes 4 bits**, and a seventh known checkpoint writer for ticket 08.

  **The rule stands until the owner decides otherwise**, because this is a trade and the choice is his: accuracy and small-batch latency favour the INT8 branch, large-batch throughput favours native.

  **And whether native W4A4 is worth reaching at all depends on the MODEL, which this rule does not mention.** Measured 2026-08-31 on the 3090, requantizing `hunyuanvideo1.5_720p_t2v_fp16` with today's pipeline and re-rendering the exact prompt, seed and sampler of the public repo's August test:

  ```
  modelo             W4A4 por passo   imagem       divergencia
  Z-Image                1.83x-1.93x MAIS RAPIDO   boa      0.7173
  HunyuanVideo 1.5       1.055x MAIS LENTO         DESTRUIDA 0.8255
  ```

  Same format, same kernel, same converter, same `convrot_groupsize` 256, output byte-size identical to August's 7.92 GiB. The public repo's warning — *"ConvRot W4A4 is slower than FP16 and destroys the output"* — **reproduced fifteen days later** across `comfy-kitchen` 0.2.23→0.2.31, ComfyUI 0.29→0.33 and torch 2.12.1→2.13.0. **Nothing in the stack fixed it**, because nothing in the stack was broken: the model is the axis.

  The leading hypothesis before the run was the `to_native.py` fused-qkv gap (scales passing through unrenamed, layer loading with no scale and no error). It died on reading, not on the GPU: `HunyuanVideo.process_unet_state_dict` carries `.comfy_quant` and `.weight_scale` through its own substring replacements — the gap is Z-Image-specific.

  Two consequences. First, **a checkpoint that converts cleanly, resolves the CUDA backend and writes a valid sidecar can still produce garbage** — the preflight proves dispatch, never quality; only a render does. Second, **latent divergence has no threshold**: 0.8255 destroyed against 0.7173 fine is a 15% gap separating "unusable" from "ship it", so no cut on that axis decides anything.

  Not covered: one seed, one prompt, 480x480, one frame, no perceptual metric, no SASS, and only `convrot_groupsize` 256 in this round. Re-run with `tools/quant_w4a4.py --profile hunyuan_video_15` then `tools/quality_ladder.py`; see `W4A4_PROGRESS.md` part 36.

  **The "is 1.49x visible?" question is now answered, and answering it corrected how this bench measures.** Three measurements of the same pair, 2026-08-30/31:

  ```
  erro por camada, ativacao real     int8 1,49x mais fiel   24/24 camadas
  epsilon por passo, entrada casada  int8 1,33x mais fiel    8/8 passos
  imagem final, trajetoria livre     2x1 em 3 sementes, ambos a ~0,3-0,5 do BF16
  ```

  The final-image comparison is the one that carries no signal, and it is the one that looked most like an answer. At 8 steps a tiny perturbation reroutes the sampler, and the destination is still a good image — so the free-running image measures chaos, not fidelity. **Do not use a generated image to compare two quantizations of the same model.** `tools/probe_epsilon_per_step.py` is the instrument that does work: the BF16 arm records every `(x, timestep)` it was called with via `model_options["model_function_wrapper"]`, and the quantized arms replay exactly those inputs, so every step is a matched comparison and trajectory divergence cannot exist by construction.

  This also **rehabilitates the per-layer criterion** that `tools/quant_mixed.py` uses. It was written off here on 2026-08-30 as non-predictive; it is not. It predicts the model's prediction error (1.49 against 1.33, same direction, same winner). What it does not predict is the free-running image, and nothing does.

  New information only the per-step view gives: the error is **concentrated at high sigma**. Native goes 6.17e-1 at sigma 1.000 down to 5.31e-2 at sigma 0.300, cosine 0.787 → 0.9986. Quantization damage lands hardest where structure is decided, and decays monotonically into texture. A criterion that weights layers by their contribution at high sigma is therefore a different thing from one that weights all steps equally. **Tested on 2026-08-31, and it does not pay.**

  `calibrate_activations.py` now records the sigma of every sampled row (`sample_sigma`), and `quant_mixed.py` takes `--sigma-weight none|sigma|sigma2|high` (default `none`, unchanged). The weighted criterion is nearly the same criterion: Spearman **+0.9935** (`sigma2`) and **+0.9629** (`high`) against flat on `err_w4a4` over 170 Z-Image layers, moving 6 and 9 layers across the 0.15 threshold.

  **The first comparison said it won 8/8 steps at 1.047x, and that was a confound, not a result.** The weighted build promoted 56 layers to 8-bit against the flat build's 53 — a bigger model, measured against a smaller one. **`ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` again, with the held axis being the promotion budget.** Always match the budget before comparing two selection criteria.

  **Rebuilt at exactly 56 promotions in all three arms and run over eight seeds, the three criteria are indistinguishable.** Paired design — within a seed the arms share the imposed BF16 trajectory:

  ```
  epsilon medio, 8 sementes    media       min       max    espalhamento entre sementes
  plano56                    1.3326e-1  1.1220e-1 1.5577e-1        1.388x
  sigma2                     1.3684e-1  1.1731e-1 1.5216e-1        1.297x
  sigma_alto                 1.3685e-1  1.2276e-1 1.4695e-1        1.197x

  diferenca pareada contra o plano:  sigma2 +3.37% (erro-padrao 3.95%, vence 4/8)
                                 sigma_alto +3.46% (erro-padrao 3.93%, vence 2/8)
  ```

  The between-seed spread inside a single arm reaches **1.39x**; the between-criteria difference is **3.4%**. The effect sits inside its own noise, and the win counts are coin flips. The sign is consistent — both weighted criteria come out slightly *worse*, never better — but at eight seeds that is a hint, not a result.

  **This corrects the three-seed version of this paragraph**, which reported "flat wins 3/3" and read as a finding. It was a small-sample artifact: adding a third arm made the ranking change between seeds, and the winner across eight seeds is 3 / 3 / 2 split between the three.

  Re-run with `tools/probe_epsilon_ckpt_ab.py`, which imposes the BF16 trajectory on N checkpoints at once. Not covered: one prompt, one model, one scheduler, eight seeds, no perceptual metric, and no formal hypothesis test — only the distance between the effect and its own spread.

  **Not covered:** one prompt, one seed, no perceptual metric, only Z-Image, only sm86, no SASS. And BF16 is the target, not the truth — it was never itself validated against float32.

  **And for a text encoder the rule is unreachable by construction, whatever the checkpoint says.** Measured 2026-08-31 on the 3090 against a real 13.20 GiB public W4A4 encoder (`qwen3vl_32b_minimax_h3-int4_convrot`, 350 `convrot_w4a4` layers, `linear_dtype` absent so the signature default `"int4"` applies). Called by hand, the kernel is genuinely native — 15/15 outputs differ from the forced INT8 fallback. Loaded through the stock `CLIPLoader` path and *counted* during a real encode:

  ```
  100/100 Linear   MixedPrecisionOps.Linear, quant_format convrot_w4a4, weight QuantizedTensor
  _convrot_w4a4_forward        0
  QuantizedTensor.dequantize 350
  ```

  Zero. The weight stays 4-bit in VRAM and the **math is dequantized** — memory saved, no time saved, kernel never reached.

  **The obvious culprit is the wrong one, and flipping it changes nothing.** `comfy/sd1_clip.py:114` hardcodes `full_precision_mm=True` for every text encoder; setting it False left the count at 0. Instrumenting each term of `_use_quantized` (`comfy/ops.py:1372-1377`) on a real forward names the one that actually bites: **`comfy_force_cast_weights=True`**, which comes from `model_patcher.py:743` via `set_model_compute_dtype`, called at **`comfy/sd.py:269` for every CLIP object** — `set_model_compute_dtype(torch.float32)`, commented "Match torch.float32 hardcode upcast in TE implemention". Two independent locks, and only releasing both fires the kernel.

  **Release it at the source, never on the modules — and this cost a run that looked like a result.** `model_patcher.py:1016` rewrites `m.comfy_force_cast_weights = self.force_cast_weights` on every module *every time the model is loaded to GPU*. Writing the attribute on the modules survives only if the model happened to be resident already, which depends on VRAM state at that instant. **The same command line gave 350 kernel calls on one run and 0 on the next.** The fix is `clip.patcher.force_cast_weights = False` (plus popping `manual_cast_dtype`), which `patch_model` then propagates. What caught it was the dispatch counter reporting 0 — a probe that only compared outputs would have reported "releasing the locks changes nothing" and been believed.

  With that fixed, the same measurement on three real encoders, each `--sem-bf16` except Qwen, which has a BF16 twin on disk:

  ```
  encoder                 formato  quant   travado  destravado    tempo            erro C-vs-B   cos
  qwen_3_4b (4B)            W4A4  2.4 GiB   70.2ms      82.9ms   1.18x MAIS LENTO    5.99e-1   0.949
  gemma_3_12B_heretic       W4A8  8.1 GiB  1772.3ms    478.9ms   3.70x mais rapido   2.11e-1   0.982
  qwen3vl_32b_minimax       W4A4 13.2 GiB   311.6ms    117.7ms   2.65x mais rapido   9.73e-2   0.99989
  ```

  **The sign is set by the sequence length, and that took varying one axis alone.** The first version of this paragraph asserted a mechanism it had not measured ("the cost scales with weight size while the token count stays tiny"), and the table above refuses it: the locked times are 70.2 ms at 2.4 GiB, **1772.3 ms at 8.1 GiB**, 311.6 ms at 13.2 GiB — not monotonic in weight size. The three encoders differ in weight size *and* in sequence length at once (Gemma's conditioning is `[1, 49, 1024, 3840]`, MiniMax's `[1, 8, 5120]`), so that table cannot separate them. Same file, same card, only the prompt changed:

  ```
  qwen_3_4b W4A4, RTX 3080 Ti, mediana de 3
  tokens   travado  destravado
      22     80.7      120.1    1.49x MAIS LENTO
      75    100.1      105.8    1.06x MAIS LENTO
     199    151.9       95.2    1.60x mais rapido    <- cruza entre 75 e 199
     424    245.2      102.0    2.40x
     850    456.1      124.3    3.67x
    1496    824.5      249.0    3.31x
  ```

  **The crossover is between 75 and 199 tokens** on this model and card. The released path is nearly flat from 22 to 424 tokens (120 → 102 ms) while the locked path climbs with the sequence, so the 4-bit kernel carries a per-layer fixed cost that a short prompt cannot amortise and the dequantized path pays a BF16 GEMM that grows with the tokens. Real prompts usually sit above that crossover: at 850 tokens the quantized encoder also beats the **BF16 original** (372.1 ms at ~700 tokens against 124.3 ms). Weight size may still matter on top of this — MiniMax won at 8 tokens — but it has not been isolated. And **the accuracy cost is not a property of the format**: two W4A4 files differ by 6x in the error releasing adds (5.99e-1 against 9.73e-2), so "W4A4 costs X" cannot be quoted without naming the checkpoint; on Qwen the long prompt is also worse than the short one (weight-only 2.55e-1 against 1.44e-1). For Qwen the full picture is available: the 4-bit *weight* alone already costs 1.44e-1 against its BF16 twin, and releasing takes it to 6.09e-1 — 4.23x.

  **Measured the same day on this project's own outputs, and the split is clean.** `tools/probe_quant_dispatch.py` counts the same way against a real load and a real forward:

  ```
  zimage-v2-w4a4        difusao  170 convrot_w4a4   340 quantizados, 0 sem, 0 dequantize
  gemma_3_12B_heretic   TE       336 asym_w4a8_int8   0 quantizados, 336 sem, 336 dequantize
  ```

  So: **the diffusion path really does run quantized** — every Z-Image number this bench has published (per-layer error, epsilon per step, the INT4-vs-INT8 comparison) was measured on genuinely quantized math, not on two dequantizations. And **our own Gemma is memory-only**, exactly as predicted, with both locks `True` on all 336 layers. The format does not matter: `asym_w4a8_int8` is blocked by the same `comfy_force_cast_weights`, so this is about being a text encoder, not about being W4A4.

  One counting trap worth keeping: `MixedPrecisionOps.Linear` is the class of **every** Linear in the model, quantized or not. Patching the class and counting all its calls first reported Z-Image as "MISTO, 340 against 76" — the 76 were layers that never had a quantized weight at all. Scope the count to `layout_type is not None`.

  **Still not measured:** what it costs in s/it on a real LTX render, and whether releasing the locks on a text encoder is safe for output quality.

  **The MiniMax half of that caveat is closed since 2026-08-31: its BF16 twin is now on disk.** `Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, 47.97 GiB, byte-exact against the published size, **351 of 351 layer names matching** the Winnougan quantization. Measured with `tools/probe_winnougan_fidelidade.py` over 20 layers, 5 shapes, 4 depths (0/16/33/49), M in {1, 64, 1024}: **int8 1.40x more faithful, 60 of 60**. That makes three independent measurements in the same direction — 1.49x on Z-Image with real activations, 1.33x on epsilon-per-step, 1.40x here — across a different model family, a **different quantizer** (theirs, not ours), and a different activation kind. Per-layer error is also flat in depth: same shape, block 0 to block 49, 2.2776e-1 → 2.2762e-1.

  Not covered: no SASS; there is no BF16 reference for the Gemma on this bench, so its error column is against its own dequantized arm and is **not** a fidelity claim; one prompt per encoder, one card; the locks were released by post-load monkeypatch, not by anything ComfyUI offers — `custom_operations` in `model_options` is the only real escape and no node exposes it. Re-run with `tools/probe_winnougan_int4.py`, `tools/probe_winnougan_load.py`, `tools/probe_te_fullprecision_mm.py` and `tools/probe_te_lock_cost.py`.

## Say which one it was: traced, or executed

Reading code and running code produce the same confident prose. That is the specific failure mode
this project keeps hitting — not carelessness, and not reluctance to be wrong, but that a careful
trace through a call chain *feels* like evidence and *reads* like a measurement, and nothing in
the writing separates them. It has cost real work here more than once:

| written as fact | what it actually was | what measurement said |
|---|---|---|
| "crest factor is the statistic ConvRot's activation path is sensitive to" | a mechanism argument, in a tool's own docstring | Spearman **+0.096** over 170 layers. No relationship. |
| "CUDA graph removes the 109 us" | inference from what graphs do | removes **83%**; ~11 us of host survives per replay |
| "host overhead is 161 us" | one median, from a contaminated process | **77 us** clean. Passed to a sibling project before it was checked. |
| "`verify_w4a4.py` runs the same check" | a claim in this very file | it resolves one op; the converter resolves two |
| "the probe is exact, with no error at all" | true of the weight, argued for the activation | BF16 scale rounding, ~1.4e-3 mean relative |

The fix is mechanical, because intent does not survive the next session. **When a conclusion comes
from reading, the artifact that carries it must say so, in the artifact.** Not in the chat, which
is gone by then.

- Tools print it. `tools/verify_w4a4.py`, `tools/test_svdq_verify.py` and
  `custom_nodes/comfy-quant-preflight/` each end their output with what they did *not* cover, on
  every run, pass or fail — a column of PASS lines otherwise reads as "verified".
- Docstrings carry the provenance and the caveat inline, next to the number, not in a paragraph
  below it. Numbers get pasted out of this repo into other sessions; a ratio with its condition
  attached ("4.89x [4.72-5.06] at M=5856 on weight [3840, 3840]; 1.5-2.0x *slower* at M=1")
  cannot be misquoted the way a bare "4.6x" can.
- **A single run is not a measurement.** Three consecutive `m_crossover` runs on an idle, locked
  3090 disagreed by up to 1.4x at the same M and shape, and the crossover itself moved a step
  between two runs in the *same* direction (2026-08-19, `W4A4_PROGRESS.md` part 11). The tool now
  repeats interleaved and prints the ratio's own min-max, because a two-decimal number from one
  burst claims a precision this bench does not have. Anything quoted from here needs a repeat
  behind it and the spread beside it.
- An unverified finding stays labelled unverified all the way into the file that acts on it, and
  **the label changes when the measurement arrives** — in both directions. The preflight package
  started with two audit-derived WARNs. `check_full_precision_matrix_mult` still says "not
  confirmed by execution" in the message itself, because a check that blocks on a hypothesis
  teaches people to disable checks. `check_lora_over_quantized` no longer says it: the hypothesis
  was **measured on 2026-08-19 and did not reproduce** — 680 of 680 dispatches stayed on the
  native kernel with the LoRA applied and in effect (latent norm moved 747.06 → 728.99, so it was
  genuinely applied). It is still a WARN, narrowed to what is actually still unknown: whether the
  delta costs *accuracy* once the weight is already 4-bit, which nobody asked.
- **No upstream PR from a trace.** Only from something run here. `AUDITORIA_2026-08-18.md` lists
  three PR candidates; only one has been proved, and the other two wait for the GPU.

Corollary that has paid off repeatedly: when a measurement contradicts a claim in this repo, the
claim is what changes, including claims in this file — and the tool that produced the wrong number
gets fixed too, not just the sentence.

## Environment

| Item | Value |
|---|---|
| Python | 3.13.12 (embedded) |
| Torch | 2.13.0+cu130 (torchvision 0.28.0, torchaudio 2.11.0, all `+cu130`) — subiu de 2.12.1 em algum ponto; `_check_accel.py` confirma Triton/Sage/FlashAttention ainda executando kernel sob 2.13 |
| comfy-kitchen | **0.2.31** (era 0.2.23 neste arquivo; subiu em algum ponto e a nota não acompanhou. É o registry que decide se `convrot_w4a4_linear` resolve para `comfy_kitchen.backends.cuda.*` — reconfira o preflight antes de confiar em conversão antiga) |
| spas_sage_attn (SpargeAttn) | 0.1.0+cu130torch2.9.0andhigher.post4 — wheel abi3 do woct0rdho, traz `_qattn_sm80.pyd`, roda kernel na sm86 |
| nunchaku (SVDQuant) | 1.2.1+cu13.0torch2.11 — build de torch 2.11 rodando sob 2.13; `ops.attention_fp16` executa na sm86, `ops.gemm_w4a4` ainda não exercitado |
| GPUs | RTX 3090 24 GB (`cuda:0`, cc 8.6), RTX 3080 Ti 12 GB (`cuda:1`) |

Non-obvious environment facts:

- ~~The cu130 Torch wheel ships only `cudart64_13.dll`, but the installed SageAttention 2.2.0 binary extensions link `cudart64_12.dll`, so a CUDA 12.6 runtime DLL was copied side-by-side into `torch\lib\cudart64_12.dll`. **Do not remove it.**~~ **OBSOLETE — but the file is NOT gone, and the sentence that said it was is a lesson.** From 2026-08-21 to 2026-08-22 this paragraph read *"the file is already gone"* and offered as proof: `find python_embeded -iname "cudart64*.dll"` returns only `cudart64_13.dll`. That command **cannot match a name ending in `.disabled`**, so it was blind by construction and returned a clean-looking result. Drop the `.dll` from the pattern and:

```
python_embeded/Lib/site-packages/torch/lib/cudart64_12.dll.disabled   556,544 bytes
  sha256 d954ca542b3b6bcf03cc2b798a7d00051501cf734ca751050e986af505cf9dad
```

It was **renamed, not deleted** — which `W4A4_PROGRESS.md:339-340` already recorded, executed, and this file contradicted for a day without either of them noticing. Windows will not load a `.dll.disabled`, so the runtime conclusion below stands unchanged; the *record* was wrong. **Keep the sha256**: the file is still sitting there unlabelled, and the hash is the only thing that identifies it. Leave it disabled.

This is the memory `arquivo-plausivel-nao-e-o-caminho` and this file's own rule — *absence in a grep is never absence in the system* — broken by the file that states it. The mechanical fix is the one already applied elsewhere here: **do not quote a command's output as proof without checking the command can see what it claims to rule out.**

What remains true, and is the part that matters: `sageattention._qattn_sm80`, `sageattention._fused`, `spas_sage_attn._qattn_sm80` and `flash_attn` **all import successfully** with the 12.6 DLL disabled — that is the Windows loader resolving the whole DLL chain, not a grep. The accel stack was reinstalled on 2026-08-16 with cu130 builds (`sageattention 2.2.0+cu130torch2.10.0andhigher.post6`, installed 15:01; `cudart64_13.dll` timestamped 15:34) which link `torch_cuda.dll` rather than cudart directly. **Do not "restore" the 12.6 DLL.** The caveat that used to travel with this — *an import proves DLL resolution, not kernel execution* — is **closed**. `_check_accel.py` was executed on the 3090 on 2026-08-22 with `CUDA_VISIBLE_DEVICES=0`: `triton v3.7.1 kernel compiled+ran`, `sageattention mean|d|=0.0006 vs SDPA`, `flash_attn mean|d|=0.0000`, xformers not installed. **ALL GOOD.** The accel stack runs kernels without the 12.6 DLL, and that is now a measurement rather than an inference. This rule survived five days after the file it protected stopped existing, in a file every session reads.
- Extra models are mounted via `ComfyUI/extra_model_paths.yaml`, which declares them at `D:/ComfyUI-Models/`. A missing model may live there, not under `ComfyUI/models/`. **`D:` is not a disk.** Measured 2026-08-22: `net use` reports `D: -> \\192.168.3.68\estoque`, a mapped SMB share, and `tools/quant_audit.py` resolves the declared path to that UNC and records both. So **408 GiB of the installation's ~1.01 TiB of model files arrive over the network** — which changes what a full walk costs, and means "the D: mount is offline" is a normal state rather than a broken one. That is a different NAS from the one holding the ERP code (`192.168.3.40`). A tool that walks both roots must treat a missing declared root as a warning, not a crash; one that is *told* to walk a path that is not there must still refuse (`quant_audit.py`, and `.scratch/varredura-2026-08-22/issues/15`).
- `venvs/ultravox311` is an unrelated side venv (Ultravox/TTS experiments — `teste_*.py` at root). Not part of ComfyUI.

## Commands

Launch:

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build
```

```bash
.\run_nvidia_gpu_8190_loopback.bat
```

**Any workflow using a Nunchaku SVDQuant loader needs `--disable-dynamic-vram`**, verified 2026-08-18 by a real generation. ComfyUI 0.33 added a Windows-only lazy `Linear` (`comfy/ops.py:520-538`, `self.weight = None` until `_load_from_state_dict`), enabled by `main.py:289`. ComfyUI-nunchaku reads `orig_attn.qkv.weight.dtype` in `patch_model` before that happens and dies with `AttributeError: 'NoneType' object has no attribute 'dtype'`. Nothing about this points at the loader or the checkpoint, and the same node called directly in-process works fine — so the error is easy to misattribute.

**The LTX 2.5 workflow on this bench needs `--disable-dynamic-vram` too**, verified 2026-08-19 by a real render (executed, not traced; the run is `F:\cortiq\e2e_comfy_512.log`, timestamped 2026-08-19 21:28 -- an earlier pass of this note said 2026-08-21, which was the date it was written down, not the date it was measured). This is a second, independent reason to pass the flag — do not read it as Nunchaku-only. The `cortiq` ledger asserted the opposite for this workflow; that assertion was a trace, not a run, and the run below is what overrides it. Without the flag the render dies with:

```
Model LTXAVTEModel_ prepared for dynamic VRAM loading. 14612MB Staged.
Model LTXAV        prepared for dynamic VRAM loading. 20484MB Staged.
aimdo: src/hostbuf.c:283:ERROR:hostbuf_read_file_slice: device copy failed
RuntimeError: HostBuffer.read_file_slice failed
torch.AcceleratorError: CUDA error: out of memory
```

14612 + 20484 = 35 GB staged on a 24 GB card. PyTorch's own summary, printed alongside, says `Allocated memory 141211 KiB` and `CUDA OOMs: 0` — the allocator that overflowed was dynamic-vram's, not torch's. **The message never names the subsystem at fault** and points the reader straight at "the model is too large", on a card that had 18 GB free at the moment of failure.

```bash
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --listen 127.0.0.1 --port 8190
```

**But `--disable-dynamic-vram` also makes two of the most-downloaded ConvRot checkpoints unloadable, and the error blames the file.** Measured 2026-08-31. Both `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` files carry bytes **past the last tensor** — 83 in FL2VA, 64 in Ref2VA — so `safetensors.safe_open` refuses them:

```
SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

That is not a broken download, and checking was cheap: the server's `Content-Length` matches ours byte for byte (15903012791 and 15093774276), and a `Range: bytes=-83` request returns the same trailing bytes we have. FL2VA's are readable text — `\nL2P_bypass_MiniMax_H3_FL2VA_..._convrot.safetensors_1785789862\n` — and Ref2VA's are 64 bytes of binary. **They are in the published file.**

The dynamic-VRAM path uses a different reader (`comfy/utils.py:85`, `comfy_aimdo.model_mmap`) which accepts them: with aimdo initialised both files open, 932 and 1132 tensors. So these checkpoints work for almost everyone — dynamic VRAM is the default — and fail for exactly the configuration this bench needs for Nunchaku and LTX 2.5. `minimax_h3_fl2va_pruned-w4a8_convrot_pruned` (Winnougan) has no trailing bytes and opens either way.

Launcher notes. There are **13** `.bat` launchers in the root, not seven — count them, do not quote this. The claim that **none** of them passes `--disable-dynamic-vram` was true on 2026-08-21 and is **false since 2026-08-26**: measured 2026-08-30, **four** pass it — `run_nvidia_gpu_8190_dual_component.bat`, `run_nvidia_gpu_8190_ultra_image.bat`, `run_nvidia_gpu_8190_ultra_video.bat`, `run_nvidia_gpu_8190_void_cache_none.bat`. Those four also set `COMFYUI_MGPU_DISABLED=1`. `run_nvidia_gpu_8190_loopback.bat` (Sage, `127.0.0.1:8190`) still does **not** pass the flag, so for a Nunchaku SVDQuant loader or the LTX 2.5 workflow above, use one of the four, call `main.py` by hand, or use `F:\cortiq-cmf\run_e2e_comfy.ps1`. Following "use the loopback launcher" alone will reproduce the 35 GB staging OOM. Other launchers: `run_nvidia_gpu.bat`, `run_nvidia_gpu_fast_fp16_accumulation.bat`, `run_nvidia_gpu_8190_flash.bat` (FlashAttention), `run_8190_limpo.bat`, `run_cpu.bat`.

**The `WinError 64` accept-loop death is NOT specific to the `0.0.0.0` launcher.** This file said `run_nvidia_gpu_8190.bat` binds `0.0.0.0` and the asyncio accept loop dies with `OSError(22, 'The specified network name is no longer available', 64)` whenever NordLynx/NordVPN reconnects — which reads as "loopback is immune". It is not. Observed 2026-08-30 at 04:29:19 on a server launched by `run_nvidia_gpu_8190_ultra_video.bat`, from its own log:

```
socket: <asyncio.TransportSocket fd=3000, family=2, type=1, proto=6, laddr=('127.0.0.1', 8190)>
```

`127.0.0.1`. Binding loopback does not protect against it. What *did* hold is the other half of the note: **the process stays alive with a dead port** — accept died at 04:29:19 and the process only ended at 04:35:18, six minutes of a listening-looking server that accepts nothing. `curl` returns `http=000` and `netstat` shows nothing `LISTENING`; the PID being alive proves nothing. Both `NordLynx` and `CloudflareWARP` tunnels were `Up`, plus a `cloudflared` container — three things that churn routes on this host.

**`main.py --windows-standalone-build` re-executes itself as a child process**, verified 2026-08-19 (executed, not traced). Killing only the PID that `Start-Process` returns leaves that child running as an orphan. On this bench the orphan held **20,578 MiB of the 3090** and **16.67 GB of RAM**, invisibly: `Get-Process -Id` reported the launched PID as dead, while `nvidia-smi` still showed the VRAM occupied. Kill the process tree, not the PID.

Tests (run from the root; `pytest.ini` lives in `ComfyUI/` with `pythonpath = .`).

**Neither `pytest` nor `ruff` is installed in the embedded interpreter** — verified 2026-08-18,
both fail with `No module named ...`. The commands below are the correct ones *once they are*,
and installing either is a package change, so it is the user's call. Test files written for this
project (`tools/test_svdq_verify.py`, the suites under `Comfy-WaveSpeed-Fixed/tests/`) therefore
carry their own runner and are executed directly:

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests-unit
```

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests -m "not inference"
```

Single test / single case:

```bash
.\python_embeded\python.exe -m pytest ComfyUI\tests-unit\path\to\test_file.py::test_name
```

Markers defined: `inference`, `execution` (deselect with `-m "not inference"`).

Lint:

```bash
.\python_embeded\python.exe -m ruff check ComfyUI
```

Verify the attention accelerators actually run kernels (import success is not proof — this does forward + numeric comparison against SDPA for Triton, Sage, FlashAttention, xformers):

```bash
.\python_embeded\python.exe -s .\_check_accel.py
```

Updates go through `update\update_comfyui.bat`; `update\update_comfyui_and_python_dependencies.bat` is destructive to the pinned stack — do not run it casually.

## Quantization pipeline

Three tools in `tools/`, all run with the embedded interpreter from the root:

```bash
.\python_embeded\python.exe .\tools\quant_audit.py
```

Recursively walks `ComfyUI/models` (metadata only — parses Safetensors headers, uses the installed GGUF reader for GGUF, safe/meta loading for `.pt`) and rewrites `quantization_inventory.json` + `quantization_inventory.md`. Rerun after producing any new output so the inventory stays current.

```bash
.\python_embeded\python.exe .\tools\quant_w4a4.py --input <src.safetensors> --profile gemma --dry-run
```

Drop `--dry-run` to write. Output name defaults to `<stem>_w4a4_convrot.safetensors` plus a `.quant.json` sidecar manifest.

```bash
.\python_embeded\python.exe .\tools\verify_w4a4.py <out.safetensors> --source <src.safetensors> --kernel-smoke
```

### Mixed precision, calibrated on real activations

A second pipeline that picks 4-bit or 8-bit **per layer** by measuring the real kernels on the
activations the model actually produced. Three tools, in order — see `W4A4_PROGRESS.md` part 9 for
the measurements and `W4A4_HANDOFF.md` for the invocation.

`tools/to_native.py` rewrites a diffusers-named checkpoint under ComfyUI's own module names.
**Not optional for Z-Image.** ComfyUI fuses `attention.to_{q,k,v}` into `attention.qkv` at load
(`comfy/model_detection.py:1498`), and only `.weight` is in that map — `weight_scale`,
`weight_s_rel` and `comfy_quant` pass through unrenamed, so the layer loads with no scale and no
error. The remap is derived from `comfy.utils.z_image_to_diffusers` and checked against ComfyUI's
own `convert_diffusers_mmdit` on meta tensors; its output is bit-identical in the latent.

`tools/calibrate_activations.py` runs real sampling with forward-pre-hooks on every candidate
Linear and reservoir-samples the real input rows (Algorithm R, so the sample is not just step 0).
Samples are stored **bfloat16, never fp16**: real Z-Image activations reach 344064 and fp16 caps
at 65504, and an `inf` there makes every error `nan`, which silently routes the worst layer in the
model to the cheapest format.

`tools/quant_mixed.py` measures `err_bf16` / `err_w4a4` / `err_w4a8` per layer against a float32
reference on those activations, then assigns `convrot_w4a4`, `asym_w4a8_int8`, or no quantization.
Mixed formats in one file are native behaviour: `comfy/utils.py:1454` turns each
`_quantization_metadata["layers"]` entry into a per-layer `<layer>.comfy_quant` tensor and
`comfy/ops.py:1142` dispatches on that layer's own JSON.

Do not choose precision by crest factor. Measured over 170 Z-Image layers its rank correlation
with the W4A4 error is **+0.10**. What correlates is the W4A8 error (+0.978).

### How the converter is built (and why)

- **Native-backend preflight.** Before touching a single tensor, `normal_comfy_backend()` spawns a subprocess that imports `comfy.quant_ops` and asks `comfy_kitchen.registry` which implementation would be selected. If either `quantize_convrot_w4a4_weight` or `convrot_w4a4_linear` does not resolve to `comfy_kitchen.backends.cuda.*`, the conversion **hard-refuses**. Keep this — it is the guard against silently producing a checkpoint that only ever runs dequantized.

  **This paragraph has now been wrong twice, in opposite directions, and the second time is the instructive one.** It first claimed `verify_w4a4.py` ran the same check; it did not. It was corrected on 2026-08-18 to say that `quant_w4a4_smooth.py` and `quant_int8.py` run **no** preflight and that smooth writes no `backend` sidecar field. All three of those became false in `d14ae48` and the paragraph did not follow — so for four days this file told readers that two converters were **less safe than they are**, which is the direction that gets acted on. Re-read against the tree on 2026-08-22: `quant_w4a4_smooth.py:182-191` imports `normal_comfy_backend` and hard-refuses, `:328-330` writes `backend` and `backend_linear`, and `quant_int8.py:187-198` preflights when `--device cuda --convrot` and prints why the `--no-convrot` path is exempt instead of pretending to.

  What is still true, and is the part worth keeping: **the converters do not all resolve the same thing.** `verify_w4a4.py` resolves only `convrot_w4a4_linear` while `quant_w4a4.py` resolves both ops — and there are **six** definitions of `normal_comfy_backend` in the tree with four different answers to "is the backend ready". Line numbers are deliberately absent here: they were `:64` and `:93-94` in this sentence and are `:70` and `:98-99` today. Grep for `def normal_comfy_backend` and count. Do not treat "the tool ran" as proof the CUDA backend was used; check the `backend` field in the sidecar. See [AUDITORIA_2026-08-18.md](AUDITORIA_2026-08-18.md) items 8 and 17, and `.scratch/varredura-2026-08-22/issues/05`.

  **Measured 2026-08-22, on the 3090, so nobody re-derives it:** the resolved implementation is *invariant* to `convrot_groupsize` and to dummy-vs-real probe tensors. All four combinations — cg 64 and 256, `torch.empty` and real quantized tensors — resolve to `comfy_kitchen.backends.cuda`, and both real calls succeed. So `quant_w4a4.py`'s hardcoded 64/64 preflight against a 256 conversion is untidy, **not** wrong; and `_native_probe.py`'s own docstring claim that dummy kwargs let the check pass where a real call would not **did not reproduce** for these two ops on this build. That is "did not reproduce under the only conditions anyone has tried", not "is false" — the mechanism at `registry.py:246` may still bite elsewhere. `tools/probe_backend_resolution.py` re-runs it.
- **Streaming writes, never mmap.** Output header offsets are computed up front, then tensors are streamed: quantized layers are read by byte range → CUDA → `ck.quantize_convrot_w4a4_weight` → written; everything else is `copy_range`'d verbatim in 16 MiB chunks. **Do not reintroduce `safe_open` / mmap for large sources.** On this Windows host mapping the 21.93 GiB Gemma source failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
- **Atomic output.** Writes go to `<output>.partial`, then `os.replace`. Refuses stale partials, refuses existing outputs/sidecars, refuses a source that already has `_quantization_metadata`.
- **Profiles are strict allowlists**, not heuristics. `PROFILE_PATTERNS` matches only `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `model.layers.N.mlp.{gate,up,down}_proj.weight`; embeddings, norms, `lm_head`, and vision towers are excluded. Only `gemma` and `qwen` exist today. **Do not extend a profile to a new architecture without confirming that architecture's loader and layer config** — Flux, Hunyuan, SeedVR2, and Z-Image each need their own recipe.
- **Group sizes.** `CONVROT_GROUP_SIZE = 256` (rotation), `QUANT_GROUP_SIZE = 64`. Layer selection requires `shape[1] % 256 == 0`. The backend-probe subprocesses in both tools use dummy `64/64` values only to resolve the implementation, which is why they differ from the real conversion values — not a bug, but don't copy those numbers into real calls.

### Output format

Standard Safetensors. Per quantized layer: `<layer>.weight` as `I8` of shape `[rows, cols/2]` (INT8 container holding signed INT4), plus `<layer>.weight_scale` as `F32` of shape `[rows]`. Everything else preserved byte-for-byte. `__metadata__` carries `_quantization_metadata` (`format_version` 1.0 + per-layer `{format: convrot_w4a4, convrot_groupsize: 256}`) and `quantization: "ConvRot W4A4"`. See [ComfyUI/QUANTIZATION.md](ComfyUI/QUANTIZATION.md) for the upstream `QuantizedTensor` / `Layout` / `MixedPrecisionOps` model this format plugs into.

**Does `.backends.cuda` in `__module__` prove the native path ran? Measured 2026-08-22: yes, here.** The whole preflight rests on that string match, and the test that settles it is cheap: W4A4 quantizes the **activation** to 4 bits too, so a dequantized-weight fallback (`F.linear(x, W_deq)`, which is what `comfy_kitchen/tensor/convrot_w4a4.py:237` does under one condition) must agree with a real call. It does not — `native` vs `W4-only` is **1.43e-1**, not 1e-6, on a `[1024, 1024]` bf16 weight at cg=256. The A4 half is real. Re-run with `tools/probe_backend_resolution.py`.

`verify_w4a4.py` checks, in order: metadata structure and per-layer dtype/shape → packed shape vs source shape → **byte-identical comparison of every preserved tensor against the source** → native backend resolution → optional real-kernel smoke against `F.linear` on the BF16 source. The smoke's relative RMSE on random inputs (~0.25 for Gemma) is a liveness signal, **not** a quality metric.

`inspect_quant.py <file>` at root is a quick header dump (tensor count, dtype histogram, metadata, scale-like keys).

## Testing a converted model

Structural verification is not acceptance. Every output must additionally pass, in order: normal ComfyUI loader compatibility (real node, not a hand-rolled load), a prompt-encoding smoke test through a long-lived ComfyUI process, and a matched-parameter benchmark against the BF16 source (identical prompt, seed, steps, resolution, sampler, scheduler, frames) recording disk, VRAM, load time, s/it, GPU utilization, power, warnings, and visual quality. Workflows for this live in `ComfyUI/user/default/workflows/` (e.g. `Video-LTX2_MultiGPU.app.json` for the Gemma text encoder).

Record negative and unsupported results rather than hiding them.

## Project tracking

- [W4A4_PROGRESS.md](W4A4_PROGRESS.md) — the running log: candidate ranking, completed steps, skipped models with reasons, failed/blocked with exact error strings. **Update it as part of the work**, not afterwards.
- [W4A4_HANDOFF.md](W4A4_HANDOFF.md) — current state snapshot and the ordered next-steps list.
- `quantization_inventory.{json,md}` — generated; do not hand-edit.

A second, separate effort also runs on this stand:

- [CORTIQ_LTX25_HANDOFF.md](CORTIQ_LTX25_HANDOFF.md) — the `cortiq` / LTX-2.5 investigation: what was **measured** (the 52× end-to-end gap, where the time goes, the PV-NT accuracy result) and the five claims that had to be withdrawn. Read this before touching `F:\cortiq-cmf`.
- [.scratch/estado-entregavel/map.md](.scratch/estado-entregavel/map.md) — the **plan**: 17 tickets across both repos, with the blocking graph. Every debt ticket carries its closing criterion, written before anyone looked at the result. `.scratch/` is the local issue tracker; nothing in it touches GitHub.

### The GPU window

Three elements, agreed 2026-08-21, and deliberately only three: **block size, how to ask, how to
let go.** A longer protocol is one nobody follows.

**The block is 30 minutes and it has a name.** Not "I need the card" — `bench:ticket25_fbcache_visual`.
The owner string is what the other side reads when it is refused, and `cortiq:bench` tells it
nothing it can plan around. Work that cannot state its block in advance does not get one: it takes
the card for a measurement, not for a session.

**Since 2026-08-22 the benchmark tools take the lock themselves, so do NOT take it for them.**
`tools/_timing.py`'s `compare()` acquires `BenchGuard`, and every timing tool now routes through it
— `m_crossover`, `attn_bench`, `attn_dtype_ab`, both `compile_*` probes, `nunchaku_compare`,
`fbcache_probe`, `fbcache_visual`. Taking `Assert-GpuLock` first and *then* running one of them
makes the tool refuse its own run:

```
F:\GPU_BENCH.lock is held: owner=bench:m_crossover_regressao_primitiva pid=74572 alive=True
Not reclaiming it. If that run is genuinely dead, delete the file by hand
```

Measured 2026-08-22, by doing exactly that. The in-process `active_guard()` lets `compare()` join a
guard the **same process** already entered; a lock held by a separate PowerShell cannot be joined,
and correctly is not. So: **take the lock by hand only for work that does not go through
`_timing.compare()`** — a converter, `verify_w4a4 --kernel-smoke`, `quant_audit`, an ad-hoc probe.
For a benchmark, just run it; it announces the acquire, the occupancy of every device, and the
release.

**Asking is `Assert-GpuLock -Owner '<repo>:<what>'`.** It throws; never `Take-GpuLock | Out-Null`,
which returns `$false`, prints "not taking", and then runs the benchmark anyway — that happened on
2026-08-19 against a live sibling with a 2-second-old heartbeat. If refused: **measure the card
itself with NVML for about a minute — the lock file is not the card.** Card idle *and* lock held is
a problem, not a wait: say so in chat ("waited X, lock held by `<owner>`") and move to work that
needs no GPU. **Do not wait, and never take a live lock silently.**

**Letting go is `Release-GpuLock`, after the work stops, never before.** Holding the lock while
idle — even announced, even for "I want the card for a comparison later" — is a false claim on a
shared card; a free lock over a busy GPU is the same lie pointing the other way. `Release-GpuLock`
removes the file **only if it is still ours**, so a lock somebody legitimately took after ours went
stale is never clobbered.

Why any of this matters: contention moves numbers. Same code, same day, `im2col 74.4 s` on a quiet
machine against `87.0 s` on a loaded one — **17%**, against a 10 s effect being measured. And on
2026-08-19 a hand-written lock named a pid that was dead the instant it was written, so the card sat
idle behind it for roughly forty minutes.

**Holding the lock is not the same as using the locked card.** Verified 2026-08-21, executed:
`F:\cortiq-cmf` selects its wgpu adapter with `request_adapter(HighPerformance)` when
`CMF_GPU_ADAPTER` is unset, and on this host that resolves to the **RTX 3080 Ti**, not the 3090.
A whole measurement session ran on `cuda:1` — which was 9-26% busy in every `nvidia-smi` sample —
while the lock sat on an idle 3090. The probe cache is what exposed it, because it stamps the
adapter name into every line:

```
0.5.95	NVIDIA GeForce RTX 3080 Ti/Vulkan	gemm-nt	gpu
```

So for any cortiq measurement: **pin the card** with `CMF_GPU_ADAPTER=3090` (index into
`cortiq gpu`, or a case-insensitive substring of the adapter name), and confirm it afterwards by
pointing `CMF_PROBE_CACHE` at a file and reading which adapter it names. A lock on the wrong card
protects nothing and reads as protection.

**The lock itself was broken until 2026-08-22, in one direction only.** `gpu_lock.py` wrote
JSON; `gpu_lock.ps1` parses `key=value`. Every regex missed, so `Get-GpuLockState` returned an
empty owner and `StaleFor = [int64]::MaxValue`, and `Take-GpuLock` **reclaimed a live lock** while
printing `reclaiming from ` with a blank owner — which is indistinguishable from a leftover file
and is the one situation where reclaiming is correct. The reverse direction always worked
(`open(path, "x")` refuses correctly), and **that asymmetry is why it survived**: one direction was
perfect, so nobody had a reason to test the other.

Fixed on the **Python** side deliberately. Teaching the `.ps1` to read JSON would have reproduced
the same theft pointing at a sibling still running the old script — a format change is not
symmetric when the other party may be running last week's code. `key=value` already had two
readers, so it won, and no sibling has to update anything. `tools/test_gpu_lock.py` drives each
implementation against a lock the *other* one wrote (23 checks, all passing), because reading
either half alone never reveals a format disagreement — which is the general lesson, not a detail
about this lock.

**Provenance of this section:** agreed with the bench owner on 2026-08-21, when only one session
held the card ("gpu liberada, so tem voce agora"). It is therefore **one side's protocol written
down, in force until contested** — not a negotiated settlement between two live sessions. The
mechanism it describes (`tools/gpu_lock.ps1`, the detached heartbeat, the 55 s staleness limit) was
executed; the *agreement* is a decision, not a measurement.

## Agent skills

Written by `/setup-matt-pocock-skills` on 2026-08-22. These three files are what the engineering
skills read as input; edit them directly rather than re-running the setup.

### Issue tracker

Local markdown under `.scratch/`, one file per ticket — this repo has **no git remote**, so there
are no GitHub issues and `gh` has nowhere to write. Every debt ticket carries its closing criterion,
written before anyone looked at the result. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).

### Triage labels

The five canonical roles, unchanged, recorded as a `Status:` line in each ticket file.
`ready-for-human` is load-bearing here: package installs, GPU windows and anything touching WSL are
the owner's call and cannot be delegated. See [docs/agents/triage-labels.md](docs/agents/triage-labels.md).

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the root, both created lazily. The provenance
rule extends to them: a glossary entry derived from reading rather than running says so in the
entry. See [docs/agents/domain.md](docs/agents/domain.md).

## Memory gotchas

Conversions check free disk (`estimate + 1 GiB`) and free RAM (`3 × largest selected tensor + 2 GiB`) and exit rather than thrash. If it refuses, the fix is to close memory-heavy WSL/worker processes **manually** — never change the pagefile or kill processes automatically.

Known benign noise: `ModelPatcher.__del__` prints an `ON_DETACH` AttributeError on short-lived interpreter shutdown; loading itself still exits 0.
