# Protocolo GPU e histórico do lock

> Referência preservada do CLAUDE.md original, linhas 1340–1423, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

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

