# 16 - Round-1 leftovers: what eight reviewers asked for and I did not do

Type: task
Status: ready-for-agent
Blocked by: -
Severity: low
Provenance: TRACED (the reviewers ran their own checks; I did not re-run these particular items)

## Problem

Round 1 closed eight tickets and every reviewer returned **SHIP WITH FIXES**. Five of those fixes
were regressions the round introduced and were handled before `13fbd8c`. The rest are real but
smaller, and they are listed here rather than silently dropped -- the whole point of a review is
lost if only the blocking half of it survives.

Full reviewer text is not in the repo. These are the items, by owner:

### ltx_studio (ticket 03)
- **A decision, not a defect:** an *absent* `Sec-Fetch-Site` is allowed. Rejecting it would make
  a plain `curl` `POST {}` return 403 instead of the 400 the criterion asks for, so criteria 3 and
  4 only reconcile if absent means allowed. The browsers this exempts predate 2020. The reviewer
  recommends ratifying it; the code says so at `:585-597`. **The owner's call.**
- `Handler.timeout` is unset, so a client declaring `Content-Length: 5000` and sending one byte
  parks a worker thread indefinitely. Pre-existing, outside the criterion.

### hf_parallel_get (ticket 11)
- Throughput display oscillates 0.53x-1.00x mid-wave. The broken per-chunk bookkeeping was
  correctly deleted; nothing replaced it.
- **A fail-open by design:** no digest available -> WARNING -> return 0, and `fetch_ltx25` then
  prints `OK` in its summary table for a file nothing verified. The WARNING is loud in
  `download()`'s output and invisible in the caller's table. Decide whether "completed but
  unverified" should be a distinct signal the summary reads.

### svdq_to_bf16 (ticket 07)
- A line-number citation in the new comments was already false when written.
- A coverage regression the implementer did not name -- see the reviewer's item 3.

### quant_audit (ticket 15)
- Smaller items listed under "Smaller, real" in that review.

### comfy-quant-preflight (ticket 14)
- `scope_line` names unchecked *classes*. It should name unchecked **widgets**: that is the
  general form of the `DualCLIPLoader.clip_name2` fail-open already fixed, and the general form is
  what stops the next one.

### docs (ticket 13)
- `artefatos-em-comfyui.md` records `8945 bytes / 207 linhas` for the tracked `__init__.py`,
  measured during the round -- and a concurrent agent grew that file to 17067/342 in the same
  round. The `stat` command is printed alongside so a reader can recheck, which is the right
  shape; but it is a live demonstration that **this ticket's disease is not cured by writing a
  fresher number.** Where a count exists only to be quoted, the command should replace it.

## Closing criterion (written before the work)

Closed when each item above is either done, or has one line here saying it was declined and why.
The two marked **decision** are the owner's and do not block the rest.

Explicitly NOT part of this ticket: re-running round 1. The suites are green
(`gpu_lock` 23, `ltx_studio` all, `hf_parallel_get` 7/7, `quant_mixed_provenance` 9/9,
`verify_formats` 13/13, preflight 30/0) and re-running them is not what these items need.

## Round 2 status, 2026-08-22, commit `4480337`

**Done:** `ltx_studio` request timeout; `hf_parallel_get`'s no-digest fail-open now returns a
distinct "completed but unverified" signal via an `outcome` dict so a caller's summary cannot print
a bare `OK` for it; throughput accounting; the `svdq_to_bf16` false line-number citation and its
named coverage regression; `quant_audit`'s smaller items; the preflight scope line now names
unchecked **widgets**.

**Deliberately not touched:** `ltx_studio`'s absent-`Sec-Fetch-Site` allowance. Still the owner's
decision, still documented in the code.

### New items, from the round-2 reviews

- **`scope_line`'s `opened` verb was over-claiming, and I fixed it** -- see below -- but two
  related things remain: `_inject`'s `stale` warning still says "N loader class(es)" while the unit
  is now the entry, and `covered` is populated from the `audit_failure` branch, which installs on
  widget names nobody confirmed exist. So the scope line can vouch for a widget that was never
  audited.
- **`_pairs` accepting bare strings** means a raw prompt `dict` passed by mistake iterates its keys
  and prints node IDs as node types. Silent nonsense rather than a raise. The trade (never raise
  inside `validate_prompt`) is right; the dict shape was not considered.
- **`tools/test_svdq_verify.py` has no owner.** The `"xb"` / `written == planned` / `fsync` /
  `finally` contract on the only writer that lacked it has **zero re-running coverage**, and
  `--limit` is no longer a cheap route to it. Needs no GPU.
- **`_native_probe`'s recipes hardcode `bfloat16`** while `quant_w4a4` quantizes at the source
  tensor's own dtype and `HIGH_PRECISION_DTYPES` admits `F16`/`F32`. Not a regression -- the old
  probes hardcoded `float16` -- but not "the real kwargs the caller is about to use" either.
- **`m_crossover` on the new primitive is unverified against its own prior numbers.** Needs a GPU
  window. This is the highest-value remaining item in the whole effort: the primitive is only
  worth having if it did not change what the tool measures.

### What I fixed on top of round 2, and why it is the same shape as round 1

`scope_line` said **`opened N file widget(s)`** -- a claim about files. `_make_validator` opens
only `.safetensors`; `MODEL_FILE_SUFFIXES` recognises seven. So six of the seven suffixes reached
the `checked` bucket and the line vouched for files nothing ever read. Not hypothetical: **35 of
the 159 files in the inventory are non-safetensors** (10 `.gguf`, 14 `.pth`, 7 `.onnx`, 4 `.pt`,
3 `.ckpt`, 1 `.bin`), and a `.gguf` UNET is a normal workflow here.

The test meant to catch it varied the widget **name** and held the extension constant. Round 1's
digest test varied tensor **shape** while the collision lived in identical shapes. **A test that
varies the axis the bug is not on** has now cost three fixtures, and the new test says so in its
own docstring.

Both sides now read one `CHECKED_FILE_SUFFIXES` tuple, and a test asserts the validator cannot go
back to its own literal.

## The headline item, settled by execution 2026-08-22

**`m_crossover` on the new primitive reproduces its own recorded numbers.** This was the highest
value item left in the effort: a timing primitive is only worth having if it did not change what
the tool measures, and "unchanged by reading" is not that claim.

3090, `CUDA_VISIBLE_DEVICES=0`, `--repeats 3`, against part 11 of `W4A4_PROGRESS.md` (2026-08-19):

| | recorded 2026-08-19 | now | |
|---|---|---|---|
| `[3840,3840]` M=5856 | 4,89x [4,72-5,06] | **4.93x [4.74-5.03]** | intervals overlap |
| `[3840,3840]` M=8192 | 5,10x [4,97-5,15] | **5.04x [5.03-5.08]** | overlap |
| `[10240,3840]` M=5856 | 5,65x [5,54-5,75] | **5.68x [5.67-5.73]** | overlap |
| `[10240,3840]` M=8192 | 5,75x [5,68-5,76] | **5.66x [5.51-5.68]** | touching |
| M=1 `[3840,3840]` | 1,5-2,0x slower | **1.81x slower** | inside |

**The decisive column is not the timings.** The relative errors came back identical to the fourth
decimal -- `w4a4 0.2231`, `w4a4/a8 0.1574`, `w4a8 0.0737` at M=5856 -- against exactly those values
recorded on 2026-08-19. The refactor moved the stopwatch and did not touch the numerics.

One difference worth naming rather than burying: on `[10240,3840]` the crossover reads between
M=128 and M=256 today, where part 11 recorded between M=64 and M=128. M=128 measures
`1.18x slower [0.88-1.83]` -- an interval that **contains 1.0**. Part 11 already said it: *"perto do
cruzamento o vencedor nao e confiavel... abaixo de M~512 vale ler empate, nao o rotulo."* The
bracket is what makes that readable instead of a moved verdict.

`attn_bench` and `attn_dtype_ab` also ran for the first time on the primitive. Both work, both take
the lock themselves, both print every device's occupancy, and both now show ties as ties --
`flash_attn` bf16-vs-fp16 reads `1.08x faster [0.99-1.09]`, which without the bracket would have
been quoted as an 8% win the data does not support.

## Two defects this run exposed

**1. The documented GPU protocol now collides with the tools.** `CLAUDE.md` says to
`Assert-GpuLock` before GPU work. Do that and then run a benchmark and the benchmark refuses its
own run -- `compare()`'s `active_guard()` can join a guard the *same process* entered, and a lock
held by a separate PowerShell cannot be joined. Measured by doing it. CLAUDE.md corrected: take the
lock by hand only for work that does not go through `_timing.compare()`.

**2. `_timing.py` overstated its own founding measurement, in its own docstring.** It said the
first-burst bias was "systematically ~3% high, every time" -- signed. The first two real sweeps
contradict it:

    attn_bench    discarded burst FASTER than the kept median on all three paths
    m_crossover   FASTER for bf16 and w4a8, SLOWER for w4a4 (1.782 vs 1.629) -- same run

The first burst is an outlier, which is why discarding it is right. The *direction* is not fixed,
and is not even fixed across the paths of one sweep. Quoting a signed percentage from one A/B is
the same overstatement this module exists to make impossible in a ratio, committed in the module's
own docstring. Corrected, with both runs quoted.

## Still open in this ticket

- ~~`tools/test_svdq_verify.py` has no owner~~ **DONE 2026-08-22.** The contract now has
  `tools/test_svdq_write_contract.py` -- 21 checks, no GPU, no CUDA, no multi-GiB source. It lives
  apart from `test_svdq_verify.py` on purpose: that file's `__main__` runs a GPU battery, so it
  cannot host a check whose whole value is being runnable *while a sibling holds the card*.
  Covered: an honest write round-trips byte-identically through `load_file`; a write short against
  its plan raises `RuntimeError(length mismatch: wrote 48, planned 64)` and leaves **neither**
  `out` nor `.partial`; a pre-existing `.partial` raises `FileExistsError` rather than being
  clobbered; `main()` refuses a stale partial before it reaches the writer; and all seven writers
  still carry `"xb"`.
  Its own first version FAILED, and correctly: it searched for `write_checkpoint(src` and matched
  the **definition** four hundred lines above the call, then reported the guard as coming after the
  writer. The test was wrong and the code was right. Fixed by making the needle unambiguous rather
  than by loosening the assertion until it passed.
- `_native_probe`'s recipes hardcode `bfloat16` while `quant_w4a4` quantizes at the source tensor's
  own dtype and `HIGH_PRECISION_DTYPES` admits `F16`/`F32`.
- ~~`_inject`'s `stale` warning counts "loader class(es)"~~ **DONE.** It counts table
  **entries** now, singular/plural correct -- two renamed widgets on one class printed "2 loader
  class(es)" where there was one class.
- ~~`covered` is populated from the `audit_failure` branch~~ **DONE.** The classes installed on
  unconfirmed widget names are tracked separately in `_STATUS["unaudited"]` and the scope line
  appends `WIDGET NAMES NOT CONFIRMED against INPUT_TYPES for N: ...`. The fail-open stays --
  declining to install would turn an unreadable class into an unchecked one -- but its cost now
  travels into the one sentence an operator reads. Only classes present in *this* graph are named:
  the line describes the run, not the registry.
- ~~`_pairs` accepting bare strings~~ **DONE.** A raw prompt `dict` used to iterate its KEYS, so
  every node id printed as a class type -- wrong and confident, which is worse than the
  `ValueError` that function exists to avoid. `{id: {class_type, inputs}}` and a bare node object
  are both unwrapped now, and a non-string class type is dropped rather than rendered.
- `ltx_studio`'s absent-`Sec-Fetch-Site` allowance -- the owner's decision, unchanged.
