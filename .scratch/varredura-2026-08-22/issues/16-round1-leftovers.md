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
