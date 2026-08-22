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
