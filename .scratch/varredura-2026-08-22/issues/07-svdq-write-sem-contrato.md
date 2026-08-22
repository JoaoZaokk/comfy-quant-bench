# 07 - svdq_to_bf16 writes 700 lines of recovered weights with none of the atomic-write contract

Type: task
Status: resolved
Blocked by: -
Severity: medium
Provenance: TRACED (direct reading of the write path)

## Problem

`tools/svdq_to_bf16.py:674-692` is the only writer in the tree that does not obey the contract the
other six do:

- `open(partial, "wb")` at `:676`. Every other converter uses `"xb"` -- `quant_w4a4.py:285`,
  `quant_w4a8.py:373`, `quant_int8.py:270`, `quant_w4a4_smooth.py:294`, `quant_mixed.py:637`,
  `to_native.py:200`. No stale-partial refusal precedes it, so a crashed run's leftover is silently
  clobbered.
- **No length assertion.** The other six all compute `written = tell() - body_start` and raise on
  mismatch (`quant_int8.py:281-283`, `quant_mixed.py:649-651`, `quant_w4a8.py:385-387`,
  `quant_w4a4_smooth.py:305-306`, `to_native.py:219-221`, `quant_w4a4.py:321-325`). Here `offset` is
  computed during planning and never checked against what was written.
- **No `flush`/`fsync`** before `os.replace` at `:692`.
- **No `try/finally`**, so a failure leaves the partial for the next run to overwrite.

Combined: a mis-planned or short write is `os.replace`d into position and the tool prints `done:` at
`:693`. Safetensors headers are self-describing, so a file short by one tensor may **load**, deliver
garbage for the tail tensors, and produce a wrong image rather than an exception.

This is the CORRUPTED OUTPUT category, in the one converter whose output has no BF16 original to fall
back on -- the file itself says so at `:694-695`.

## Second defect in the same file

`--limit` (`:499-500`) writes a **full** output file that mixes recovered BF16 with raw SVDQuant
tensors, and the metadata declares the whole file dequantized. A `--limit 4` smoke run therefore
produces something that looks like a finished checkpoint and is not one.

## Why this is the strongest argument for ticket 08

`svdq_to_bf16.py` is not careless. It is 700 lines of genuinely hard SVDQuant recovery, and the write
path is the boring last twenty lines that got typed from memory instead of copied. A converter core
makes typing it from memory impossible.

## Closing criterion (written before the fix)

Closed when:

1. the write path uses `"xb"`, refuses a stale `.partial`, asserts bytes-written equals bytes-planned,
   `flush` + `fsync` before `os.replace`, and unlinks the partial in a `finally`;
2. `--limit` either refuses to write at all (smoke mode prints and exits) or writes a file whose
   metadata says it is partial and which `verify_*` rejects;
3. an artificially truncated write -- inject a short tensor -- raises instead of producing a file.

## Closed 2026-08-22, commit `13fbd8c`

All three criterion items met, and the reviewer produced more executed evidence than the
implementer did.

The write path now matches the contract the other six writers already obey -- `"xb"`,
stale-partial refusal, bytes-written == bytes-planned, `flush` + `fsync`, `os.replace`, and a
`finally` that unlinks. A deliberately truncated write raises instead of producing a file.

One improvement beyond exit-code cosmetics: `SystemExit` -> `RuntimeError` at `:520`. The old
`SystemExit` fired inside `open(partial, "wb")` with no `finally`, so it left the partial on disk
-- the failure path was itself creating the stale partial the next run would clobber.

Two smaller items the reviewer raised (an already-false line-number citation, and a coverage
regression) are carried to ticket 16.
