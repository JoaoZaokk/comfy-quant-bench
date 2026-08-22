# 14 - comfy-quant-preflight covers four loader widgets and fails open on every upstream rename

Type: task
Status: resolved
Blocked by: -
Severity: low
Provenance: TRACED

## Problem

`custom_nodes/comfy-quant-preflight/` is the right idea, argued well at `__init__.py:22-25` -- it hooks
`VALIDATE_INPUTS` and wraps `validate_prompt` rather than forking, because *"This fails loudly and a
fork fails quietly."* The implementation does not hold that line:

- `__init__.py:51` -- covers 4 of the loader widgets that exist, and does not say which it skipped. A
  workflow using an uncovered loader gets a clean pass.
- `__init__.py:56` -- `_resolve()` swallows every exception, turning an upstream `folder_paths` change
  into "no problem found".
- `__init__.py:122` -- injection reports success while being unreachable if a loader migrates.
- `__init__.py:201` -- installs behind a swallowing `except`, and silently declines.
- `checks.py:266` -- the Nunchaku / dynamic-VRAM check fails open on a renamed upstream module or flag.
- `checks.py:148` -- the `full_precision_matrix_mult` check states as fact something that was not
  confirmed by execution, which is precisely what CLAUDE.md says these two checks must not do.

Every one of these is a **fail-open** in a component whose entire purpose is to refuse. The pattern is
the same as ticket 05: the failure mode is a green light, not a red one.

## Closing criterion (written before the fix)

Closed when:

1. the package prints, on every run, which loaders it covered and which node types it saw and did not
   cover -- so a clean pass states its own scope;
2. no `except` in the package swallows without at least a WARN naming the exception;
3. an upstream symbol that cannot be resolved produces a WARN that says the check did not run, never a
   silent pass;
4. `checks.py:148` carries "not confirmed by execution" in the message itself, matching ticket 02;
5. `test_checks.py` gains a case per fail-open path, asserting the WARN is emitted.

## Closed 2026-08-22, commit `13fbd8c`

Every fail-open in the ticket closed: the package states its own scope on every run, no `except`
swallows without a WARN naming the exception, an unresolvable upstream symbol produces a WARN that
says the check did not run, and `test_checks.py` gained a case per path (30 PASS, 0 FAIL).

Criterion item 4 met: `check_full_precision_matrix_mult` carries "not confirmed by execution" in
the message itself and states the consequence as *would*, not *does*. The docstring says why, and
it is the general rule rather than a note about this check: *the person who reads it in the UI does
not have this docstring in front of them, and a caveat that lives only next to the source is a
caveat that does not travel.*

One correction applied after review, and the change had **created** it: `DualCLIPLoader` has two
file widgets and only `clip_name1` was listed, so a quantized encoder in slot 2 went unchecked
while the class was reported as covered. A fail-open the coverage line actively concealed, which
is worse than one it merely misses. Second entry added.

Carried to ticket 16: `scope_line` should name unchecked **widgets**, not unchecked classes. That
is the general form of the same bug, and the general form is what stops the next one.
