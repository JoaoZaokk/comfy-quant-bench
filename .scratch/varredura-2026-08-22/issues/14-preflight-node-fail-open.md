# 14 - comfy-quant-preflight covers four loader widgets and fails open on every upstream rename

Type: task
Status: ready-for-agent
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
