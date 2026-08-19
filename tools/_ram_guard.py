"""How much RAM a two-pass converter really needs, instead of the streaming heuristic.

`quant_w4a4.py` streams: it reads one tensor, quantizes it, writes it, and drops it. For that
design `largest * 3 + 2 GiB` is a fair estimate of peak RAM, and it is correct there.

Three other converters copied the line into a **two-pass** design that quantizes every selected
layer into a dict first and only then writes the file. Those hold the entire quantized model in
RAM at once, so the guard understates the requirement by the number of layers. Measured by the
audit of 2026-08-18:

    quant_int8.py   asks 2.375 GiB, accumulates 19.144 GiB on LTX-2.5   (8.1x)
    quant_w4a8.py   asks 2.375 GiB, accumulates 10.777 GiB              (4.5x)
    quant_mixed.py  asks 2.250 GiB, accumulates  2.920 GiB on the *smallest* model here

A guard that passes and then thrashes is worse than no guard: it converts a clear refusal into a
machine that stops responding. Sizes here are computed from the header alone -- no tensor is
read -- so calling this costs nothing.
"""

from __future__ import annotations

GIB = 1024 ** 3


def convrot_w4a4_bytes(rows: int, cols: int) -> int:
    """int8 container holding packed int4 [rows, cols//2] + f32 scale [rows]."""
    return rows * (cols // 2) + rows * 4


def w4a8_bytes(rows: int, cols: int, group_size: int = 16, codebook: bool = True) -> int:
    """packed int4 + fp8 per-group scale + f32 per-channel scale + 16-entry codebook."""
    return (rows * (cols // 2)
            + rows * (cols // group_size)      # s_rel, fp8 => 1 byte per group
            + rows * 4                          # s_channel, f32
            + (16 * 4 if codebook else 0))


def int8_bytes(rows: int, cols: int) -> int:
    """int8 weight + f32 scale per row."""
    return rows * cols + rows * 4


def check(available_bytes: int, accumulated_bytes: int, headroom_gib: float = 2.0,
          label: str = "conversion") -> str | None:
    """Return a refusal message, or None when there is room.

    `headroom_gib` covers the one source tensor held during quantization plus interpreter
    overhead; it is deliberately generous because the failure mode on the other side is swapping,
    not an exception.
    """
    need = accumulated_bytes + int(headroom_gib * GIB)
    if available_bytes >= need:
        return None
    return (f"Insufficient RAM for {label}: this design accumulates "
            f"{accumulated_bytes / GIB:.2f} GiB of quantized tensors before writing, and with "
            f"{headroom_gib:.1f} GiB of headroom needs {need / GIB:.2f} GiB, but only "
            f"{available_bytes / GIB:.2f} GiB is available. Close memory-heavy processes "
            "(vmmemWSL on this host is the usual one) and retry -- do not change the pagefile.")
