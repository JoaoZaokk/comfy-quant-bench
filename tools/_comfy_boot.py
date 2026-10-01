"""The bootstrap a tool needs before importing ComfyUI as a library -- in one place.

Around sixteen tools in this directory each hand-copy the same preamble (see `quality_ladder.py`
lines 46-70 for the long explanation, which is the reference this module was lifted from):

  1. ComfyUI's argument parser reads `sys.argv` and chokes on the TOOL's own arguments;
  2. `comfy.options.enable_args_parsing()` only ARMS the parse -- it happens at the first
     `import comfy.cli_args`, so the argv must be neutral at that moment and restored after;
  3. the neutral argv is the only place to choose ComfyUI flags such as `--bf16-unet`: the
     compute dtype is decided in that parse and not exposed afterwards.

And a fourth thing most of the copies skip: `main.py` also turns on DynamicVRAM (comfy-aimdo),
without which every weight is materialised and commit-charged in host RAM at construction --
measured on 2026-08-19 at a 44.5 GiB working set for a 39.1 GiB checkpoint (`_dynamic_vram.py`).
`boot(dinamico=True)` does that too and reports whether it came up.

    import _comfy_boot
    flags, resto = _comfy_boot.separa_flags(sys.argv[1:])
    estado = _comfy_boot.boot(flags, dinamico=True)
    import comfy.sd  # only after boot()

Created in the review of 2026-09-29. Migrating the existing copies is separate work: each of them
is a measured tool, and moving its preamble changes nothing but must still be run to be trusted.
NOT COVERED: this does not check the commit charge before a load (see `_ram_guard`), and it cannot
re-parse flags once `comfy.cli_args` was imported by someone else -- it says so instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# ComfyUI flags a tool may forward to the neutral argv. Closed list on purpose: anything else in
# the tool's argv belongs to the tool.
FLAGS_DTYPE = ("--bf16-unet", "--fp16-unet", "--fp32-unet", "--fp8_e4m3fn-unet",
               "--fp16-text-enc", "--fp32-text-enc", "--bf16-vae", "--fp16-vae", "--fp32-vae")


def separa_flags(argv: list[str], conhecidas: tuple[str, ...] = FLAGS_DTYPE) -> tuple[list[str], list[str]]:
    """(flags para o ComfyUI, resto para a ferramenta)."""
    return [a for a in argv if a in conhecidas], [a for a in argv if a not in conhecidas]


def boot(flags_comfy: list[str] | tuple[str, ...] = (), *, dinamico: bool = False,
         headroom_gib: float = 1.0) -> dict:
    """Neutraliza o argv, liga e FORCA o parse do ComfyUI com `flags_comfy`, restaura o argv.

    Devolve `{"flags", "ja_iniciado", "dynamic_vram"}`. `ja_iniciado=True` quer dizer que alguem
    importou `comfy.cli_args` antes: o parse ja aconteceu e `flags_comfy` NAO foi aplicado -- quem
    chama decide se isso e fatal. `dynamic_vram` e None sem `dinamico`, senao o retorno de
    `_dynamic_vram.enable()` (False e informacao, nao erro: o perfil de memoria muda)."""
    for p in (str(RAIZ / "ComfyUI"), str(Path(__file__).resolve().parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    estado = {"flags": list(flags_comfy), "ja_iniciado": "comfy.cli_args" in sys.modules,
              "dynamic_vram": None}
    if not estado["ja_iniciado"]:
        salvo = sys.argv[:]
        sys.argv = ["main.py", *flags_comfy]
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import comfy.cli_args  # e aqui que o parse acontece
        finally:
            sys.argv = salvo
    if dinamico:
        from _dynamic_vram import enable
        estado["dynamic_vram"] = enable(headroom_gib)
    return estado
