from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/test_verify_formats.py")
s = p.read_text(encoding="utf-8")


def troca(velho: str, novo: str) -> None:
    global s
    assert s.count(velho) == 1, velho[:80]
    s = s.replace(velho, novo)


troca('ELEMENT_SIZE = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4}',
      'ELEMENT_SIZE = {"I8": 1, "U8": 1, "F8_E4M3": 1, "F16": 2, "BF16": 2, "F32": 4}')
troca('''            f"{layer}.weight_scale": ("F32", [ROWS, 1], filler("F32", [ROWS, 1], seed + 1)),
        }
    raise AssertionError(fmt_name)
''', '''            f"{layer}.weight_scale": ("F32", [ROWS, 1], filler("F32", [ROWS, 1], seed + 1)),
        }
    if fmt_name == "awq_w4a16":
        # quant_awq_w4a16.py (`_formats.AwqW4A16`, group 32): escala e zeros [K/G, N] em BF16.
        groups = COLS // AWQ_GROUP
        return {
            f"{layer}.weight": ("I8", [ROWS, COLS // 2], filler("I8", [ROWS, COLS // 2], seed)),
            f"{layer}.weight_scale": ("BF16", [groups, ROWS], filler("BF16", [groups, ROWS], seed + 1)),
            f"{layer}.weight_zeros": ("BF16", [groups, ROWS], filler("BF16", [groups, ROWS], seed + 2)),
        }
    if fmt_name == "float8_e4m3fn":
        # quant_te_residuos.py --format fp8: peso fp8 na forma da fonte + UMA escala escalar.
        return {
            f"{layer}.weight": ("F8_E4M3", [ROWS, COLS], filler("F8_E4M3", [ROWS, COLS], seed)),
            f"{layer}.weight_scale": ("F32", [], filler("F32", [], seed + 1)),
        }
    raise AssertionError(fmt_name)
''')
troca('''    "int8_tensorwise": {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": CONVROT},
}
''', '''    "int8_tensorwise": {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": CONVROT},
    "awq_w4a16": {"format": "awq_w4a16", "group_size": AWQ_GROUP},
    "float8_e4m3fn": {"format": "float8_e4m3fn"},
}
''')
troca("ROWS, COLS, GROUP_SIZE, CONVROT = 8, 256, 16, 256\n",
      "ROWS, COLS, GROUP_SIZE, CONVROT = 8, 256, 16, 256\nAWQ_GROUP = 32\n")
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
