from pathlib import Path

T = Path(__file__).resolve().parent
topo = (T / "perfis_topo.py").read_text(encoding="utf-8")
bloco = (T / "bloco_perfis.txt").read_text(encoding="utf-8")
fim = (T / "perfis_fim.py").read_text(encoding="utf-8")
assert bloco.startswith("PROFILE_PATTERNS = {")
bloco = bloco.replace("PROFILE_PATTERNS = {", "MODULE_PATTERNS = {", 1)
bloco = bloco.replace("PROFILE_FILE_PATTERNS = dict(PROFILE_PATTERNS)", "FILE_PATTERNS = dict(MODULE_PATTERNS)")
bloco = bloco.replace("PROFILE_FILE_PATTERNS[", "FILE_PATTERNS[")
bloco = bloco.replace("the rest fall back to PROFILE_PATTERNS.", "the rest fall back to MODULE_PATTERNS.")
assert "PROFILE_" not in bloco, [l for l in bloco.splitlines() if "PROFILE_" in l]
Path("F:/COMFY_PORTABLE/tools/_profiles.py").write_text(topo + bloco + fim, encoding="utf-8", newline="\n")
print("ok")
