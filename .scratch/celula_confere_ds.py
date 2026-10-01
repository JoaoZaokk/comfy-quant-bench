import subprocess
from pathlib import Path
src = Path("/content/qat/gera_dataset_klein.py").read_text()
# 2026-09-29: o gerador passou a metadados por VM (classe Metadados) e o QAT foi para qat_klein/ (professor.py)
prof = Path("/content/qat/qat_klein/professor.py")
print("codigo novo (Metadados por VM):", "class Metadados" in src, "| estat no tool:",
      prof.is_file() and "_ganchos_estat" in prof.read_text())
st = Path("/content/ds/status.json")
print("status:", st.read_text() if st.is_file() else "(sem status.json)")
print(subprocess.run(["pgrep", "-af", "gera_dataset"], capture_output=True, text=True).stdout.strip())
print(Path("/content/ds/ds.log").read_text()[-600:])
