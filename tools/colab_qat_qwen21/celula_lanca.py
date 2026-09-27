"""Celula (colab exec): sobe o QAT Qwen 2.1 em background (setsid nohup) e retorna. O supervisor com
probe_qat_qwen21.py a cada 4 min e obrigatorio depois disto (kernel ocioso = VM podada)."""
import json, subprocess, sys
from pathlib import Path
Q = Path("/content/qatq")
cfg = json.loads((Q / "config.json").read_text())
d = Path(cfg["dir"]); d.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "-u", str(Q / "qat_qwen21_blocos.py"), "--comfy", "/content/ComfyUI",
       "--dit", "/content/qwen21/diffusion_models/qwen_image_2.1_bf16.safetensors",
       "--conds", str(Q / "conds_qwen21.pt"), "--out", str(d), *cfg.get("args", [])]
print("CMD", " ".join(cmd))
log = open(d / "qat.log", "ab")
p = subprocess.Popen(["setsid", "nohup", *cmd], stdout=log, stderr=subprocess.STDOUT, cwd=str(Q), start_new_session=True)
print(f"LANCADO pid={p.pid} log={d / 'qat.log'}")
