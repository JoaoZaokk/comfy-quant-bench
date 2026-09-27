import subprocess, sys
r = subprocess.run([sys.executable, "/content/qatq/qat_qwen21_blocos.py", "--smoke", "--comfy", "/content/ComfyUI", "--out", "/content/qatq/smoke"],
                   capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2500:]); print("rc", r.returncode)
