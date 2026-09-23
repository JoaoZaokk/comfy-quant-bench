import subprocess, time
subprocess.run(["pkill", "-f", "qat_ternario_klein.py"])
time.sleep(5)
vivo = subprocess.run(["pgrep", "-f", "[q]at_ternario_klein.py"], stdout=subprocess.DEVNULL).returncode == 0
import glob
print("job vivo" if vivo else "job parado", "| shards professor:", len(glob.glob("/content/qat_run/professor/*.pt")))
