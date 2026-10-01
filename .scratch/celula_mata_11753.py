import subprocess, time
subprocess.run(["kill", "11753"]); time.sleep(5)
print(subprocess.run(["pgrep", "-af", "gera_dataset"], capture_output=True, text=True).stdout.strip())
