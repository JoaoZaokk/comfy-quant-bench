from pathlib import Path
print(Path("/content/qatq/run/qat.log").read_text()[-4000:])
