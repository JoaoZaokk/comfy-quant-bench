import os
from huggingface_hub import HfApi
a = HfApi()
w = a.whoami()
print("conta", w.get("name"), "| isPro", w.get("isPro"), "| plano", (w.get("auth") or {}).get("accessToken", {}).get("role"))
os.makedirs("/content/diag", exist_ok=True)
with open("/content/diag/teste_50mb.bin", "wb") as f:
    f.write(os.urandom(50 * 2**20))
try:
    a.upload_file(path_or_fileobj="/content/diag/teste_50mb.bin", path_in_repo="diag/teste_50mb.bin",
                  repo_id="JoaoZaokk/klein4b-qat-ckpt")
    print("upload 50 MB OK")
    a.delete_file("diag/teste_50mb.bin", repo_id="JoaoZaokk/klein4b-qat-ckpt")
    print("apagado")
except Exception as e:
    print("ERRO COMPLETO:", type(e).__name__, str(e)[:3000])
