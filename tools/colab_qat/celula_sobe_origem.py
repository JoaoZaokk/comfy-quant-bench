import time, torch
from huggingface_hub import HfApi
p = "/content/inicio/ckpt/ultimo.pt"
print("passo da origem:", torch.load(p, map_location="cpu", weights_only=False, mmap=True)["passo"])
t = time.time()
HfApi().upload_file(path_or_fileobj=p, path_in_repo="origem/ckpt_origem.pt", repo_id="JoaoZaokk/klein4b-qat-replay")
print("origem enviada em %.0f s" % (time.time() - t))
