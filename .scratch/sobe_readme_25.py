"""Re-sobe SO o README do card do LTX 2.5 com a correcao do mecanismo (commit, nao SMB). O token e
o do ambiente (HF_TOKEN / login em cache); este script nao cria nem grava token nenhum."""
import os
from huggingface_hub import HfApi

REPO = "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot"
api = HfApi()
me = api.whoami()
print("usuario:", me.get("name"))
r = api.upload_file(path_or_fileobj="bench/hf/ltx25-22b-w4a8/README.md", path_in_repo="README.md", repo_id=REPO,
                    repo_type="model", commit_message="README: the time columns are whole-run wall-clock, not generation speed; no speed claim between arms (corrected 2026-09-14)")
print("commit:", getattr(r, "oid", r))
info = api.model_info(REPO, files_metadata=True)
sz = {s.rfilename: s.size for s in info.siblings}
print("README.md no hub:", sz.get("README.md"), "B; local:", os.path.getsize("bench/hf/ltx25-22b-w4a8/README.md"), "B")
print("LTX25_README_OK" if sz.get("README.md") == os.path.getsize("bench/hf/ltx25-22b-w4a8/README.md") else "TAMANHO_DIFERE")
