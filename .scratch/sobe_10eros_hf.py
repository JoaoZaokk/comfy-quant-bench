"""Cria JoaoZaokk/10Eros-v1.5-W4A8-ConvRot (publico) e sobe README, licenca, sidecars e os dois checkpoints.
Pedido do dono 25/09. Retomavel: pula o que o repo ja tem com o mesmo tamanho."""
import sys, time
from pathlib import Path
from huggingface_hub import HfApi
REPO = "JoaoZaokk/10Eros-v1.5-W4A8-ConvRot"
SP = Path(sys.argv[1])
CK = Path(r"P:\ComfyBench\checkpoints")
ARQ = [(SP / "README.md", "README.md"), (SP / "LICENSE_LTX_2_COMMUNITY.txt", "LICENSE_LTX_2_COMMUNITY.txt"),
       (CK / "10Eros_v1.5_bf16_w4a8.quant.json", "10Eros_v1.5_bf16_w4a8.quant.json"),
       (CK / "10Eros_v1.5_bf16_w4a8_audioint8.quant.json", "10Eros_v1.5_bf16_w4a8_audioint8.quant.json"),
       (CK / "10Eros_v1.5_bf16_w4a8_audioint8.safetensors", "10Eros_v1.5_bf16_w4a8_audioint8.safetensors"),
       (CK / "10Eros_v1.5_bf16_w4a8.safetensors", "10Eros_v1.5_bf16_w4a8.safetensors")]
api = HfApi(token=Path("F:/COMFY_PORTABLE/.hf/token").read_text().strip())  # token de escrita do dono (conteudo nao impresso)
api.create_repo(REPO, private=False, exist_ok=True)
tem = {s.rfilename: s.size for s in api.model_info(REPO, files_metadata=True).siblings}
for src, dst in ARQ:
    if tem.get(dst) == src.stat().st_size and dst.endswith(".safetensors"):
        print("ja no repo", dst, flush=True); continue
    for t in range(5):
        try:
            t0 = time.time()
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, commit_message=f"Add {dst}")
            dt = time.time() - t0
            print(f"ok {dst} {src.stat().st_size / 2**30:.2f} GiB em {dt:.0f} s ({src.stat().st_size / 2**20 / max(dt, 1):.0f} MB/s)", flush=True)
            break
        except Exception as e:
            print(f"falha {dst} tentativa {t + 1}: {type(e).__name__}: {str(e)[:300]}", flush=True); time.sleep(60)
    else:
        print(f"DESISTI {dst}", flush=True); sys.exit(1)
print("FIM", REPO, flush=True)
