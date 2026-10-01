"""Fila do criterio_mestre_dtype_2026-09-22.md: dois ajustes (fp32, bf16-sr) sob um BenchGuard."""
import subprocess, sys
from pathlib import Path
R = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(R / "tools"))
from _bench_guard import BenchGuard
O = "P:/ComfyBench/originais"
with BenchGuard("bench:klein4b_mestre_dtype") as g:
    if g.refused:
        print(g.refused); raise SystemExit(1)
    for mestre, nome in (("fp32", "klein4b_braco1f_mestre_fp32"), ("bf16-sr", "klein4b_braco1s_bf16_sr")):
        cmd = [str(R / "python_embeded/python.exe"), "-s", str(R / "tools/ajusta_denso_diffusers.py"),
               "--raiz", f"{O}/FLUX.2-klein-4B",
               "--ref-transformer", "F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors",
               "--aluno-transformer", f"{O}/klein4b_ternario_ingenuo.safetensors",
               "--saida", f"{O}/{nome}.safetensors", "--mestre", mestre, "--device", "0"]
        with open(R / f"bench/render_braco1_2026-09-22/ajuste_{mestre}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(R))
        print(f"ajuste {mestre} rc={r.returncode}", flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
print("FIM ajustes")
