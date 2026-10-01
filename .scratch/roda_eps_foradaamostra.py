"""Fila do epsilon fora da amostra (criterio_render_braco1_2026-09-22.md, bloco E).

O probe_epsilon_ckpt_ab NAO toma o lock sozinho, entao este driver segura o BenchGuard e chama o
probe como subprocesso, uma corrida por (prompt, semente). Chave de 'semente' no log =
100*indice_do_prompt + semente, porque o agregador indexa por semente e aqui ha 5 prompts.
"""
import os, subprocess, sys
from pathlib import Path
R = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(R / "tools"))
from _bench_guard import BenchGuard

prompts = [l.strip() for l in (R / "bench/render_braco1_2026-09-22/prompts.txt").read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("#")]
arms = ["braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors",
        "braco1_ajuste=klein4b_braco1_compensado_bfl.safetensors",
        "braco2_bonsai=klein4b_braco2_bonsai_ternario_bfl.safetensors"]
out_dir = R / "bench/render_braco1_2026-09-22"
with BenchGuard("bench:klein4b_eps_foradaamostra") as g:
    if g.refused:
        print(g.refused); raise SystemExit(1)
    for pi, pr in enumerate(prompts):
        log = out_dir / ("eps_dentro.log" if pi == 4 else "eps_fora.log")
        for s in (11, 12):
            cmd = [str(R / "python_embeded/python.exe"), "-s", str(R / "tools/probe_epsilon_ckpt_ab.py"),
                   "--ref-unet", "klein4b_braco3_bf16_original_bfl.safetensors",
                   *sum((["--arm", a] for a in arms), []),
                   "--clip", "qwen_3_4b.safetensors", "--clip-type", "flux2",
                   "--prompt", pr, "--seed", str(s), "--steps", "8", "--cfg", "1.0",
                   "--size", "1024", "--device", "0"]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(R))
            with log.open("a", encoding="utf-8") as f:
                f.write(f"########## SEMENTE {100 * pi + s} ##########\n# prompt {pi}: {pr}\n")
                f.write(r.stdout)
                if r.returncode:
                    f.write(f"\n!!! rc={r.returncode}\n{r.stderr[-3000:]}\n")
            print(f"p{pi} s{s} rc={r.returncode}", flush=True)
            if r.returncode:
                raise SystemExit(r.returncode)
print("FIM eps")
