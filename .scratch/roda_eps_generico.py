"""Epsilon fora da amostra, protocolo do roda_eps_qatA100b.py (F0-F3, sementes 11 12, 1024 px), com os
bracos e o nome do log na linha de comando:

    roda_eps_generico.py <nome_log> rotulo=arquivo_bfl.safetensors [rotulo=...]

Chave de 'semente' no log = 100*indice_do_prompt + semente (o agregador indexa por semente).
Segura o BenchGuard e chama o probe como subprocesso, uma corrida por (prompt, semente).
"""
import subprocess
import sys
from pathlib import Path

R = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(R / "tools"))
from _bench_guard import BenchGuard  # noqa: E402

nome, arms = sys.argv[1], sys.argv[2:]
prompts = [ln.strip() for ln in (R / "bench/render_braco1_2026-09-22/prompts.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")]
log = R / "bench/qat_klein" / f"eps_{nome}.log"
with BenchGuard(f"bench:klein4b_eps_{nome}") as g:
    if g.refused:
        print(g.refused)
        raise SystemExit(1)
    for pi, pr in enumerate(prompts[:4]):
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
print(f"FIM eps {nome}")
