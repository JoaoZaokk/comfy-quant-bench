"""Avaliacao cega das imagens da bateria Qwen-Image-2.1 pelo MiMo V2.6 Pro (opencode run).

Duas rodadas, sem o nome do build em lugar nenhum que o modelo veja:
  nota:    cada imagem sozinha, copiada com nome neutro, rubrica fixa, resposta em JSON.
  ranking: as N imagens de um mesmo prompt/seed juntas, em ordem embaralhada fixa por (prompt, seed).
bf16 e bf16junto saem identicas (mesmos pesos), entao o par e o controle de ruido do juiz: a diferenca de nota
entre duas copias da mesma imagem mede quanto o juiz varia sozinho.

    python_embeded\\python.exe -s avalia_mimo.py <saida_dir> <dit> [<dit> ...]
Retoma: respostas ja gravadas em <saida_dir>/respostas/*.json nao sao repetidas.
"""
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ast

# monta_bateria.py grava os grafos ao ser importado; so as constantes interessam aqui
_arv = ast.parse((Path(__file__).parent / "monta_bateria.py").read_text(encoding="utf-8"))
_const = {n.targets[0].id: ast.literal_eval(n.value) for n in _arv.body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("PROMPTS", "SEEDS")}
PROMPTS, SEEDS = _const["PROMPTS"], _const["SEEDS"]

MODELO = "xiaomi-token-plan-sgp/mimo-v2.6-pro"
IMAGENS = Path("F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria")
OPENCODE = shutil.which("opencode")
SAL = "qwen21-cego-2026-09-26"

RUBRICA = """You are a strict image-quality judge for a text-to-image model evaluation.
The image was generated from this prompt:
<<{prompt}>>

Score the image on each criterion from 0 to 10 (10 = perfect). Be critical and use the full range; do not give 10 unless flawless.
- prompt_adherence: every element of the prompt is present and correct (objects, counts, colors, style, composition).
- text_accuracy: if the prompt asks for written text, is it spelled exactly and legible? Use null if the prompt has no text.
- artifacts: 10 = no generation artifacts; lower for noise, banding, grid/checker patterns, color blotches, smearing, oversharpening, neon/posterized fringes, broken textures.
- anatomy: hands, faces, animals and objects are structurally correct. Use null if nothing applies.
- detail_realism: fine detail, texture and lighting plausibility for the requested style.
- overall: overall quality as a final deliverable.
List concrete defects you see (short phrases, max 6).
Reply with ONLY one JSON object, no markdown fences:
{{"prompt_adherence": n, "text_accuracy": n|null, "artifacts": n, "anatomy": n|null, "detail_realism": n, "overall": n, "defects": ["..."]}}"""

RANKING = """You are a strict image-quality judge. The {n} attached images (in attachment order: Image 1 .. Image {n}) were all generated from the same prompt and seed by different versions of one model:
<<{prompt}>>

Some images may be pixel-identical; if two are identical, say so and give them adjacent ranks.
Compare them for prompt adherence, text spelling (if any), artifacts (noise, banding, grid patterns, color blotches, neon fringes), anatomy and fine detail.
Rank all images from best to worst. For each image give a 0-10 quality score and its main defects.
Reply with ONLY one JSON object, no markdown fences:
{{"ranking": [best_image_number, ..., worst_image_number], "scores": {{"1": n, "2": n, ...}}, "identical_pairs": [[a, b], ...], "defects": {{"1": ["..."], ...}}, "reason": "one sentence"}}"""


def neutro(dit, i, s):
    return hashlib.sha256(f"{SAL}|{dit}|{i}|{s}".encode()).hexdigest()[:12]


def chama(mensagem, arquivos):
    cmd = [OPENCODE, "run", "--pure", "-m", MODELO, mensagem]
    for a in arquivos:
        cmd += ["-f", str(a)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
                       cwd=Path(__file__).parent / "mimo")
    texto = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)
    achado = None
    for m in re.finditer(r"\{.*\}", texto, re.S):
        achado = m.group(0)
    if achado is None:
        raise RuntimeError(f"sem JSON (rc={p.returncode}): {texto[-400:]} {p.stderr[-400:]}")
    return json.loads(achado), texto


def tarefa(caminho, mensagem, arquivos, extra):
    if caminho.exists():
        return
    for tentativa in range(3):
        try:
            resp, bruto = chama(mensagem, arquivos)
            caminho.write_text(json.dumps({**extra, "resposta": resp, "bruto": bruto[-4000:]}, indent=1), encoding="utf-8")
            print("OK", caminho.name, flush=True)
            return
        except Exception as e:  # rede/JSON: tenta de novo, depois registra
            print("ERRO", caminho.name, tentativa, str(e)[:300], flush=True)
    caminho.with_suffix(".falhou").write_text("3 tentativas", encoding="utf-8")


def main():
    global IMAGENS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("saida", type=Path)
    ap.add_argument("dits", nargs="+")
    ap.add_argument("--raiz", type=Path, default=IMAGENS)
    ap.add_argument("--prompts", default=",".join(str(i) for i in range(len(PROMPTS))))
    ap.add_argument("--sem-nota", action="store_true", help="so o ranking")
    a = ap.parse_args()
    saida, dits, IMAGENS = a.saida, a.dits, a.raiz
    indices = [int(x) for x in a.prompts.split(",")]
    cego = saida / "cego"
    resp = saida / "respostas"
    for d in (cego, resp, Path(__file__).parent / "mimo"):
        d.mkdir(parents=True, exist_ok=True)
    mapa = {}
    for dit in dits:
        for i in indices:
            for s in SEEDS:
                src = IMAGENS / dit / f"p{i}_s{s}_00001_.png"
                if not src.exists():
                    raise SystemExit(f"falta {src}")
                n = neutro(dit, i, s)
                dst = cego / f"{n}.png"
                if not dst.exists():
                    # regravado so com os pixels: o PNG do ComfyUI leva o grafo (com o nome do DiT) nos metadados
                    from PIL import Image
                    Image.open(src).convert("RGB").save(dst)
                mapa[n] = {"dit": dit, "prompt": i, "seed": s}
    (saida / "mapa_cego.json").write_text(json.dumps(mapa, indent=1), encoding="utf-8")

    jobs = []
    for n, info in ([] if a.sem_nota else mapa.items()):
        jobs.append((resp / f"nota_{n}.json", RUBRICA.format(prompt=PROMPTS[info["prompt"]]), [cego / f"{n}.png"],
                     {"tipo": "nota", "imagem": n}))
    for i in indices:
        for s in SEEDS:
            ordem = [neutro(d, i, s) for d in dits]
            random.Random(f"{SAL}|{i}|{s}").shuffle(ordem)
            jobs.append((resp / f"ranking_p{i}_s{s}.json", RANKING.format(n=len(ordem), prompt=PROMPTS[i]),
                         [cego / f"{n}.png" for n in ordem], {"tipo": "ranking", "prompt": i, "seed": s, "ordem": ordem}))
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: tarefa(*j), jobs))
    print("FIM avaliacao", len(jobs), "tarefas", flush=True)


if __name__ == "__main__":
    main()
