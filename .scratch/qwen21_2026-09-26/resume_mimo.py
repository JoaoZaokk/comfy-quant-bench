"""Resume a avaliacao cega do MiMo: notas medias por build, posicao media no ranking, e ruido do juiz (pares identicos).

    python_embeded\\python.exe -s resume_mimo.py <saida_dir> [<par_controle_a> <par_controle_b>]
"""
import json
import statistics as st
import sys
from pathlib import Path

saida = Path(sys.argv[1])
controle = sys.argv[2:4]
mapa = json.loads((saida / "mapa_cego.json").read_text(encoding="utf-8"))
CRIT = ["prompt_adherence", "text_accuracy", "artifacts", "anatomy", "detail_realism", "overall"]
notas, pos, score_rank, primeiro, defeitos = {}, {}, {}, {}, {}
por_imagem = {}
for f in sorted((saida / "respostas").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    r = d["resposta"]
    if d["tipo"] == "nota":
        info = mapa[d["imagem"]]
        por_imagem[(info["dit"], info["prompt"], info["seed"])] = r
        for c in CRIT:
            if isinstance(r.get(c), (int, float)):
                notas.setdefault(info["dit"], {}).setdefault(c, []).append(r[c])
        defeitos.setdefault(info["dit"], []).extend(r.get("defects", []))
    else:
        ordem = d["ordem"]
        rank = [int(x) for x in r["ranking"]]
        for p, num in enumerate(rank, 1):
            dit = mapa[ordem[num - 1]]["dit"]
            pos.setdefault(dit, []).append(p)
            if p == 1:
                primeiro[dit] = primeiro.get(dit, 0) + 1
        for num, sc in (r.get("scores") or {}).items():
            dit = mapa[ordem[int(num) - 1]]["dit"]
            score_rank.setdefault(dit, []).append(float(sc))
res = {}
for dit in sorted(set(notas) | set(pos)):
    res[dit] = {c: round(st.mean(v), 2) for c, v in notas.get(dit, {}).items()}
    if dit in pos:
        res[dit].update({"posicao_media": round(st.mean(pos[dit]), 2), "primeiro_lugar": primeiro.get(dit, 0),
                         "score_ranking": round(st.mean(score_rank.get(dit, [0])), 2), "n_rankings": len(pos[dit])})
if len(controle) == 2:
    a, b = controle
    difs = [abs(por_imagem[(a, p, s)]["overall"] - por_imagem[(b, p, s)]["overall"])
            for (d, p, s) in por_imagem if d == a and (b, p, s) in por_imagem]
    difs_all = [abs(por_imagem[(a, p, s)][c] - por_imagem[(b, p, s)][c])
                for (d, p, s) in por_imagem if d == a and (b, p, s) in por_imagem
                for c in CRIT if isinstance(por_imagem[(a, p, s)].get(c), (int, float))
                and isinstance(por_imagem[(b, p, s)].get(c), (int, float))]
    res["_ruido_do_juiz"] = {"par": controle, "overall_dif_media": round(st.mean(difs), 2) if difs else None,
                             "overall_dif_max": max(difs) if difs else None,
                             "todos_criterios_dif_media": round(st.mean(difs_all), 2) if difs_all else None, "n": len(difs)}
(saida / "resumo.json").write_text(json.dumps({"resumo": res, "defeitos": defeitos}, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
print(json.dumps(res, indent=1))
