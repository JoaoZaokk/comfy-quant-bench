"""Resume vram_<v>.jsonl: uma linha por evento com livre/alocado/pico na cuda:0 e cuda:1 e os modelos residentes."""
import json
import sys

t0 = None
for linha in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(linha)
    t0 = t0 or r["t"]
    if r["ev"] in ("instalado", "execv_interceptado"):
        continue
    pl = " | ".join(f"c{p['cuda']} livre {p['livre']:>6} aloc {p['alocado']:>6} pico {p['pico_alocado']:>6}"
                    for p in r["placas"])
    mods = ", ".join(f"{m['modelo']}@{m['placa'][-1]} {m['na_placa']}/{m['total']}" for m in r["modelos"] if "modelo" in m)
    if r["ev"] == "free":
        o = f"free {r['placa']} pede {r['pede_mib']}"
    else:
        o = f"{r['ev']} {','.join(r['pede'])}"
        if r["ev"] == "load_antes":
            o += f" mem_req {r.get('memory_required')} min {r.get('minimum_memory_required')}"
    print(f"{r['t'] - t0:7.0f}s  {o}\n          {pl}\n          [{mods}]")
