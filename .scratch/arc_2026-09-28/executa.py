"""Execução na Arc sobre tools/comfy_client.py (copiado ao lado; só stdlib). Prazo por grafo, JSONL por prompt_id."""
import json
import sys
import urllib.parse
import urllib.request

import comfy_client as cc

SERVIDOR = "127.0.0.1:8188"
PRAZO_S = 1800


def bateria(saida, casos):
    """casos: [(extra, grafo)]. Para no primeiro erro (a VRAM da Arc não perdoa repetir o que estourou)."""
    comfy = cc.Comfy(SERVIDOR)
    ctl = cc.controles(comfy)
    regs = []
    for extra, grafo in casos:
        reg = cc.roda_um(comfy, grafo, PRAZO_S, rotulo=json.dumps(extra), extra={**ctl, **extra})
        print(json.dumps({k: reg.get(k) for k in ("grafo", "status", "server_side_s", "erro")})[:600], flush=True)
        cc.grava_jsonl(saida, reg)
        regs.append(reg)
        if reg["status"] != "success":
            break
    sys.exit(cc.codigo_de_saida(regs))


def sobe_tmp_api(wf):
    """Sobe {nome: (grafo_api, nota)} em userdata tmp_api/ para o frontend converter em workflow de UI."""
    for nome, (g, nota) in wf.items():
        body = json.dumps({"api": g, "nota": nota}, ensure_ascii=False).encode()
        url = (f"http://{SERVIDOR}/api/userdata/" + urllib.parse.quote("tmp_api/" + nome + ".json", safe="")
               + "?overwrite=true")
        urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=30)
        print("ok", nome)
