"""Fecha o desvio do cachorro, e traz o braco `misto` para o teste de edicao pela primeira vez.

O QUE ESTA ABERTO
-----------------
Na verificacao de 2026-09-12 o W4A4 pos DOIS cachorros onde a instrucao pedia um. O controle que
rodou em seguida -- mesma instrucao, dois checkpoints, sementes 2002/3002 -- deu **int8 3 de 3
certos contra W4A4 2 de 3**. Isso NOMEIA o desvio e nao o fecha: um de tres nao se distingue de
sorte, e esta bancada ja errou exatamente assim (o criterio ponderado por sigma "venceu 3/3" e a
vitoria dissolveu em oito sementes).

O DESENHO
---------
Oito sementes, **as mesmas oito nos tres bracos**, pareado. O terceiro braco (`misto`) nunca foi
exercitado no caminho de edicao; ele entra aqui porque compartilha as sementes de graca e porque
uma tabela de versoes que nao diz se o `misto` edita esta incompleta.

    braco   arquivo                              formato
    int8    krea2_turbo_int8_convrot             int8_tensorwise x224   (referencia publica)
    w4a4    krea2_turbo_w4a4                     convrot_w4a4 x224      (nosso)
    misto   krea2_turbo_mixed                    173 w4a4 + 51 w4a8     (nosso)

A instrucao carrega um numeral explicito (`a small brown dog`, singular, "one") porque a pergunta
e se o checkpoint respeita a contagem -- nao se ele adivinha que era uma so.

NAO COBERTO
-----------
Contar cachorro e olho humano, nao metrica: este script gera as imagens e a folha de contato, o
veredito e de quem olha. Uma imagem de origem, uma instrucao, um tamanho, um sampler.
"""
import json
import pathlib
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, r"F:\COMFY_PORTABLE\tools")
from comfy_run_workflow import bases_irmas  # noqa: E402

PAI = "http://127.0.0.1:8190"
API = pathlib.Path(r"F:\COMFY_PORTABLE\.scratch\krea2_edit_api.json")
TEXTO = "Add a small brown dog sitting on the grass next to her."
ARMS = [("int8", "krea2_turbo_int8_convrot.safetensors"),
        ("w4a4", "krea2_turbo_w4a4.safetensors"),
        ("misto", "krea2_turbo_mixed.safetensors")]
SEMENTES = [2002, 3002, 4002, 5002, 6002, 7002, 8002, 9002]


def espera(pid, limite=1500):
    t = time.time(); ultima = 0.0; bases = [PAI]
    while time.time() - t < limite:
        if time.time() - ultima >= 10.0:
            ultima = time.time()
            for b in bases_irmas(PAI):
                if b not in bases:
                    bases.append(b)
        for base in bases:
            try:
                h = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=15))
            except Exception:  # noqa: BLE001
                continue
            if not h:
                continue
            e = h[pid]
            s = [im["filename"] for o in (e.get("outputs") or {}).values()
                 for im in o.get("images", [])]
            return e["status"]["status_str"], s, time.time() - t
        time.sleep(3)
    return "TIMEOUT", [], time.time() - t


base = json.loads(API.read_text(encoding="utf-8"))
base = base.get("prompt", base)
loader = next(k for k, v in base.items() if v["class_type"] == "UNETLoaderMultiGPU")
save = next(k for k, v in base.items() if v["class_type"] == "SaveImage")
encodes = [k for k, v in base.items() if v["class_type"] == "Krea2EditGroundedEncode"]
positivo = next(k for k in encodes if base[k]["inputs"].get("prompt"))
ks = next(k for k, v in base.items() if v["class_type"] == "KSampler")

registro = []
# Ordem: braco por FORA, semente por dentro. A alternativa (semente por fora) deixaria sementes
# completas se a corrida fosse interrompida, o que seria melhor -- mas o `UNETLoaderMultiGPU`
# guarda um modelo so, entao trocar de checkpoint a cada corrida sao **24 cargas** de 7,5 a 12,6
# GiB em vez de 3. Oito vezes o I/O de disco por uma seguranca contra uma interrupcao que nao
# esta prevista. Se esta corrida for cortada no meio, a leitura pareada exige descartar as
# sementes que nao tem os tres bracos -- e isso esta escrito aqui de proposito.
for rotulo, unet in ARMS:
    for semente in SEMENTES:
        p = json.loads(json.dumps(base))
        p[loader]["inputs"]["unet_name"] = unet
        p[save]["inputs"]["filename_prefix"] = f"krea2_caoOrig_{rotulo}_{semente}"
        p[positivo]["inputs"]["prompt"] = TEXTO
        p[ks]["inputs"]["seed"] = semente
        req = urllib.request.Request(
            f"{PAI}/prompt",
            data=json.dumps({"prompt": p, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
        st, saidas, dt = espera(pid)
        print(f"{rotulo:5} semente {semente}  {st:8} {dt:6.1f}s  {saidas}", flush=True)
        registro.append({"braco": rotulo, "semente": semente, "status": st,
                         "arquivos": saidas, "segundos": round(dt, 1)})

pathlib.Path(r"F:\COMFY_PORTABLE\.scratch\edit_8sementes_original.json").write_text(
    json.dumps(registro, indent=2), encoding="utf-8")
falhas = [r for r in registro if r["status"] != "success"]
print(f"\n{len(registro)} corridas, {len(falhas)} nao-success", flush=True)
print("NAO COBERTO: contar cachorro e olho humano. Este script produz as imagens; o veredito "
      "sai da folha de contato.", flush=True)
