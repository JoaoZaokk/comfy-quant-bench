r"""A foto. 2:4 esparso + int4 num render de verdade do Z-Image, contra BF16 e contra o W4A4.

POR QUE ESTA FERRAMENTA E A UNICA QUE DECIDE
---------------------------------------------
Tudo que esta bancada mediu sobre esparsidade em 2026-09-01 e GEMM: erro por camada e tempo por
GEMM. O proprio `CLAUDE.md` diz, com todas as letras, que **um checkpoint que converte limpo,
resolve o backend CUDA e escreve um sidecar valido ainda pode produzir lixo -- o preflight prova
despacho, nunca qualidade; so um render prova.** O HunyuanVideo 1.5 despacha nativo nesta placa e
sai destruido. Entao ate existir imagem, o 0,1391 medido nao autoriza conclusao nenhuma.

O QUE CADA BRACO E, E O QUE ELE **NAO** E
------------------------------------------
    bf16              o checkpoint intacto. A referencia.
    w4a4              `zimage-v2-w4a4`, o que esta bancada entrega hoje. 4,0 bits/peso.
    esp_peso          2:4 por PAR (Wanda) + int4 no PESO. 2,5 bits/peso.
    esp_peso_ativ     o mesmo, mais int4 na ATIVACAO. E o que o kernel de fato computa.

**Os dois ultimos sao SIMULACAO, nao o kernel.** Os numeros do peso sao identicos aos que
`tools/sparse24_sm86/` executa -- mesma poda por par, mesma quantizacao simetrica -- mas o GEMM aqui
e bf16 sobre valores reconstruidos. Isso separa a pergunta "o formato estraga a imagem?" da pergunta
"o kernel esta certo?", que ja foi respondida a parte e com controle bit-exato.

`esp_peso` sozinho mediria so metade: o kernel quantiza a ativacao tambem, e ignorar isso daria uma
imagem melhor do que a que o formato entrega. Por isso os dois estao aqui e o de cima existe apenas
para separar quanto veio do peso e quanto veio da ativacao.

O CONTROLE QUE FAZ A CIRURGIA VALER
------------------------------------
Depois de mexer nos pesos, e depois de amostrar, o braco **reconta o padrao 2:4 por par nas camadas
que diz ter alterado** e imprime a fracao. Uma cirurgia que nao pegou -- porque o ModelPatcher
recarregou, porque o nome nao bateu, porque a camada estava em meta -- produziria uma imagem
identica a BF16 e eu leria isso como "o formato nao estraga nada". O contador e o que impede.

NAO COBERTO
-----------
Uma semente e um prompt sao uma amostra, nao uma avaliacao. Sem metrica perceptual (LPIPS/FID) --
so diferenca no pixel e a folha lado a lado para olho humano. So Z-Image, so sm86. A ativacao e
quantizada **por token**, que e mais generoso do que o intervalo unico usado no benchmark de tempo.
O int4 e simetrico por linha e **nao e ConvRot** -- e a rotacao e justamente o que leva o W4A4 ao
erro dele, entao o braco esparso corre sem a defesa que o braco W4A4 tem.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "bench" / "esparso_visual"

BRACO = r'''
import json, sys, time
sys.path.insert(0, "ComfyUI")
sys.argv = ["main.py", "--disable-dynamic-vram"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, torch.nn as nn, folder_paths, comfy.sd, comfy.sample

UNET=%(UNET)r; CLIP=%(CLIP)r; PROMPT=%(PROMPT)r; NEG=%(NEG)r
SEED=%(SEED)d; STEPS=%(STEPS)d; CFG=%(CFG)s; SIDE=%(SIDE)d
OUTPT=%(OUTPT)r; MODO=%(MODO)r; CALIB=%(CALIB)r

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))

def q4(t, dim):
    """int4 simetrico, escala por `dim`. Devolve o valor RECONSTRUIDO."""
    s = t.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 7
    return (t / s).round().clamp(-8, 7) * s

CG = 256   # convrot_groupsize; tem de ser potencia de 4, e 256 e o que esta bancada usa

def rotacao(k, dev, dt):
    """Hadamard regular normalizada, bloco de CG. Simetrica e ortogonal, entao H@H = I."""
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    return _build_hadamard(CG, dev, torch.float32).to(dt)

def gira_peso(w, h):
    """W_rot = W @ H^T, por grupo de CG colunas."""
    o, i = w.shape
    return torch.matmul(w.reshape(o, i // CG, CG), h.T.to(w.dtype)).reshape(o, i)

def gira_ativacao(x, h):
    """x_rot = x @ H, por grupo de CG. O par de `gira_peso`: x_rot @ W_rot^T == x @ W^T."""
    f = x.shape[-1]
    return torch.matmul(x.reshape(-1, f // CG, CG), h.to(x.dtype)).reshape(x.shape)

def poda_elem(w, norma):
    """2 dos 4 VALORES de cada 4 colunas. E o que o kernel INT8 esparso aceita; o INT4 nao."""
    n, k = w.shape
    p = w.abs() if norma is None else w.abs() * norma.to(w.device, w.dtype).unsqueeze(0)
    esc = p.reshape(n, k // 4, 4)
    idx = esc.argsort(dim=-1)[..., :2]
    m = torch.ones_like(esc, dtype=torch.bool).scatter_(-1, idx, False)
    return (w.reshape(n, k // 4, 4) * m).reshape(n, k)

def poda_par(w, norma):
    """2 dos 4 PARES de cada 8 colunas. Criterio Wanda quando ha norma da ativacao real."""
    n, k = w.shape
    p = w.abs() if norma is None else w.abs() * norma.to(w.device, w.dtype).unsqueeze(0)
    esc = p.reshape(n, k // 8, 4, 2).sum(-1)
    idx = esc.argsort(dim=-1)[..., :2]
    m = torch.ones_like(esc, dtype=torch.bool).scatter_(-1, idx, False)
    return (w.reshape(n, k // 8, 4, 2) * m.unsqueeze(-1)).reshape(n, k)

def fracao_par24(w):
    """O contador do controle, e a primeira versao dele estava ERRADA.

    Ela contava grupos com EXATAMENTE 2 dos 4 pares vivos e deu 0,9422, o que eu li como "a
    cirurgia nao sobreviveu". Nao foi isso: apos quantizar para int4, um par MANTIDO cujos dois
    valores arredondam para zero vira indistinguivel de um par podado. `== 2` conta esses como
    falha. O invariante que a quantizacao NAO pode quebrar e `<= 2`: quantizar so pode zerar mais,
    nunca ressuscitar um par podado. Entao `no_maximo_2` e o controle e `exatamente_2` e
    informacao -- a distancia entre os dois mede quanto o int4 zerou por conta propria.

    E o modo de falha aqui e o mesmo que `poda24_pares` documenta e evita: reduzir a mascara de
    `w != 0`. Escrevi a advertencia la e cometi o erro aqui.
    """
    n, k = w.shape
    vivo = (w.reshape(n, k // 8, 4, 2) != 0).any(-1).sum(-1)
    return {"no_maximo_2": (vivo <= 2).float().mean().item(),
            "exatamente_2": (vivo == 2).float().mean().item()}


def fracao_elem24(w):
    """O invariante do braco por ELEMENTO: no maximo 2 valores vivos em cada grupo de 4.

    Cobrar aqui o invariante de PAR (`fracao_par24`) reprova a cirurgia CORRETA: podar 2 de cada 4
    valores pode deixar os dois sobreviventes em pares diferentes, e ai os 4 pares do grupo de 8
    estao vivos, legitimamente. Foi assim que `so_poda_elem` saiu com "a cirurgia nao sobreviveu"
    em duas execucoes seguidas de um patch que estava correto o tempo todo.
    """
    n, k = w.shape
    vivo = (w.reshape(n, k // 4, 4) != 0).sum(-1)
    return {"no_maximo_2": (vivo <= 2).float().mean().item(),
            "exatamente_2": (vivo == 2).float().mean().item()}

def recupera_cg(w, m, h, iteracoes=256, tol=1e-6):
    """Gradiente conjugado mascarado -- o mesmo de tools/recupera_esparso.py.

    Acha os pesos sobreviventes que minimizam ||X Ws^T - X W^T|| com Ws = M (*) Ws. O operador
    `V -> M (*) (V H)` e simetrico definido positivo dentro do suporte, entao CG vale e trata
    todas as linhas de saida em paralelo: cada iteracao e UM matmul [N x K] @ [K x K].
    """
    w, h, m = w.float(), h.float(), m.float()
    ws = w * m
    r = m * ((w - ws) @ h)
    pd = r.clone()
    rr = (r * r).sum()
    rr0 = rr.clone()
    for _ in range(iteracoes):
        ap = m * (pd @ h)
        pap = (pd * ap).sum()
        if pap <= 0:
            break
        al = rr / pap
        ws = ws + al * pd
        r = r - al * ap
        rn = (r * r).sum()
        if rn <= tol * tol * rr0:
            break
        pd = r + (rn / rr) * pd
        rr = rn
    return ws * m

def q8(t):
    """int8 simetrico por linha, RECONSTRUIDO. E o que o kernel 2:4 int8 de fato consome."""
    e = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return (t / e).round().clamp(-127, 127) * e

tocadas, com_wanda = [], 0
if MODO != "nenhum":
    normas, amostras = {}, {}
    RECUPERA = MODO.startswith("recup")
    calib_bruta = None
    if CALIB:
        calib_bruta = torch.load(CALIB, map_location="cpu", weights_only=False)
        for k, v in calib_bruta["layers"].items():
            x = v["sample"] if isinstance(v, dict) else v
            if torch.is_tensor(x):
                # A norma e pequena e fica; a AMOSTRA e enorme e so e guardada quando este braco
                # vai reconstruir. Com a calibragem de 8192 linhas sao 12,9 GiB, e guardar tudo
                # fez o ComfyUI despejar o modelo por pressao de memoria no meio da amostragem.
                normas[k] = x.float().reshape(-1, x.shape[-1]).norm(dim=0)
                if RECUPERA:
                    amostras[k] = x
        if not RECUPERA:
            del calib_bruta
            calib_bruta = None
            import gc; gc.collect()
    GIRA = MODO.startswith("convrot")
    QUANT_ATIV = MODO.endswith("_ativ")
    puladas_por_cg = 0
    puladas_sem_amostra = 0
    recuperadas = 0
    for nome, mod in model.model.diffusion_model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight is None: continue
        w = mod.weight
        if w.dim() != 2 or w.shape[1] %% 8 or w.is_meta: continue
        if GIRA and w.shape[1] %% CG:
            # Sem rotacao possivel nesta camada. PULAR e mais honesto que rodar metade dela sem
            # rotacao: misturaria dois formatos no mesmo braco e a folha nao diria qual.
            puladas_por_cg += 1
            continue
        nrm = normas.get(nome)
        com_wanda += nrm is not None
        with torch.no_grad():
            wf = w.data.float()
            if GIRA:
                h = rotacao(w.shape[1], w.device, torch.float32)
                wf = gira_peso(wf, h)
                # O criterio Wanda tem de ver a norma da ativacao NO MESMO ESPACO em que o peso
                # esta sendo podado. Usar a norma nao rotacionada pontuaria colunas que nao
                # existem mais depois da rotacao.
                if nrm is not None:
                    nr = nrm.to(w.device, torch.float32)
                    nrm = gira_ativacao(nr.reshape(1, -1), h).abs().reshape(-1)
            podado = poda_elem(wf, nrm) if "elem" in MODO else poda_par(wf, nrm)
            if MODO.startswith("recup"):
                # Reconstrucao por camada. Sem amostra nao ha Hessiana, e ai a camada fica
                # PODADA E NAO RECUPERADA -- que e um formato diferente do que o braco diz medir.
                # Pular e mais honesto: a contagem de recuperadas viaja no relatorio.
                xs = amostras.pop(nome, None)
                if xs is None:
                    puladas_sem_amostra += 1
                    continue
                xf_ = xs.to(w.device).float().reshape(-1, xs.shape[-1])
                if GIRA:
                    xf_ = gira_ativacao(xf_, h)
                hh = xf_.double().T @ xf_.double()
                hh += torch.eye(hh.shape[0], device=hh.device, dtype=hh.dtype) * (
                    0.01 * hh.diag().mean())
                mask = (podado != 0).float()
                rec = recupera_cg(wf, mask, hh.float())
                novo = (q8(rec) if MODO.endswith("_int8") else rec).to(w.dtype)
                del xf_, hh, mask, rec
                recuperadas += 1
            else:
                # `so_poda`: bf16 nos valores mantidos, 9,0 bits/peso. Isola a PODA do int4 --
                # sem este braco, "destruido" nao distingue qual dos dois estragou.
                novo = (podado if MODO.startswith("so_poda") else q4(podado, 1)).to(w.dtype)
            w.data.copy_(novo)
        tocadas.append(nome)

    # Tudo da calibragem morre ANTES de amostrar. O que restou em `amostras` sao camadas que o
    # loop nao tocou (largura incompativel, por exemplo) e elas nao servem para mais nada aqui.
    amostras.clear()
    if calib_bruta is not None:
        del calib_bruta
    normas.clear()
    import gc; gc.collect()
    torch.cuda.empty_cache()

    if GIRA or QUANT_ATIV:
        def gancho(m, args):
            x = args[0]
            if GIRA:
                x = gira_ativacao(x.float(), rotacao(x.shape[-1], x.device, torch.float32))
            if QUANT_ATIV:
                x = q4(x.float(), -1)
            return (x.to(args[0].dtype),) + tuple(args[1:])
        alvo = dict(model.model.diffusion_model.named_modules())
        for nome in tocadas:
            alvo[nome].register_forward_pre_hook(gancho)

clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.LUMINA2)
positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0])]
negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(NEG))[0])]
del clip
import comfy.model_management as mm; mm.soft_empty_cache()

lf = model.model.latent_format
side = SIDE // 8
latent = torch.zeros([1, lf.latent_channels, side, side], device="cpu")
noise = comfy.sample.prepare_noise(latent, SEED, None)
t0 = time.perf_counter()
out = comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple",
                          positive, negative, latent, denoise=1.0, disable_noise=False,
                          start_step=None, last_step=None, force_full_denoise=False,
                          noise_mask=None, callback=None, disable_pbar=True, seed=SEED)
el = time.perf_counter() - t0
out = out.cpu().float()
torch.save(out, OUTPT)

# O CONTROLE: reconferido DEPOIS de amostrar, porque o que interessa e se o peso continuava
# alterado enquanto o modelo rodava -- nao se a copia funcionou um segundo antes.
alvo = dict(model.model.diffusion_model.named_modules())
# O invariante segue a granularidade do braco, nao o nome do arquivo.
conta24 = fracao_elem24 if ("elem" in MODO) else fracao_par24
fr = ([conta24(alvo[n].weight.data.float()) for n in tocadas[:16]]
      if tocadas and "elem" not in MODO else [])
print("RESULT " + json.dumps({
    "modo": MODO, "camadas_tocadas": len(tocadas), "com_wanda": com_wanda,
    "puladas_por_groupsize": puladas_por_cg if MODO != "nenhum" else 0,
    "controle_no_maximo_2": (sum(f["no_maximo_2"] for f in fr)/len(fr)) if fr else None,
    "controle_granularidade": ("elemento: <=2 vivos por grupo de 4" if "elem" in MODO
                               else "par: <=2 pares vivos por grupo de 8"),
    "exatamente_2": (sum(f["exatamente_2"] for f in fr)/len(fr)) if fr else None,
    "seconds": el, "mean": float(out.mean()), "std": float(out.std()),
    "absmax": float(out.abs().max()),
}))
'''


def roda(a, nome, unet, modo, calib, semente) -> dict | None:
    saida = SAIDA / f"latent_{nome}_s{semente}.pt"
    src = BRACO % {"UNET": unet, "CLIP": a.clip, "PROMPT": a.prompt, "NEG": a.negative,
                   "SEED": semente, "STEPS": a.steps, "CFG": repr(a.cfg), "SIDE": a.size,
                   "OUTPT": str(saida), "MODO": modo, "CALIB": calib}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    r = subprocess.run([str(RAIZ / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(RAIZ), timeout=5400)
    for linha in r.stdout.splitlines():
        if linha.startswith("RESULT "):
            return json.loads(linha[7:])
    print(f"  BRACO {nome} FALHOU\n  stdout: {r.stdout.strip()[-900:]}")
    print(f"  stderr: {r.stderr.strip()[-2500:]}")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet-bf16", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--unet-w4a4", default="zimage-v2-w4a4.safetensors")
    p.add_argument("--clip", default="qwen_3_4b.safetensors")
    p.add_argument("--calib", default=str(RAIZ / "calib" / "zimage_v2_sigma.calib.pt"))
    p.add_argument("--prompt", default="a red apple on a weathered wooden table, soft window "
                                       "light, shallow depth of field, photographic")
    p.add_argument("--negative", default="")
    # Tres sementes e nao uma: uma imagem so nao distingue "o formato estraga" de "esta
    # semente saiu ruim", e esta bancada ja publicou um card cuja referencia FP16 estava quebrada
    # numa semente e boa na outra -- exatamente o que uma amostra unica esconde.
    p.add_argument("--modos", nargs="+", default=None,
                   help="roda so estes bracos pelo nome; sem isto roda todos")
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 7, 20260901])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    bracos = [("bf16", a.unet_bf16, "nenhum", ""),
              ("w4a4", a.unet_w4a4, "nenhum", ""),
              ("esp_peso", a.unet_bf16, "peso", a.calib),
              ("esp_peso_ativ", a.unet_bf16, "peso_ativ", a.calib),
              # A UNICA diferenca destes dois para os dois de cima e a rotacao ConvRot antes da
              # poda. Um eixo, e e o eixo que faz o W4A4 chegar a 0,0956.
              ("convrot_peso", a.unet_bf16, "convrot_peso", a.calib),
              ("convrot_peso_ativ", a.unet_bf16, "convrot_peso_ativ", a.calib),
              # Poda pura, sem quantizacao nenhuma. Se estes sairem bem, o culpado e o int4;
              # se sairem ruido, e a poda -- e nenhum kernel esparso salva este modelo.
              ("so_poda_elem", a.unet_bf16, "so_poda_elem", a.calib),
              ("recup_elem", a.unet_bf16, "recup_elem", a.calib),
              ("recup_elem_int8", a.unet_bf16, "recup_elem_int8", a.calib),
              ("so_poda_par", a.unet_bf16, "so_poda_par", a.calib)]

    if a.modos:
        bracos = [b for b in bracos if b[0] in set(a.modos)]
        print(f"filtrado para {len(bracos)} braco(s): {[b[0] for b in bracos]}")
    res = {}
    for semente in a.seeds:
        print(f"\n######## semente {semente} ########", flush=True)
        for nome, unet, modo, calib in bracos:
            print(f"--- {nome} ({unet}, modo={modo}) ---", flush=True)
            r = roda(a, nome, unet, modo, calib, semente)
            if r is None:
                continue
            res[f"{nome}_s{semente}"] = r
            fr = r["controle_no_maximo_2"]
            print(f"  {r['seconds']:.1f}s   {r['camadas_tocadas']} camadas tocadas "
                  f"({r['com_wanda']} com Wanda)   |latente| std {r['std']:.4f}"
                  + (f"   controle <=2 pares: {fr:.4f}  (exatamente 2: {r['exatamente_2']:.4f}, "
                     f"a diferenca e o int4 zerando por conta propria)" if fr is not None else ""))
            if r["camadas_tocadas"] and (fr is None or fr < 0.999):
                print(f"  invariante cobrado: {r.get('controle_granularidade', '?')}")
                print("  CONTROLE FALHOU: a cirurgia nao sobreviveu ate o fim da amostragem. "
                      "Qualquer imagem deste braco descreve outro modelo.", file=sys.stderr)

    (SAIDA / "resumo.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nlatentes em {SAIDA}. Decodifique com tools/decode_esparso_visual.py")
    print("\nNAO COBERTO: uma semente, um prompt. Sem metrica perceptual. O int4 e simetrico por")
    print("  linha e NAO e ConvRot, entao o braco esparso corre sem a defesa que o W4A4 tem. Os")
    print("  bracos esparsos sao SIMULACAO dos numeros do formato, nao o kernel -- o kernel foi")
    print("  verificado a parte, com controle bit-exato, em tools/sparse24_sm86/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
