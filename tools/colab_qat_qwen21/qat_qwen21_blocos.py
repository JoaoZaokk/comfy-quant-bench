"""QAT por bloco do Qwen-Image-2.1 para W4A4 ConvRot nativo (reconstrucao bloco a bloco com quantizacao simulada).

POR QUE. Na bancada (26/09) a W4A4 pura e a mais rapida no Ampere (3,26 it/s contra 2,21 do mixed 0,10), mas a
ativacao de 4 bits na MLP produz traco duplo em neon e pele aspera; refinar so a escala nao conserta (o erro e do
arredondamento da ativacao). Aqui os PESOS de cada bloco aprendem a conviver com a ativacao em 4 bits.

COMO. Professor = o proprio BF16. Para cada bloco b, em ordem:
    X_q  = entrada do bloco b vinda dos blocos 0..b-1 JA quantizados (o erro acumulado entra no treino)
    X_fp = entrada do bloco b no BF16 puro;  alvo Y = bloco_b_BF16(X_fp)
    treina as 6 lineares do bloco b, com peso E ativacao em W4A4 simulado, para bloco_b_q(X_q) ~ Y
    avanca X_q e X_fp pelo bloco b
A simulacao reproduz o kernel `convrot_w4a4_linear` do comfy-kitchen: rotacao Hadamard em grupos de 256 no eixo K,
escala por linha = absmax/7, codigos em [-7, 7], tanto no peso quanto na ativacao (por token). STE no arredondamento;
a escala e recalculada a cada passo (nao e parametro). O peso mestre fica em FP32 e os codigos saem dele (a lição do
bug de 26/09: rotacao em BF16 muda codigos).

Um bloco so recebe os pesos treinados se o erro no conjunto de validacao cair; senao fica o RTN (arredondamento puro).

DADOS. Condicionamentos de texto calculados em casa pelo ComfyUI (`gera_conds_qwen21.py`), trajetorias do
professor geradas aqui mesmo: 25 passos euler/simple, shift 0,69, 1024^2 (latente 64x64x64), como o KSampler.

SAIDA. `<out>/qwen_image_2.1_w4a4_qat.safetensors` no formato nativo do ComfyUI (convrot_w4a4, as 192 lineares; o
resto copiado byte a byte da fonte) + `relatorio.json` (erro por bloco antes/depois, erro fim a fim no holdout).
Retomavel: cada bloco treinado grava `<out>/blocos/bloco_XX.pt`; ao retomar, os blocos prontos sao recarregados.

    python qat_qwen21_blocos.py --comfy /content/ComfyUI --dit qwen_image_2.1_bf16.safetensors \\
        --conds conds_qwen21.pt --out /content/qat_qwen21
    python qat_qwen21_blocos.py --smoke --out /tmp/smoke      (modelo minusculo aleatorio, CPU, confere o kernel)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

INT4_MAX = 7
CONVROT = 256
CAMADAS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# W4A4 simulado (identico ao comfy_kitchen.backends.eager.convrot_w4a4)
# ---------------------------------------------------------------------------------------------------------------
_HAD = {}


def hadamard(n, device):
    k = (n, str(device))
    if k not in _HAD:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
        h, s = h4, 4
        while s < n:
            h, s = torch.kron(h, h4), s * 4
        _HAD[k] = h / n ** 0.5
    return _HAD[k]


def gira(t, g):
    """Rotacao por grupos de g no ultimo eixo, na dtype de `t` (como o ck: a ativacao gira em bf16, o peso em fp32).
    H e simetrica e ortonormal: vale para peso (w @ H.T) e ativacao (x @ H)."""
    sh = t.shape
    return (t.reshape(-1, sh[-1] // g, g) @ hadamard(g, t.device).to(t.dtype)).reshape(sh)


class _Arredonda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def quant_linha(x):
    """Codigo inteiro por linha (absmax/7) com STE, na dtype de `x` como `quantize_signed_int4_rowwise`; devolve o
    valor dequantizado em fp32. A escala fica fora do grafo."""
    s = x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-10) / INT4_MAX
    return _Arredonda.apply(x / s).clamp(-INT4_MAX, INT4_MAX).float() * s.float()


class LinearW4A4(torch.nn.Module):
    """Linear com peso mestre FP32 e forward em W4A4 ConvRot simulado."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.weight = torch.nn.Parameter(linear.weight.detach().float().clone())
        self.in_features, self.out_features = self.weight.shape[1], self.weight.shape[0]

    def forward(self, x):
        dt = x.dtype
        xq = quant_linha(gira(x, CONVROT))          # ativacao: gira e quantiza em bf16, como o kernel
        wq = quant_linha(gira(self.weight, CONVROT))  # peso: mestre fp32, como o conversor corrigido
        return (xq @ wq.t()).to(dt)


def codigos(w):
    """Codigos e escalas finais a partir do mestre FP32 -- mesma conta do quantize_convrot_w4a4_weight."""
    wr = gira(w.float(), CONVROT)
    s = wr.abs().amax(-1, keepdim=True).clamp(min=1e-10) / INT4_MAX
    q = (wr / s).round().clamp(-INT4_MAX, INT4_MAX).to(torch.int32)
    lo, hi = q[:, 0::2] & 0x0F, q[:, 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.int8).contiguous(), s.reshape(-1).float().contiguous()


def troca_lineares(bloco):
    for nome in CAMADAS:
        pai, filho = nome.rsplit(".", 1)
        mp = bloco.get_submodule(pai)
        velho = mp.get_submodule(filho)
        mp._modules[filho] = LinearW4A4(velho).to(velho.weight.device)


def lineares(bloco):
    return {n: bloco.get_submodule(n) for n in CAMADAS}


# ---------------------------------------------------------------------------------------------------------------
# Modelo, amostragem e captura das entradas do bloco 0
# ---------------------------------------------------------------------------------------------------------------
def monta_comfy(comfy_dir):
    sys.path.insert(0, str(comfy_dir))
    import comfy.cli_args
    if not torch.cuda.is_available():
        comfy.cli_args.args.cpu = True  # o equivalente do --cpu (smoke e sessao de CPU)
    import comfy.model_management as mm
    mm.in_training = True  # caminho torch puro e diferenciavel no bloco (sem os kernels fundidos)
    return mm


def carrega_dit(path, device):
    import comfy.model_detection
    import comfy.ops
    import comfy.utils
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    sd = comfy.utils.load_torch_file(str(path), device=device)
    cfg = comfy.model_detection.detect_unet_config(sd, "")
    if cfg.get("image_model") != "qwen_image21":
        raise SystemExit(f"{path} nao e Qwen-Image-2.1: {cfg}")
    with torch.device("meta"):
        m = QwenImage21Transformer2DModel(**cfg, dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    m.load_state_dict(sd, strict=True, assign=True)
    return m.to(device).eval().requires_grad_(False), cfg


def modelo_smoke(device):
    import comfy.ops
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    torch.manual_seed(0)
    cfg = dict(in_channels=16, out_channels=16, num_layers=2, attention_head_dim=32, num_attention_heads=8,
               context_in_dim=64, mlp_ratio=3, axes_dims_rope=(4, 14, 14), fused_mlp=True, image_model="qwen_image21")
    m = QwenImage21Transformer2DModel(**cfg, dtype=torch.float32, device=device, operations=comfy.ops.manual_cast)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.to(torch.bfloat16).eval().requires_grad_(False), cfg


def sigmas_simple(passos, shift=0.69, n=10000):
    t = torch.arange(1, n + 1, dtype=torch.float64) / n
    tab = math.exp(shift) / (math.exp(shift) + (1 / t - 1))
    ss = n / passos
    return [float(tab[-(1 + int(x * ss))]) for x in range(passos)] + [0.0]


@torch.no_grad()
def trajetoria(modelo, ctx, lado, canais, seed, passos, device, guardar=()):
    """Euler do KSampler (modelo CONST de fluxo): x_T = sigma_max * ruido; d = saida; x += d * (s_next - s)."""
    sig = sigmas_simple(passos)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn((1, canais, lado, lado), generator=g) * sig[0]).to(device, torch.bfloat16)
    salvos = {}
    for i in range(passos):
        if i in guardar:
            salvos[i] = (x.clone(), sig[i])
        s = torch.full((1,), sig[i], device=device, dtype=torch.float32)
        v = modelo(x, s, ctx)
        x = (x.float() + v.float() * (sig[i + 1] - sig[i])).to(torch.bfloat16)
    return x, salvos


class _Pare(Exception):
    pass


@torch.no_grad()
def entrada_bloco0(modelo, x, sigma, ctx):
    """Roda o forward ate o bloco 0 e devolve os argumentos que ele receberia (hidden, mod, pe, attn_fn, prefix_len)."""
    guardado = {}

    def gancho(mod, args, kwargs):
        guardado["args"] = args
        raise _Pare

    h = modelo.transformer_blocks[0].register_forward_pre_hook(gancho, with_kwargs=True)
    try:
        modelo(x, torch.full((1,), sigma, device=x.device, dtype=torch.float32), ctx)
    except _Pare:
        pass
    finally:
        h.remove()
    hidden, mod, pe, attn_fn, prefix_len = guardado["args"][:5]
    return hidden, (mod, pe, attn_fn, prefix_len)


# ---------------------------------------------------------------------------------------------------------------
# Treino bloco a bloco
# ---------------------------------------------------------------------------------------------------------------
def erro_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def treina_bloco(b, bloco, fp, X_q, X_fp, extras, idx_tr, idx_val, a):
    """Treina as 6 lineares de `bloco` (ja W4A4) para imitar `fp`. Devolve (erro_val_rtn, erro_val_final, aceito)."""
    def fwd(m, x, e):
        mod, pe, attn_fn, prefix_len = e
        return m(x, mod, pe, attn_fn, prefix_len, {})

    @torch.no_grad()
    def val():
        errs = [erro_rel(fwd(bloco, X_q[i], extras[i]), fwd(fp, X_fp[i], extras[i])) for i in idx_val]
        return sum(errs) / len(errs)

    params = [m.weight for m in lineares(bloco).values()]
    inicial = [p.detach().clone() for p in params]
    e0 = val()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    melhor, estado = e0, None
    for passo in range(1, a.passos + 1):
        i = random.choice(idx_tr)
        with torch.no_grad():
            alvo = fwd(fp, X_fp[i], extras[i]).float()
        y = fwd(bloco, X_q[i], extras[i]).float()
        perda = F.mse_loss(y, alvo) / alvo.pow(2).mean().clamp(min=1e-12)
        opt.zero_grad(set_to_none=True)
        perda.backward()
        opt.step()
        if passo % a.val_cada == 0 or passo == a.passos:
            e = val()
            if e < melhor:
                melhor, estado = e, [p.detach().clone() for p in params]
            log(f"bloco {b:2d} passo {passo:4d} perda {float(perda.detach()):.5f} val {e:.5f} (rtn {e0:.5f})")
    aceito = estado is not None
    with torch.no_grad():
        for p, v in zip(params, estado if aceito else inicial):
            p.copy_(v)
    del opt
    return e0, melhor, aceito


@torch.no_grad()
def avanca(bloco, X, extras):
    for i in range(len(X)):
        mod, pe, attn_fn, prefix_len = extras[i]
        X[i] = bloco(X[i], mod, pe, attn_fn, prefix_len, {})


# ---------------------------------------------------------------------------------------------------------------
# Exportacao no formato nativo do ComfyUI
# ---------------------------------------------------------------------------------------------------------------
def exporta(fonte: Path, saida: Path, codigos_por_camada: dict):
    parcial = saida.with_name(saida.name + ".partial")
    for p in (saida, parcial):
        if p.exists():
            raise SystemExit(f"recusado: {p} existe")
    with open(fonte, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cab = json.loads(f.read(n))
    base = 8 + n
    meta = dict(cab.pop("__metadata__", {}) or {})
    entradas, pos = [], 0
    for k, info in cab.items():
        camada = k[:-len(".weight")] if k.endswith(".weight") else None
        if camada in codigos_por_camada:
            q, s = codigos_por_camada[camada]
            for nome, t, dt in ((k, q, "I8"), (camada + ".weight_scale", s, "F32")):
                nb = t.numel() * t.element_size()
                entradas.append((nome, {"dtype": dt, "shape": list(t.shape), "data_offsets": [pos, pos + nb]}, ("t", t)))
                pos += nb
        else:
            a, b = info["data_offsets"]
            entradas.append((k, {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [pos, pos + b - a]},
                             ("c", base + a, b - a)))
            pos += b - a
    meta["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": {
        c: {"format": "convrot_w4a4", "convrot_groupsize": CONVROT} for c in codigos_por_camada}}, separators=(",", ":"))
    meta["quantization"] = "convrot_w4a4"
    meta["qat"] = "tools/colab_qat_qwen21/qat_qwen21_blocos.py (reconstrucao por bloco, W4A4 simulado)"
    cab2 = {k: i for k, i, _ in entradas}
    cab2["__metadata__"] = meta
    h = json.dumps(cab2, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(fonte, "rb") as src, open(parcial, "xb") as out:
        out.write(struct.pack("<Q", len(h)))
        out.write(h)
        for _, _, fonte_dado in entradas:
            if fonte_dado[0] == "t":
                t = fonte_dado[1].cpu().contiguous()
                out.write(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.int8 else t.numpy().tobytes())
            else:
                src.seek(fonte_dado[1])
                falta = fonte_dado[2]
                while falta:
                    buf = src.read(min(64 << 20, falta))
                    if not buf:
                        raise IOError("fim inesperado da fonte")
                    out.write(buf)
                    falta -= len(buf)
        out.flush()
        os.fsync(out.fileno())
    esperado = 8 + len(h) + pos
    if parcial.stat().st_size != esperado:
        raise SystemExit(f"tamanho {parcial.stat().st_size} != {esperado}; partial mantido")
    os.replace(parcial, saida)
    log(f"exportado {saida} ({esperado / 2**30:.2f} GiB, {len(codigos_por_camada)} camadas W4A4)")


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", type=Path, default=Path(__file__).resolve().parents[2] / "ComfyUI")
    ap.add_argument("--dit", type=Path)
    ap.add_argument("--conds", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passos-traj", type=int, default=25)
    ap.add_argument("--amostras-por-traj", type=int, default=4)
    ap.add_argument("--passos", type=int, default=300, help="passos de otimizacao por bloco")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val-cada", type=int, default=50)
    ap.add_argument("--n-val", type=int, default=16, help="amostras reservadas para validacao por bloco")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "blocos").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    monta_comfy(a.comfy)
    t0 = time.time()

    if a.smoke:
        modelo, cfg = modelo_smoke(device)
        lado, canais = 8, cfg["in_channels"]
        conds = {"treino": [torch.randn(1, 7 + i % 3, cfg["context_in_dim"]).to(torch.bfloat16) for i in range(6)],
                 "holdout": [torch.randn(1, 6, cfg["context_in_dim"]).to(torch.bfloat16) for _ in range(2)]}
        a.passos, a.val_cada, a.n_val, a.passos_traj = 20, 10, 2, 6
    else:
        log(f"carregando {a.dit}")
        modelo, cfg = carrega_dit(a.dit, device)
        lado, canais = 64, cfg["in_channels"]
        conds = torch.load(a.conds, map_location="cpu")
    nb = len(modelo.transformer_blocks)
    log(f"modelo: {nb} blocos, cfg {cfg}; {len(conds['treino'])} conds de treino, {len(conds['holdout'])} de holdout")

    # --- trajetorias do professor e entradas do bloco 0 -------------------------------------------------------------
    passos_ok = list(range(1, a.passos_traj - 1))
    X_fp, extras = [], []
    for ci, c in enumerate(conds["treino"]):
        ctx = c.to(device, torch.bfloat16)
        escolhidos = sorted(random.sample(passos_ok, min(a.amostras_por_traj, len(passos_ok))))
        _, salvos = trajetoria(modelo, ctx, lado, canais, 1000 + ci, a.passos_traj, device, guardar=escolhidos)
        for i in escolhidos:
            x, s = salvos[i]
            h, e = entrada_bloco0(modelo, x, s, ctx)
            X_fp.append(h)
            extras.append(e)
    log(f"{len(X_fp)} amostras de treino capturadas em {time.time() - t0:.0f} s")
    X_q = [x.clone() for x in X_fp]
    idx = list(range(len(X_fp)))
    random.shuffle(idx)
    idx_val, idx_tr = idx[:a.n_val], idx[a.n_val:]

    # --- holdout: trajetoria do professor (referencia fim a fim) ---------------------------------------------------
    ref_holdout = []
    for hi, c in enumerate(conds["holdout"]):
        xf, _ = trajetoria(modelo, c.to(device, torch.bfloat16), lado, canais, 5000 + hi, a.passos_traj, device)
        ref_holdout.append(xf)

    def fim_a_fim():
        errs = []
        for hi, c in enumerate(conds["holdout"]):
            xf, _ = trajetoria(modelo, c.to(device, torch.bfloat16), lado, canais, 5000 + hi, a.passos_traj, device)
            errs.append(erro_rel(xf, ref_holdout[hi]))
        return errs

    # --- baseline RTN (todos os blocos em W4A4 sem treino) ---------------------------------------------------------
    originais = {}
    for b in range(nb):
        originais[b] = {n: m.weight.detach().clone() for n, m in lineares(modelo.transformer_blocks[b]).items()}
        troca_lineares(modelo.transformer_blocks[b])
    rtn_fim = fim_a_fim()
    log(f"holdout fim a fim RTN W4A4: erro rel do latente final {[round(e, 4) for e in rtn_fim]}")

    # --- blocos, em ordem ------------------------------------------------------------------------------------------
    import copy
    rel = {"cfg": {k: v for k, v in cfg.items() if k != "axes_dims_rope"}, "args": {k: str(v) for k, v in vars(a).items()},
           "amostras": len(X_fp), "rtn_fim_a_fim": rtn_fim, "blocos": {}}
    for b in range(nb):
        bloco = modelo.transformer_blocks[b]
        fp = copy.deepcopy(bloco)  # professor deste bloco: pesos originais em bf16, sem quantizacao
        for n, m in lineares(fp).items():
            pai, filho = n.rsplit(".", 1)
            lin = torch.nn.Linear(m.in_features, m.out_features, bias=False, device=device, dtype=torch.bfloat16)
            lin.weight.data.copy_(originais[b][n])
            fp.get_submodule(pai)._modules[filho] = lin
        fp.requires_grad_(False)
        salvo = a.out / "blocos" / f"bloco_{b:02d}.pt"
        if salvo.exists():
            d = torch.load(salvo, map_location=device)
            with torch.no_grad():
                for n, m in lineares(bloco).items():
                    m.weight.copy_(d["pesos"][n])
            rel["blocos"][b] = d["rel"]
            log(f"bloco {b:2d} retomado do disco")
        else:
            bloco.requires_grad_(False)
            for m in lineares(bloco).values():
                m.weight.requires_grad_(True)
            e0, e1, ok = treina_bloco(b, bloco, fp, X_q, X_fp, extras, idx_tr, idx_val, a)
            bloco.requires_grad_(False)
            rel["blocos"][b] = {"val_rtn": e0, "val_final": e1, "aceito": ok}
            parcial = salvo.with_suffix(".pt.partial")
            torch.save({"pesos": {n: m.weight.detach().cpu() for n, m in lineares(bloco).items()}, "rel": rel["blocos"][b]}, parcial)
            os.replace(parcial, salvo)
            log(f"bloco {b:2d}: val {e0:.5f} -> {e1:.5f} {'ACEITO' if ok else 'mantido RTN'} ({time.time() - t0:.0f} s)")
        avanca(bloco, X_q, extras)
        avanca(fp, X_fp, extras)
        del fp
        if device.type == "cuda":
            torch.cuda.empty_cache()

    qat_fim = fim_a_fim()
    rel["qat_fim_a_fim"] = qat_fim
    log(f"holdout fim a fim: RTN {sum(rtn_fim) / len(rtn_fim):.4f} -> QAT {sum(qat_fim) / len(qat_fim):.4f}")

    cods = {f"transformer_blocks.{b}.{n}": codigos(m.weight)
            for b in range(nb) for n, m in lineares(modelo.transformer_blocks[b]).items()}
    rel["segundos"] = round(time.time() - t0)
    (a.out / "relatorio.json").write_text(json.dumps(rel, indent=1), encoding="utf-8")

    if a.smoke:
        verifica_kernel(modelo, cods, device)
    else:
        exporta(a.dit, a.out / "qwen_image_2.1_w4a4_qat.safetensors", cods)
    log(f"FIM {time.strftime('%H:%M:%S')}")


def verifica_kernel(modelo, cods, device):
    """Smoke: a Linear simulada tem de dar o mesmo que o kernel eager do comfy-kitchen com os codigos exportados."""
    import comfy_kitchen.backends.eager.convrot_w4a4 as ck
    nome, (q, s) = next(iter(cods.items()))
    m = modelo.get_submodule(nome)
    x = torch.randn(5, m.in_features, dtype=torch.bfloat16, device=device)
    sim = m(x).float()
    ref = ck.convrot_w4a4_linear(x, q.to(device), s.to(device), None, CONVROT, 64).float()
    qk, sk = ck.quantize_convrot_w4a4_weight(m.weight.detach().float(), CONVROT, 64)
    log(f"smoke {nome}: codigos iguais ao quantizador do ck {bool((qk == q).all())}, escalas {float((sk - s).abs().max()):.2e}, "
        f"simulado x kernel rel {erro_rel(sim, ref):.2e}")


if __name__ == "__main__":
    main()
