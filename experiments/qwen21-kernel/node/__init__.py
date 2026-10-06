"""Pacote temporario (fora do checkout), fase 7: ConvRot W4A4 REAL com escala por grupo nas 192 lineares do
Qwen-Image-2.1 BF16 (attn.to_q/k/v/to_out.0, img_mlp.gate_up, img_mlp.out), para medir qualidade e tempo do kernel
g64 contra o W4A4 de producao (criterio_fase7.md). Carregado so via --extra-model-paths-config (custom_nodes); sem
G64_POL no ambiente nao faz nada. Nao registra nos.

Envolve comfy.sd.load_diffusion_model_state_dict: depois de criar o modelo, cada linear alvo tem o peso BF16 rotacionado
(Hadamard regular de bloco 256 do ck), quantizado em int4 conforme a politica e guardado como buffers do proprio modulo
(o ComfyUI os move para a GPU e conta na VRAM); o peso BF16 vira um tensor vazio. O forward chama os kernels da fase 7:
quantizador de ativacao (g64_quant, rotacao + escala por grupo, SwiGLU fundido na mlp.out) e GEMM (g64_gemm3).
    G64_POL=3  peso g64 / ativacao g64 (FP)          G64_POL=4  g128 / g128 (FP)
    G64_POL=1  peso g64 / ativacao por linha (FP)    G64_POL=5  peso g64 em razao inteira / ativacao por linha (IMAD)
    G64_POL=2  peso por linha / ativacao g64 (FP)    G64_POL=0  por linha / por linha (o kernel proprio, controle)
    G64_POL=7  peso g128 em razao inteira / ativacao por linha (so no v6); 15, 17 = 5, 7 com a razao em uint8 (v6)
    G64_CFG    configuracao do GEMM (padrao 1: bloco 128 x 128, 4 estagios; auto = afina por forma na primeira chamada,
               mediana de 3 chamadas por configuracao, como o autotune do CUTLASS do ck)
    G64_MOD    modulo do GEMM (padrao g64_gemm3; g64_gemm6 = politicas 0, 4, 5, 7, 15, 17)
    G64_WCLIP  mse: escala do peso por grupo (politicas 1, 3, 4) com a razao de recorte de menor MSE do peso (11 razoes
               de 1,0 a 0,70; erro_mse.py); padrao = absmax/7
    G64_ACLIP  r em (0, 1]: recorte da ativacao por grupo (politicas 2, 3, 4), escala = absmax * r / 7 (g64_quant2)
    G64_SUBST  JSON {"camadas": [...]}: modo misto. O modelo vem do checkpoint do grafo (ex.: p008, INT8 + W4A4) e so
               as camadas da lista viram a politica acima, com o peso lido do BF16 em G64_BF16 (leitura direta, um
               tensor por vez, sem mmap); as outras ficam como o checkpoint as tem (INT8 nativo do ComfyUI).
    G64_TROCA  k (com G64_SUBST): as camadas da lista guardam o g128 e MANTEM o peso do checkpoint; nos k primeiros
               passos de cada amostragem usam o caminho original (ex.: INT8/W4A4 do p008), depois o g128. O passo vem
               do _forward do modelo (zera quando o timestep sobe).
"""
import json
import struct
import importlib.util
import logging
import os

import torch

import comfy.ops
import comfy.sd

_POL = os.environ.get("G64_POL", "").strip()
_CFG_TXT = os.environ.get("G64_CFG", "1").strip()
_CFG = None if _CFG_TXT == "auto" else int(_CFG_TXT)
_CFGS = {"g64_gemm3": (0, 1, 2), "g64_gemm6": (0, 1, 2, 3, 4)}
_AUTO = {}                                                       # (politica, M, N, K) -> configuracao
_MOD = os.environ.get("G64_MOD", "g64_gemm3").strip()
_F7 = os.environ.get("G64_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
_TIPOS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")
_GEMM = _QUANT = None
_SUBST = os.environ.get("G64_SUBST", "").strip()
_WCLIP = os.environ.get("G64_WCLIP", "").strip()
_ACLIP = float(os.environ.get("G64_ACLIP", "1") or 1)
_RAZOES = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70)
_BF16 = os.environ.get("G64_BF16", "//NAS/purple/ComfyBench/diffusion_models/qwen_image_2.1_bf16.safetensors")
_TROCA = int(os.environ.get("G64_TROCA", "0") or 0)
_ESTADO = {"passo": 0, "t_ant": float("-inf")}


def _alta(mod):
    """True = este modulo usa o caminho original do checkpoint neste passo (modo G64_TROCA)."""
    return mod._g64_manter and _ESTADO["passo"] < _TROCA


def _mod(nome):
    pyd = open(os.path.join(_F7, "build_g64", nome, "OK"), encoding="utf-8").read().strip()
    spec = importlib.util.spec_from_file_location(nome, pyd)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _hadamard(dev):
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard
    return _build_hadamard(256, device=dev, dtype=torch.float32)


def _bf16_up(t):
    """menor valor bf16 >= t (t > 0), como o quantizador de ativacao."""
    u = t.float().contiguous().view(torch.int32)
    low = u & 0xFFFF
    u = (u & ~0xFFFF) + (low != 0).to(torch.int32) * 0x10000
    return u.view(torch.float32)


def _pack(q):
    """q int (N, K) em [-7, 7] -> (N, K/2) int8, nibble baixo = coluna par (como o ck)."""
    lo = q[:, 0::2].to(torch.int16) & 0xF
    hi = q[:, 1::2].to(torch.int16) & 0xF
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8).contiguous()


def _quant_weight(w, pol, h):
    """w (N, K) bf16/fp32 na GPU -> dict de buffers (CPU) para a politica."""
    n, k = w.shape
    wr = (w.float().reshape(n, k // 256, 256) @ h).reshape(n, k)
    out = {}
    if pol in (0, 2):                                            # peso por linha
        s = (wr.abs().amax(1) / 7).clamp_min(1e-12)
        q = torch.clamp(torch.round(wr / s[:, None]), -7, 7)
        out["g64_sb"] = s
    elif pol in (1, 3, 4):                                       # peso por grupo, escala bf16 (primeiro fma)
        g = 128 if pol == 4 else 64
        x = wr.reshape(n, k // g, g)
        am = x.abs().amax(-1)
        s = _bf16_up((am / 7).clamp_min(1e-30))
        if _WCLIP not in ("", "mse"):
            raise ValueError(f"G64_WCLIP {_WCLIP}")
        if _WCLIP == "mse":                                      # razao de recorte por grupo de menor MSE do peso
            err = ((torch.clamp(torch.round(x / s[..., None]), -7, 7) * s[..., None] - x) ** 2).sum(-1)
            for r in _RAZOES[1:]:
                sr = _bf16_up((am * r / 7).clamp_min(1e-30))
                er = ((torch.clamp(torch.round(x / sr[..., None]), -7, 7) * sr[..., None] - x) ** 2).sum(-1)
                m = er < err
                s, err = torch.where(m, sr, s), torch.where(m, er, err)
        q = torch.clamp(torch.round(x / s[..., None]), -7, 7).reshape(n, k)
        out["g64_sb"] = s.t().contiguous()                       # (K/G, N)
    elif pol % 10 in (5, 7):                                     # razao inteira de 8 bits sobre a escala da linha
        g = 64 if pol % 10 == 5 else 128
        am = wr.reshape(n, k // g, g).abs().amax(-1)
        sw = (am.amax(1) / 7).clamp_min(1e-12)
        c = torch.ceil(am / 7 / sw[:, None] * 255).clamp(1, 255)
        s = sw[:, None] * c / 255
        q = torch.clamp(torch.round(wr.reshape(n, k // g, g) / s[..., None]), -7, 7).reshape(n, k)
        out["g64_sb"] = (sw / 255).contiguous()
        out["g64_cw"] = c.to(torch.uint8 if pol >= 10 else torch.int32).t().contiguous()       # (K/G, N)
    else:
        raise ValueError(f"G64_POL {pol}")
    out["g64_q"] = _pack(q.to(torch.int8))
    return {k_: v.cpu() for k_, v in out.items()}


def _act_quant(x2, swiglu):
    """x2 (M, K) bf16 (ou (M, 2K) com swiglu) -> (q (M_pad, K/2), sa) no formato da politica."""
    pol = int(_POL)
    if pol in (2, 3, 4):
        g = 128 if pol == 4 else 64
        return _QUANT.quant(x2, g, swiglu, 128, _ACLIP) if _ACLIP != 1.0 else _QUANT.quant(x2, g, swiglu, 128)
    # ativacao por linha (pol 0, 1, 5): os quantizadores ConvRot do ck (os de producao; o SwiGLU fundido na mlp.out,
    # como o W4A4 atual), com as linhas completadas ate 128 (silu(0) * 0 = 0)
    from comfy_kitchen.backends import cuda as ckc
    m = x2.shape[0]
    mp = (m + 127) // 128 * 128
    if mp != m:
        x2 = torch.cat([x2, x2.new_zeros(mp - m, x2.shape[1])])
    x2 = x2.contiguous()
    if swiglu and getattr(ckc, "_swiglu_quant", None) is not None and (x2.shape[1] // 2) % 256 == 0 and x2.shape[1] // 2 <= 16384:
        q, s = ckc._swiglu_quant.swiglu_quantize_int4_rowwise_convrot64(x2)
    else:
        if swiglu:
            x2 = comfy.ops._eager_input_act(x2, "swiglu").contiguous()
        q, s = ckc.quantize_int4_rowwise_convrot64(x2, 256)
    return q, s.reshape(-1).float().contiguous()


def _gemm(q, wq, sa, sb, vazio, cw, bias, pol):
    if _CFG is not None:
        return _GEMM.gemm(q, wq, sa, sb, vazio, cw, bias, pol, _CFG)
    chave = (pol, q.shape[0], wq.shape[0], q.shape[1] * 2)
    cfg = _AUTO.get(chave)
    if cfg is None:
        tempos = {}
        for c in _CFGS.get(_MOD, (1,)):
            try:
                _GEMM.gemm(q, wq, sa, sb, vazio, cw, bias, pol, c)
            except RuntimeError:
                continue
            ts = []
            for _ in range(3):
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); _GEMM.gemm(q, wq, sa, sb, vazio, cw, bias, pol, c); e1.record()
                torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
            tempos[c] = sorted(ts)[1]
        cfg = min(tempos, key=tempos.get)
        _AUTO[chave] = cfg
        logging.info(f"[g64] auto {chave}: cfg {cfg} ({', '.join(f'{c}={t:.3f}' for c, t in sorted(tempos.items()))} ms)")
    return _GEMM.gemm(q, wq, sa, sb, vazio, cw, bias, pol, cfg)


def _run(self, x, swiglu=False):
    sh = x.shape
    x2 = x.reshape(-1, sh[-1])
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    q, sa = _act_quant(x2, swiglu)
    vazio = self.g64_cw.new_empty(0) if hasattr(self, "g64_cw") else torch.empty(0, dtype=torch.int32, device=x2.device)
    cw = self.g64_cw if hasattr(self, "g64_cw") else vazio
    bias = self.bias.to(torch.bfloat16).contiguous() if self.bias is not None else torch.empty(0, dtype=torch.bfloat16, device=x2.device)
    y = _gemm(q, self.g64_q, sa, self.g64_sb, vazio, cw, bias, int(_POL))
    return y[: x2.shape[0]].reshape(*sh[:-1], y.shape[-1]).to(x.dtype)


def _le_bf16(nomes):
    """pesos BF16 das camadas pedidas, um por vez: (nome, tensor bf16 CPU)."""
    with open(_BF16, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        pre = "model.diffusion_model." if f"model.diffusion_model.{nomes[0]}.weight" in h else ""
        for nome in nomes:
            m = h[f"{pre}{nome}.weight"]
            if m["dtype"] != "BF16":
                raise RuntimeError(f"[g64] {nome}: {m['dtype']} no arquivo BF16")
            a, b = m["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            yield nome, torch.frombuffer(buf, dtype=torch.bfloat16).reshape(m["shape"])


def _g64_forward(self, x):
    if _alta(self):
        return type(self).forward(self, x)
    return _run(self, x, False)


if _POL:
    _GEMM = _mod(_MOD)
    _QUANT = _mod("g64_quant2" if _ACLIP != 1.0 else "g64_quant")
    _orig_load = comfy.sd.load_diffusion_model_state_dict
    _orig_lia = comfy.ops.linear_input_act

    def _load(sd, *args, **kwargs):
        model = _orig_load(sd, *args, **kwargs)
        if model is None:
            return model
        dm = model.model.diffusion_model
        dev = torch.device("cuda")
        h = _hadamard(dev)
        pol = int(_POL)
        n = 0

        manter = bool(_TROCA and _SUBST)

        def converte(mod, w):
            for k_, v in _quant_weight(w, pol, h).items():
                mod.register_buffer(k_, v, persistent=False)
            mod._g64 = True
            mod._g64_manter = manter
            if not manter:                                        # sem troca: o peso original nao e mais usado
                mod.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
                mod._full_precision_mm = True
            mod.forward = _g64_forward.__get__(mod)

        if manter:
            orig_fw = dm._forward

            def fw(x, timesteps, *a, **k):
                t = float(timesteps.flatten()[0])
                if t > _ESTADO["t_ant"] + 1e-6:                  # timestep subiu: nova amostragem
                    _ESTADO["passo"] = 0
                elif t < _ESTADO["t_ant"] - 1e-6:                # desceu: proximo passo (com cfg > 1, cond e uncond
                    _ESTADO["passo"] += 1                         # podem chegar em chamadas separadas com o mesmo t)
                    if _ESTADO["passo"] == _TROCA:
                        logging.info(f"[g64] passo {_TROCA} (t={t:.4g}): g128 daqui em diante")
                _ESTADO["t_ant"] = t
                return orig_fw(x, timesteps, *a, **k)

            dm._forward = fw
            logging.info(f"[g64] troca por passo: {_TROCA} primeiros passos no caminho do checkpoint, depois g128")

        if _SUBST:
            lista = json.load(open(_SUBST, encoding="utf-8"))["camadas"]
            mods = dict(dm.named_modules())
            for nome, w in _le_bf16(lista):
                converte(mods[nome], w.to(dev))
                n += 1
                del w
            logging.info(f"[g64] misto {os.path.basename(_SUBST)}: {n} de {len(lista)} camadas trocadas a partir do BF16")
        else:
            for nome, mod in dm.named_modules():
                if not any(nome.endswith(t) for t in _TIPOS) or not nome.startswith("transformer_blocks."):
                    continue
                w = mod.weight.detach().to(dev)
                converte(mod, w)
                n += 1
                del w
        torch.cuda.empty_cache()
        logging.info(f"[g64] politica {pol} cfg {_CFG_TXT}: {n} lineares em int4 real")
        return model

    def _lia(linear, x, input_act, *args, **kwargs):
        if getattr(linear, "_g64", False) and input_act == "swiglu" and not args and not kwargs and not _alta(linear):
            return _run(linear, x, True)
        return _orig_lia(linear, x, input_act, *args, **kwargs)

    comfy.sd.load_diffusion_model_state_dict = _load
    comfy.ops.linear_input_act = _lia
    logging.info(f"[g64] ativo: politica {_POL}, cfg {_CFG_TXT}, modulo {_MOD}"
                 + (f", peso {_WCLIP}" if _WCLIP else "") + (f", recorte da ativacao {_ACLIP}" if _ACLIP != 1.0 else ""))

NODE_CLASS_MAPPINGS = {}
