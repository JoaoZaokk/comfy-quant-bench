"""Codifica prompts para o LTX 2.x SEM passar o encoder de 24 GB pelo leitor normal do ComfyUI.

Por que existe (medido 2026-09-14, `tools/probe_commit_mmap.py` e vizinhos): `safetensors.safe_open`
cobra 2x o arquivo em commit ao abrir (duas views copy-on-write) e o `torch.empty` do modelo mais
1x. Para o Gemma 3 12B BF16 (24,4 GB) sao ~71 GiB de commit transiente; com ~53 GiB livres nesta
maquina isso forca o pagefile a crescer, e sob expansao o servidor morre com access violation --
foi o que matou a rodada de LoRA do 2.3 as 22:24 de 2026-09-13 (`comfy_8190_g.err`, morte dentro
de `nn.Linear.__init__` ao construir o encoder com o W4A8 ja carregado).

Como: processo nu, ComfyUI importado como biblioteca. `comfy.utils.load_torch_file` e trocado por
um leitor por `numpy.memmap` somente-leitura + `torch.frombuffer` (0 de commit -- os tensores sao
views do cache de arquivo; o `load_state_dict` copia para os parametros do modelo, e o
`torch.empty` desses parametros -- 22,7 GiB -- e o unico commit privado, que cabe sem expansao).
Depois chama exatamente o que o servidor chama: `LTXAVTextEncoderLoader.execute`
(`comfy_extras/nodes_lt_audio.py`) e o corpo de `CLIPTextEncode.encode`
(`clip.encode_from_tokens_scheduled(clip.tokenize(text))`), e grava como o `LTXVSaveConditioning`
do ComfyUI-LTXVideo grava (`conditioning_data_{i}` bf16 contiguo + `attention_mask_{i}`, metadata
`num_conditionings`/`dtype`/`created_at`) em `models/embeddings/<saida>_pos|_neg.safetensors`,
que `LTXVLoadConditioning` le no servidor (`tools/ltx_video.py --cond-from <saida>`).

`--compare PREFIXO` compara o resultado, tensor a tensor, com um par gravado pelo servidor
(`<PREFIXO>_pos|_neg`), e imprime igualdade exata e diferenca maxima -- o autoteste de que este
caminho produz o mesmo condicionamento que o caminho normal.

Placa: `--gpu N` vira CUDA_VISIBLE_DEVICES antes do torch (a 3080 Ti e a 1); o ComfyUI carrega
parcialmente se nao couber. `--device cpu` forca CPU (mais lento, sem tocar em placa nenhuma).

Nao mede: o render; um encoder so (LTX 2.3 + Gemma 3 12B) foi exercitado; a igualdade com o
servidor depende da placa (kernels iguais -> bits iguais; placa diferente pode dar diferenca
pequena, e ela e impressa, nao escondida).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as W
import json
import os
import struct
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIB = float(1 << 30)


class _PI(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", W.DWORD),
                ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD)]


def commit_gib():
    try:
        p = _PI(); p.cb = ctypes.sizeof(p)
        ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
        return p.CommitTotal * p.PageSize / GIB, p.CommitLimit * p.PageSize / GIB
    except Exception:  # noqa: BLE001 -- fora do Windows nao ha o contador; o resto funciona
        return float("nan"), float("nan")


DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64",
          "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn",
          "F8_E5M2": "float8_e5m2"}


def leitor_memmap_ro(torch, np):
    """Substituto de comfy.utils.load_torch_file para .safetensors: memmap somente-leitura + frombuffer."""
    vivos = []  # mantem os memmaps vivos enquanto os tensores existirem

    def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
        if not (ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")):
            raise RuntimeError(f"leitor de baixo commit so cobre safetensors; recebeu {ckpt}")
        with open(ckpt, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
        meta = hdr.pop("__metadata__", None)
        base = 8 + hl
        mm = np.memmap(ckpt, dtype=np.uint8, mode="r")
        vivos.append(mm)
        sd = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            for k, v in hdr.items():
                a, b = v["data_offsets"]
                dt = getattr(torch, DTYPES[v["dtype"]])
                shape = tuple(int(x) for x in v["shape"])
                if b > a:
                    t = torch.frombuffer(mm[base + a:base + b], dtype=dt).reshape(shape)
                else:
                    t = torch.empty(shape, dtype=dt)
                sd[k] = t
        c, lim = commit_gib()
        print(f"  [leitor RO] {os.path.basename(ckpt)}: {len(sd)} tensores, {os.path.getsize(ckpt) / GIB:.2f} GiB, "
              f"commit agora {c:.1f}/{lim:.1f} GiB", flush=True)
        if return_metadata:
            return sd, meta
        return sd
    return load_torch_file


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, help="nome em text_encoders/ (ex.: gemma_3_12B_it_W.safetensors)")
    p.add_argument("--proj-checkpoint", required=True, help="nome em checkpoints/ com text_embedding_projection (ex.: ltx-2.3_text_projection_bf16_W.safetensors)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="blurry, out of focus, low contrast, washed out")
    p.add_argument("--saida", required=True, help="prefixo em models/embeddings/: <saida>_pos e <saida>_neg")
    p.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES (1 = RTX 3080 Ti nesta bancada)")
    p.add_argument("--device", choices=["default", "cpu"], default="default")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                   help="dtype do tensor gravado; float32 e o que o encoder vivo entrega ao sampler")
    p.add_argument("--compare", default=None, help="prefixo de um par gravado pelo servidor para comparar bit a bit")
    p.add_argument("--json", default=None)
    p.add_argument("--stock-locks", action="store_true",
                   help="passa --disable-quantized-text-encoder ao ComfyUI: matematica dequantizada, o caminho do ComfyUI de fabrica (esta arvore tem o patch que destrava por padrao)")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "ComfyUI"))
    sys.argv = ["main.py"] + (["--disable-quantized-text-encoder"] if a.stock_locks else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    import numpy as np
    import torch
    import folder_paths
    from utils.extra_config import load_extra_path_config
    _yaml = ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if _yaml.exists():
        load_extra_path_config(str(_yaml))
    import comfy.utils
    import comfy.sd  # noqa: F401  (importado antes do patch; ele chama comfy.utils.load_torch_file pelo modulo)
    original = comfy.utils.load_torch_file
    comfy.utils.load_torch_file = leitor_memmap_ro(torch, np)
    from comfy_extras.nodes_lt_audio import LTXAVTextEncoderLoader

    c0, lim = commit_gib()
    print(f"commit antes: {c0:.1f}/{lim:.1f} GiB; encoder {a.encoder}; projecao {a.proj_checkpoint}; device {a.device}; CUDA_VISIBLE_DEVICES={a.gpu}", flush=True)
    t0 = time.time()
    out = LTXAVTextEncoderLoader.execute(a.encoder, a.proj_checkpoint, a.device)
    clip = out.args[0] if hasattr(out, "args") else out[0]
    c1, _ = commit_gib()
    print(f"encoder construido em {time.time() - t0:.0f}s; commit {c1:.1f} GiB (delta {c1 - c0:+.1f})", flush=True)

    def codifica(texto):
        tokens = clip.tokenize(texto)
        return clip.encode_from_tokens_scheduled(tokens)

    t0 = time.time()
    pos = codifica(a.prompt)
    neg = codifica(a.negative)
    c2, _ = commit_gib()
    print(f"codificado em {time.time() - t0:.0f}s; commit {c2:.1f} GiB; pos {[tuple(c[0].shape) for c in pos]} neg {[tuple(c[0].shape) for c in neg]}", flush=True)

    # O que o encoder devolve alem do tensor IMPORTA: o LTX 2.3 devolve
    # `extra = {"unprocessed_ltxav_embeds": True}` (comfy/text_encoders/lt.py:201-204) e o modelo so
    # aplica caption_projection + connectors quando essa chave chega (comfy/model_base.py:1185 ->
    # av_model.py:583 preprocess_text_embeds). O LTXVSaveConditioning do ComfyUI-LTXVideo grava so o
    # tensor (+ attention_mask) e o LTXVLoadConditioning devolve o tensor sem a chave: o render sai
    # ruido (medido 2026-09-14: MAE 75,9 contra o encoder vivo, bench/ltx23/cond_identity). Aqui
    # TODAS as opcoes vao junto: tensores como `opt_{i}_{chave}`, o resto como JSON em
    # `__metadata__["options_{i}"]`; o no VoidLoadConditioningFull (custom_nodes/comfy-void-stage-tools)
    # devolve tudo. Tensor em float32 por padrao, que e o dtype que o encoder vivo entrega.
    print("opcoes do encoder:", [sorted(o.keys()) for _, o in pos], flush=True)
    pasta = Path(folder_paths.get_folder_paths("embeddings")[0])
    pasta.mkdir(parents=True, exist_ok=True)
    escritos = {}
    dt_out = torch.float32 if a.dtype == "float32" else torch.bfloat16
    for sufixo, cond in (("_pos", pos), ("_neg", neg)):
        tensors = {}
        meta = {"num_conditionings": str(len(cond)), "dtype": a.dtype, "created_at": str(datetime.now()),
                "prompt": a.prompt if sufixo == "_pos" else a.negative, "encoder": a.encoder, "proj_checkpoint": a.proj_checkpoint,
                "tool": "tools/ltx_encode_lowcommit.py"}
        for idx, (ct, opts) in enumerate(cond):
            tensors[f"conditioning_data_{idx}"] = ct.to(dtype=dt_out).contiguous().cpu()
            escalares = {}
            for k, v in opts.items():
                if isinstance(v, torch.Tensor):
                    tensors[f"opt_{idx}_{k}"] = v.contiguous().cpu()
                    if k == "attention_mask":
                        tensors[f"attention_mask_{idx}"] = v.contiguous().cpu()
                else:
                    try:
                        json.dumps(v)
                        escalares[k] = v
                    except TypeError:
                        escalares[k] = repr(v)
                        print(f"  AVISO: opcao {k} nao serializavel ({type(v).__name__}); gravada como repr", flush=True)
            meta[f"options_{idx}"] = json.dumps(escalares)
        destino = pasta / f"{a.saida}{sufixo}.safetensors"
        if destino.exists():
            raise SystemExit(f"RECUSA: {destino} ja existe; nao sobrescrevo condicionamento")
        comfy.utils.save_torch_file(tensors, str(destino), metadata=meta)
        escritos[sufixo] = {"arquivo": str(destino), "bytes": destino.stat().st_size, "tensores": {k: list(v.shape) for k, v in tensors.items()},
                            "options": {k: meta[k] for k in meta if k.startswith("options_")}}
        print(f"gravado {destino} ({destino.stat().st_size / 2**20:.2f} MiB): {list(tensors)} options {[meta[k] for k in meta if k.startswith('options_')]}", flush=True)

    comparacao = None
    if a.compare:
        import safetensors
        comparacao = {}
        for sufixo, cond in (("_pos", pos), ("_neg", neg)):
            ref = pasta / f"{a.compare}{sufixo}.safetensors"
            with safetensors.safe_open(str(ref), framework="pt", device="cpu") as f:
                for idx, (ct, opts) in enumerate(cond):
                    r = f.get_tensor(f"conditioning_data_{idx}")
                    mine = ct.to(torch.bfloat16).contiguous().cpu()
                    igual_forma = tuple(r.shape) == tuple(mine.shape)
                    dmax = float((r.float() - mine.float()).abs().max()) if igual_forma else float("nan")
                    exato = bool(igual_forma and torch.equal(r, mine))
                    mask_ok = None
                    if f"attention_mask_{idx}" in f.keys() and "attention_mask" in opts:
                        mask_ok = bool(torch.equal(f.get_tensor(f"attention_mask_{idx}"), opts["attention_mask"].cpu()))
                    comparacao[f"{sufixo}{idx}"] = {"forma_igual": igual_forma, "exato": exato, "dif_max": dmax, "mascara_igual": mask_ok,
                                                    "ref_forma": list(r.shape), "minha_forma": list(mine.shape)}
                    print(f"  compare {a.compare}{sufixo}[{idx}]: forma {'igual' if igual_forma else 'DIFERE'} {list(r.shape)} vs {list(mine.shape)}; "
                          f"bits {'IGUAIS' if exato else 'diferem'}; dif_max {dmax:.3g}; mascara {mask_ok}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"encoder": a.encoder, "proj_checkpoint": a.proj_checkpoint, "prompt": a.prompt, "negative": a.negative,
                       "saida": a.saida, "device": a.device, "gpu": a.gpu, "commit_antes": c0, "commit_depois_encoder": c1,
                       "commit_limite": lim, "escritos": escritos, "compare": comparacao, "stock_locks": bool(a.stock_locks)}, fh, indent=1)
        print(f"json em {a.json}")
    comfy.utils.load_torch_file = original
    print("NAO COBERTO: o render; so este encoder e esta projecao; igualdade com o servidor depende da placa e e "
          "impressa, nao presumida; nenhum outro formato de arquivo alem de safetensors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
