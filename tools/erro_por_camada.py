"""Erro de SAIDA por camada de varios checkpoints quantizados, sobre ativacoes reais, com os kernels reais.

Para cada Linear da calibracao (`calibrate_activations.py`, linhas reais que a camada recebeu durante uma
geracao do BF16), calcula `Y_ref = F.linear(X, W_bf16)` e, para cada build, `Y_q` pelo mesmo op que o
ComfyUI despacha (`int8_linear`, `w4a8_int8_linear`, `convrot_w4a4_linear` do comfy-kitchen, via as mesmas
tabelas de `verify_w4a4.py`), ou pelo `SVDQW4A4Linear` do Nunchaku para checkpoints
`qwen21-nunchaku-svdq-int4-v1`. Metrica: rel-RMSE = ||Y_q - Y_ref|| / ||Y_ref|| por camada.

Layouts diferentes da mesma camada sao casados: `img_mlp.gate_up` (Comfy-Org, fundido) = cat(gate_layer, proj)
(diffusers), nessa ordem, conferido byte a byte em 26/09. Um build com o layout separado roda as duas
metades e concatena.

Nao mede qualidade final: erro por camada nao soma linearmente nem ve a trajetoria. Serve para comparar
builds entre si na mesma entrada e achar onde cada um perde.

    python_embeded\\python.exe -s tools/erro_por_camada.py --source <bf16> --calib <calib.pt> \\
        --build nome=arquivo.safetensors [--build ...] --out resultado.json
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_w4a4 import FORMATS, load_tensor_cuda, read_header  # noqa: E402

NUNCHAKU = "qwen21-nunchaku-svdq-int4-v1"


def configs_por_camada(header: dict, metadata: dict, path: Path) -> dict:
    """Formato de cada camada: `_quantization_metadata` do header ou o tensor `<camada>.comfy_quant`."""
    out = {}
    if "_quantization_metadata" in metadata:
        out.update(json.loads(metadata["_quantization_metadata"])["layers"])
    for k, info in header.items():
        if k.endswith(".comfy_quant"):
            t = load_tensor_cuda(path, info).cpu()
            out[k.removesuffix(".comfy_quant")] = json.loads(bytes(t.tolist()).decode())
    return out


class Build:
    def __init__(self, nome: str, path: Path):
        self.nome, self.path = nome, path
        self.header, self.metadata = read_header(path)
        man = json.loads(self.metadata.get("manifest", "{}"))
        self.nunchaku = man.get("backend_format") == NUNCHAKU
        self.camadas = man["layers"] if self.nunchaku else configs_por_camada(self.header, self.metadata, path)

    def load(self, nome_tensor, view=None, optional=False):
        info = self.header.get(nome_tensor)
        if info is None:
            if optional:
                return None
            raise KeyError(f"{self.nome}: falta {nome_tensor}")
        return load_tensor_cuda(self.path, info, view=view)

    def _uma(self, camada: str, x: torch.Tensor) -> torch.Tensor:
        if self.nunchaku:
            from nunchaku.models.linear import SVDQW4A4Linear
            i = self.camadas[camada]
            m = SVDQW4A4Linear(i["in_features"], i["out_features"], rank=i["rank"], bias=i["bias"], precision="int4",
                               act_unsigned=False, torch_dtype=torch.bfloat16, device="cuda")
            sd = {k[len(camada) + 1:]: self.load(k) for k in self.header if k.startswith(camada + ".")}
            m.load_state_dict(sd, strict=True)
            with torch.no_grad():
                return m(x.unsqueeze(0)).squeeze(0)
        cfg = self.camadas.get(camada)
        if cfg is None:  # camada mantida em alta precisao neste build
            return torch.nn.functional.linear(x, self.load(camada + ".weight").to(x.dtype))
        fmt = FORMATS[cfg["format"]]
        import comfy_kitchen as ck
        kwargs = fmt.smoke_kwargs(camada, cfg, lambda s, view=None, optional=False: self.load(camada + s, view, optional),
                                  x, {})
        return getattr(ck, fmt.linear_op)(**kwargs)

    def tem(self, camada):
        return camada in self.camadas or f"{camada}.weight" in self.header

    def saida(self, camada: str, x: torch.Tensor) -> torch.Tensor:
        if self.tem(camada):
            return self._uma(camada, x)
        if camada.endswith("img_mlp.gate_up"):
            base = camada.removesuffix("gate_up")
            return torch.cat([self._uma(base + "gate_layer", x), self._uma(base + "proj", x)], dim=-1)
        raise KeyError(f"{self.nome}: sem camada {camada}")

    def formato(self, camada):
        if self.nunchaku:
            return "svdq_int4_r%d" % self.camadas.get(camada, next(iter(self.camadas.values())))["rank"]
        cfg = self.camadas.get(camada) or self.camadas.get(camada.replace("gate_up", "proj"))
        return cfg["format"] if cfg else "bf16"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--build", action="append", required=True, help="nome=arquivo.safetensors")
    ap.add_argument("--rows", type=int, default=0, help="limita linhas por camada (0 = todas)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"recusado: {args.out} existe")
    src_header, _ = read_header(args.source)
    calib = torch.load(args.calib, map_location="cpu", weights_only=False)
    builds = [Build(n, Path(a)) for n, a in (b.split("=", 1) for b in args.build)]
    resultado = {"source": str(args.source), "calib": str(args.calib), "calib_meta": calib["meta"],
                 "builds": {b.nome: str(b.path) for b in builds}, "camadas": {}}
    camadas = sorted(calib["layers"], key=lambda n: (int(n.split(".")[1]), n))
    for idx, camada in enumerate(camadas):
        x = calib["layers"][camada]["sample"]
        if args.rows:
            x = x[:args.rows]
        x = x.to("cuda", torch.bfloat16)
        w = load_tensor_cuda(args.source, src_header[camada + ".weight"])
        ref = torch.nn.functional.linear(x.float(), w.float())
        linha = {"rows": x.shape[0], "K": w.shape[1], "N": w.shape[0]}
        for b in builds:
            y = b.saida(camada, x).float()
            linha[b.nome] = {"rel_rmse": float((y - ref).norm() / ref.norm()), "formato": b.formato(camada),
                             "cos": float(torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0))}
        resultado["camadas"][camada] = linha
        if idx % 24 == 0:
            print(idx, camada, {b.nome: round(linha[b.nome]["rel_rmse"], 4) for b in builds}, flush=True)
        del x, w, ref
        torch.cuda.empty_cache()
    # resumo por build e por tipo de camada
    resumo = {}
    for b in builds:
        por_tipo = {}
        for camada, linha in resultado["camadas"].items():
            tipo = ".".join(camada.split(".")[2:])
            por_tipo.setdefault(tipo, []).append(linha[b.nome]["rel_rmse"])
        todos = [v for vs in por_tipo.values() for v in vs]
        resumo[b.nome] = {"media": sum(todos) / len(todos), "max": max(todos),
                          "por_tipo": {t: {"media": sum(v) / len(v), "max": max(v)} for t, v in sorted(por_tipo.items())}}
    resultado["resumo"] = resumo
    parcial = args.out.with_suffix(args.out.suffix + ".partial")
    parcial.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    parcial.replace(args.out)
    print(json.dumps(resumo, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
