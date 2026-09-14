"""Escreve um GGUF BF16 (sem perda) a partir de um safetensors BF16, lendo por mmap somente-leitura.

Por que existe -- medido em 2026-09-14, nesta maquina (Windows 11, torch 2.13.0+cu130, safetensors
0.8.0), com `scratchpad/probe_commit_mmap.py`, `probe_safeopen_trace.py` e `probe_double_map.py`:

- `safetensors.safe_open(framework="pt")` mapeia o arquivo como copy-on-write DUAS vezes (a view do
  memmap2 para o cabecalho e `torch.UntypedStorage.from_file(shared=False)` para os dados), e o
  Windows cobra commit pelo tamanho INTEIRO de cada view no ato do mapeamento: um arquivo de
  39,13 GiB custa **+80,2 GiB de commit** antes de qualquer tensor ser lido. `torch.empty` do modelo
  custa mais 39. Um mmap somente-leitura (o que numpy/gguf usam) custa **0**.
- O teto de commit desta maquina e 124,8 GiB com ~70 GiB ja comprometidos por outros processos.
  Quando a cobranca precisa EXPANDIR o pagefile, a view (ou o heap do `torch.empty`) volta sem
  lastro e o primeiro toque da `Windows fatal exception: access violation` -- reproduzido num
  processo nu em C: e em W:, e e a assinatura das sete mortes do servidor no braco BF16 do LTX 2.3.
- O loader do ComfyUI-GGUF le por `numpy.memmap` (`loader.py:125`, `torch.from_numpy`) e o
  `GGMLOps.Linear` nao faz `torch.empty` (`ops.py:232-239`) nem copia no `load_state_dict`
  (`ops.py:120-133`): commit zero para os pesos. Um GGUF BF16 e o mesmo BF16, bit a bit, num
  container que esta maquina consegue abrir.

Regras de tipo copiadas de `ComfyUI-GGUF/tools/convert.py::handle_tensors`: 1-D -> F32; ate 1024
elementos -> F32; nome contendo `scale_shift_table` -> F32 (nn.Parameter, nao passa pelo
GGMLLayer); o resto fica BF16. BF16 -> F32 e exato (`int16 << 16`). Fonte F32 fica F32; fonte F16
ficaria F16. Prefixo `model.diffusion_model.` removido, como o convert.py faz.

`--template` (um GGUF do MESMO modelo, ex.: o Q6_K de terceiro) confere: mesmo conjunto de nomes,
mesmas formas, politica de tipos (F32 no template <-> F32 aqui; quantizado ou BF16 no template <->
BF16 aqui), e compara os BYTES de ate 32 tensores F32 entre os dois -- se o terceiro partiu do mesmo
BF16, os F32 tem de ser identicos.

Escreve em `<saida>.partial` e renomeia no fim; recusa saida existente e `.partial` velho; confere o
espaco livre antes. Nao mede: se o ComfyUI carrega e amostra o resultado (isso e o render);
so `ltxv` foi exercitado; big-endian nao e tratado.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

PREFIXO = "model.diffusion_model."
LIMIAR_QUANT = 1024          # convert.py QUANTIZATION_THRESHOLD
MAX_NOME = 127               # convert.py MAX_TENSOR_NAME_LENGTH
MAX_DIMS = 4                 # convert.py MAX_TENSOR_DIMS
GIB = float(1 << 30)


def le_cabecalho(caminho: str):
    with open(caminho, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    meta = hdr.pop("__metadata__", {}) or {}
    return 8 + hl, hdr, meta


def bf16_para_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arch", default="ltxv")
    p.add_argument("--prefix", default=PREFIXO)
    p.add_argument("--template", default=None, help="GGUF do mesmo modelo para conferencia de nomes/formas/tipos")
    p.add_argument("--hiprec", default="scale_shift_table", help="substrings (virgula) que ficam em F32")
    a = p.parse_args()

    import gguf  # noqa: E402  (pacote do ComfyUI-GGUF, ja instalado no embutido)

    if os.path.exists(a.output):
        print(f"RECUSA: saida ja existe: {a.output}")
        return 2
    parcial = a.output + ".partial"
    if os.path.exists(parcial):
        print(f"RECUSA: .partial velho em {parcial}; apague a mao se a corrida anterior morreu")
        return 2

    base, hdr, meta = le_cabecalho(a.input)
    tam = os.path.getsize(a.input)
    hiprec = [s for s in a.hiprec.split(",") if s]
    print(f"fonte    : {a.input}  {tam / GIB:.2f} GiB  {len(hdr)} tensores  metadata {sorted(meta)}")

    # ---- plano de tipos (sem ler dado nenhum) ----
    plano = []
    total_out = 0
    for k, v in hdr.items():
        nome = k[len(a.prefix):] if a.prefix and k.startswith(a.prefix) else k
        if len(nome) > MAX_NOME:
            print(f"RECUSA: nome com {len(nome)} > {MAX_NOME}: {nome}")
            return 2
        forma = tuple(int(x) for x in v["shape"])
        if len(forma) > MAX_DIMS:
            print(f"RECUSA: {nome} tem {len(forma)} dims (> {MAX_DIMS}); o convert.py exige fix_5d")
            return 2
        n = int(np.prod(forma)) if forma else 1
        dt = v["dtype"]
        a0, a1 = v["data_offsets"]
        if dt == "F32":
            saida = "F32"
        elif dt == "F16":
            saida = "F16"
        elif dt == "BF16":
            if len(forma) <= 1 or n <= LIMIAR_QUANT or any(h in nome for h in hiprec):
                saida = "F32"
            else:
                saida = "BF16"
        else:
            print(f"RECUSA: dtype {dt} em {nome} nao tratado")
            return 2
        nbytes = n * (4 if saida == "F32" else 2)
        total_out += nbytes
        plano.append((k, nome, forma, dt, saida, base + a0, base + a1, n))
    hist_in = {}
    hist_out = {}
    for _, _, _, dt, so, *_ in plano:
        hist_in[dt] = hist_in.get(dt, 0) + 1
        hist_out[so] = hist_out.get(so, 0) + 1
    print(f"tipos    : fonte {hist_in}  ->  saida {hist_out}   ({total_out / GIB:.2f} GiB de dados)")

    livre = shutil.disk_usage(os.path.dirname(os.path.abspath(a.output)) or ".").free
    if livre < total_out + (1 << 30):
        print(f"RECUSA: {livre / GIB:.1f} GiB livres no destino, preciso de {total_out / GIB:.1f} + 1")
        return 2

    # ---- conferencia contra o template (nomes, formas, politica de tipos) ----
    tpl = None
    if a.template:
        r = gguf.GGUFReader(a.template)
        tpl = {t.name: (tuple(int(x) for x in reversed(t.shape)) if False else tuple(int(x) for x in t.shape), t.tensor_type.name, t) for t in r.tensors}
        nomes_aqui = {nome for _, nome, *_ in plano}
        so_tpl = sorted(set(tpl) - nomes_aqui)
        so_aqui = sorted(nomes_aqui - set(tpl))
        print(f"template : {a.template}  {len(tpl)} tensores; so no template {len(so_tpl)}; so aqui {len(so_aqui)}")
        for x in so_tpl[:5]:
            print(f"   so no template: {x}")
        for x in so_aqui[:5]:
            print(f"   so aqui       : {x}")
        if so_tpl or so_aqui:
            print("RECUSA: conjuntos de nomes diferem; o template nao e do mesmo modelo (ou o prefixo esta errado)")
            return 2
        desacordo = 0
        for _, nome, forma, _, saida, *_ in plano:
            tforma, ttipo, _ = tpl[nome]
            # GGUFReader devolve shape em ordem ggml (invertida)
            if tuple(reversed(tforma)) != forma and tforma != forma:
                desacordo += 1
                if desacordo <= 5:
                    print(f"   forma difere: {nome}  aqui {forma}  template {tforma}")
                continue
            esperado = "F32" if ttipo == "F32" else ("F16" if ttipo == "F16" else "BF16")
            if esperado != saida:
                desacordo += 1
                if desacordo <= 5:
                    print(f"   tipo difere : {nome}  aqui {saida}  template {ttipo}")
        print(f"template : formas e politica de tipos: {len(plano) - desacordo}/{len(plano)} concordam")
        if desacordo:
            print("RECUSA: desacordo com o template")
            return 2

    # ---- escrita: memmap somente-leitura da fonte, um tensor por vez ----
    mm = np.memmap(a.input, dtype=np.uint8, mode="r")
    w = gguf.GGUFWriter(path=None, arch=a.arch)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    if "config" in meta:
        w.add_string("config", meta["config"])
        print("metadata : `config` copiado do __metadata__ da fonte")
    elif tpl is not None:
        fld = r.fields.get("config")
        if fld is not None:
            cfg = bytes(fld.parts[fld.data[0]]).decode()
            w.add_string("config", cfg)
            print("metadata : `config` copiado do template")

    t0 = time.time()
    f32_conferidos = 0
    f32_iguais = 0
    for k, nome, forma, dt, saida, o0, o1, n in plano:
        bruto = mm[o0:o1]
        if dt == "BF16" and saida == "BF16":
            arr = bruto.reshape(forma[:-1] + (forma[-1] * 2,))          # bytes crus, forma em bytes
            w.add_tensor(nome, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
        elif dt == "BF16" and saida == "F32":
            arr = bf16_para_f32(np.frombuffer(bruto, dtype=np.uint16).copy()).reshape(forma)
            w.add_tensor(nome, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            if tpl is not None and f32_conferidos < 32:
                _, ttipo, tt = tpl[nome]
                if ttipo == "F32":
                    f32_conferidos += 1
                    if np.asarray(tt.data).reshape(-1).tobytes() == arr.reshape(-1).tobytes():
                        f32_iguais += 1
        elif dt == "F32":
            arr = np.frombuffer(bruto, dtype=np.float32).reshape(forma)
            w.add_tensor(nome, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:  # F16
            arr = np.frombuffer(bruto, dtype=np.float16).reshape(forma)
            w.add_tensor(nome, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
    print(f"plano    : {len(plano)} tensores adicionados ao writer em {time.time() - t0:.1f}s", flush=True)
    if tpl is not None:
        print(f"template : bytes F32 identicos em {f32_iguais}/{f32_conferidos} tensores conferidos", flush=True)

    t0 = time.time()
    w.write_header_to_file(path=parcial)
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    os.replace(parcial, a.output)
    esc = os.path.getsize(a.output)
    print(f"escrito  : {a.output}  {esc} B ({esc / GIB:.2f} GiB) em {time.time() - t0:.0f}s")

    # releitura rapida: cabecalho e histograma de tipos
    r2 = gguf.GGUFReader(a.output)
    h = {}
    for t in r2.tensors:
        h[t.tensor_type.name] = h.get(t.tensor_type.name, 0) + 1
    arch = r2.fields.get("general.architecture")
    arch = bytes(arch.parts[arch.data[0]]).decode() if arch is not None else None
    print(f"releitura: arch {arch}  {len(r2.tensors)} tensores  {h}")
    print("NAO COBERTO: carga e amostragem no ComfyUI (e o render que prova); so ltxv exercitado; "
          "big-endian; fontes F16 nunca vistas aqui; a conferencia de bytes F32 contra o template cobre "
          "ate 32 tensores, nao todos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
