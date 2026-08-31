"""O ComfyUI carrega este encoder quantizado pelo caminho normal, ou nao?

Verificacao estrutural nao e aceitacao (CLAUDE.md). O probe irmao
`probe_winnougan_int4.py` provou que os bytes deste arquivo emitem MMA de 4 bits quando
chamados na mao. Isso NAO prova que o loader do ComfyUI chega ate la.

O QUE SE LEU, e que este probe existe para confirmar ou derrubar. Em `comfy/sd.py`,
`load_text_encoder_state_dicts` chama `comfy.utils.detect_layer_quantization` em UM ramo
so -- o do MiniMax **Music3**, guardado por `"model.audio_decoder.projection.weight"`. O
ramo `TEModel.QWEN3VL_32B`, que e o deste arquivo, nao chama. Se isso valer na execucao,
`model_options["quantization_metadata"]` chega vazio em `sd1_clip.py:110` e o encoder e
construido com Linear normal sobre peso INT4 empacotado.

OS DOIS BRACOS, variando um eixo so:

  A  caminho de estoque      `comfy.sd.load_clip`, que e o que o no CLIPLoader chama
                             (`nodes.py:1031`) -- e que passa por `convert_old_quants`
  B  metadata injetada       identico, mais `model_options["quantization_metadata"]`
                             preenchido com `detect_layer_quantization(sd, "")`

CUIDADO, ja errado uma vez aqui: chamar `load_text_encoder_state_dicts` com um
`load_torch_file` cru NAO e o caminho de estoque. Ele pula `convert_old_quants`
(`comfy/sd.py:1536`), entao nenhum tensor `.comfy_quant` existe e o erro que sai e um
size mismatch 2560-contra-5120 que parece o modelo errado, e nao metadata faltando.

A falhar e B passar => o ComfyUI TEM a maquinaria e nao a liga neste ramo: e um PR de
uma linha, nao um recurso ausente. Os dois falharem => e outra coisa, e o traceback diz.

NAO COBERTO: nao mede qualidade do condicionamento (nao ha referencia BF16 deste modelo
nesta bancada); nao roda o modelo de video, so o encoder; uma placa sm86; um prompt.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:/COMFY_PORTABLE")
CKPT = ROOT / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-int4_convrot.safetensors"
PROMPT = "a red fox walking through snow at dusk"

ARM_SRC = r'''
import json, sys, traceback
from pathlib import Path
ROOT = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(ROOT / "ComfyUI"))

import torch
import comfy.utils, comfy.sd, comfy.model_management

CKPT = %(CKPT)r
INJECT = %(INJECT)s
PROMPT = %(PROMPT)r

import os as _os
from comfy_kitchen.backends import cuda as _ckc
rep = {
    "inject_quant_metadata": INJECT,
    "force_fallback_env": _os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag_seen_by_module": bool(_ckc._FORCE_INT4_INT8_FALLBACK),
}

# Inspecao so para o relatorio: o estado do sd DEPOIS de convert_old_quants, que e o que o
# caminho de estoque faz (`comfy/sd.py:1536`). Um `load_torch_file` cru NAO tem
# `.comfy_quant` -- essa foi a primeira versao errada deste probe.
sd, metadata = comfy.utils.load_torch_file(CKPT, safe_load=True, return_metadata=True)
sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
rep["n_tensors"] = len(sd)
rep["n_comfy_quant"] = sum(1 for k in sd if k.endswith(".comfy_quant"))
rep["te_model"] = str(comfy.sd.detect_te_model(sd))
quant = comfy.utils.detect_layer_quantization(sd, "")
rep["detect_layer_quantization_finds"] = None if quant is None else len(quant.get("layers", quant))
del sd
import gc; gc.collect()

model_options = {}
if INJECT and quant is not None:
    model_options["quantization_metadata"] = quant

try:
    # o caminho de estoque de verdade: o que `CLIPLoader.load_clip` chama (`nodes.py:1031`)
    clip = comfy.sd.load_clip(
        ckpt_paths=[CKPT], clip_type=comfy.sd.CLIPType.MINIMAX, model_options=model_options)
    rep["loaded"] = True
except Exception:
    rep["loaded"] = False
    rep["load_traceback"] = traceback.format_exc()[-2500:]
    print("@@JSON@@" + json.dumps(rep)); sys.exit(0)

# que tipo de modulo ficou nas Linear quantizadas?
tr = clip.cond_stage_model.qwen3vl_32b.transformer
seen = {}
example = None
for name, mod in tr.named_modules():
    if name.endswith("self_attn.q_proj") or name.endswith("mlp.down_proj"):
        t = type(mod).__name__
        seen[t] = seen.get(t, 0) + 1
        if example is None:
            w = getattr(mod, "weight", None)
            example = {
                "module": name, "type": t,
                "weight_type": type(w).__name__ if w is not None else None,
                "weight_dtype": str(w.dtype) if w is not None else None,
                "weight_shape": list(w.shape) if w is not None else None,
                "has_comfy_quant_attr": hasattr(mod, "comfy_quant"),
            }
rep["linear_module_types"] = seen
rep["example_layer"] = example

try:
    tokens = clip.tokenize(PROMPT)
    out = clip.encode_from_tokens_scheduled(tokens)
    cond = out[0][0]
    rep["encoded"] = True
    rep["cond_shape"] = list(cond.shape)
    rep["cond_dtype"] = str(cond.dtype)
    f = cond.float()
    rep["cond_norm"] = float(f.norm())
    rep["cond_absmean"] = float(f.abs().mean())
    rep["cond_has_nan"] = bool(torch.isnan(f).any())
    rep["cond_all_zero"] = bool((f == 0).all())
    rep["cond_fingerprint"] = [float(v) for v in f.reshape(-1)[:8]]
except Exception:
    rep["encoded"] = False
    rep["encode_traceback"] = traceback.format_exc()[-2500:]

rep["vram_used_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
print("@@JSON@@" + json.dumps(rep))
'''


def run_arm(inject, force_fallback=False):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1" if force_fallback else "0"
    src = ARM_SRC % {"CKPT": str(CKPT), "INJECT": "True" if inject else "False", "PROMPT": PROMPT}
    proc = subprocess.run(
        [str(ROOT / "python_embeded/python.exe"), "-s", "-c", src],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):]), proc
    return None, proc


def show(label, rep, proc):
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    if rep is None:
        print(f"braco morreu sem devolver JSON (rc={proc.returncode})")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        return
    print(f"tensores no state dict          {rep['n_tensors']}")
    print(f"tensores .comfy_quant           {rep['n_comfy_quant']}")
    print(f"detect_te_model                 {rep['te_model']}")
    print(f"detect_layer_quantization acha  {rep['detect_layer_quantization_finds']} camadas")
    print(f"model_options injetado          {rep['inject_quant_metadata']}")
    print(f"FORCE_INT4_INT8_FALLBACK        env={rep['force_fallback_env']}  "
          f"visto pelo modulo={rep['flag_seen_by_module']}")
    print(f"carregou                        {rep['loaded']}")
    if not rep["loaded"]:
        print(rep["load_traceback"])
        return
    print(f"tipos de modulo nas Linear      {rep['linear_module_types']}")
    print(f"exemplo                         {rep['example_layer']}")
    print(f"codificou                       {rep['encoded']}")
    if rep["encoded"]:
        print(f"cond                            shape {rep['cond_shape']} {rep['cond_dtype']}")
        print(f"                                norm {rep['cond_norm']:.4f}  "
              f"absmean {rep['cond_absmean']:.6f}")
        print(f"                                nan={rep['cond_has_nan']}  "
              f"tudo-zero={rep['cond_all_zero']}")
        print(f"pico VRAM alocada               {rep['vram_used_mib']} MiB")
    else:
        print(rep["encode_traceback"])


def main():
    print(f"arquivo {CKPT.name}  {CKPT.stat().st_size / 1024**3:.2f} GiB")
    a, pa = run_arm(False)
    show("BRACO A -- caminho de estoque (o que um CLIPLoader faz hoje)", a, pa)
    b, pb = run_arm(True)
    show("BRACO B -- identico, mais quantization_metadata injetada", b, pb)
    c, pc = run_arm(False, force_fallback=True)
    show("BRACO C -- estoque, com COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1", c, pc)

    print()
    print("-" * 78)
    print("VEREDITO")
    print("-" * 78)
    ok = lambda r: bool(r and r.get("loaded") and r.get("encoded") and not r.get("cond_has_nan")
                        and not r.get("cond_all_zero"))
    oa, ob = ok(a), ok(b)
    if oa and ob:
        print("os dois passam: o ramo QWEN3VL_32B ja lida com o checkpoint quantizado.")
    elif ob and not oa:
        print("A falha, B passa: a maquinaria existe e NAO e ligada no ramo QWEN3VL_32B de")
        print("`load_text_encoder_state_dicts`. Candidato a PR de uma linha -- e este probe")
        print("e execucao, nao leitura, entao serve para abrir um.")
    elif oa and not ob:
        print("A passa e B falha -- inesperado. Ler os dois tracebacks antes de concluir.")
    else:
        print("os dois falham: nao e a metadata. Ler o traceback do braco A.")
    if a and b and oa and ob:
        na, nb = a.get("cond_norm"), b.get("cond_norm")
        if na and nb:
            r = max(na, nb) / min(na, nb)
            print(f"norma do condicionamento: A {na:.4f}  B {nb:.4f}  ({r:.3f}x entre si)")

    print()
    print("-" * 78)
    print("o ENCODE passou mesmo pelo ramo INT4 nativo?")
    print("-" * 78)
    if not (ok(a) and ok(c)):
        print("indeterminado: A ou C nao completou.")
    else:
        na, nc = a["cond_norm"], c["cond_norm"]
        same = a["cond_fingerprint"] == c["cond_fingerprint"]
        print(f"A  nativo    norma {na:.4f}  absmean {a['cond_absmean']:.6f}")
        print(f"C  fallback  norma {nc:.4f}  absmean {c['cond_absmean']:.6f}")
        if same:
            print("IDENTICOS bit a bit -> os dois bracos tomaram o MESMO caminho. O encode NAO")
            print("  esta emitindo o kernel de 4 bits, apesar de o arquivo pedir int4.")
        else:
            r = max(na, nc) / min(na, nc)
            print(f"DIFEREM ({r:.4f}x na norma) -> o encode inteiro, com os 50 blocos e o")
            print("  loader de estoque, desce mesmo pelo ramo INT4 nativo. Nao e so o kernel")
            print("  chamado na mao: e o caminho que um usuario percorre.")

    print()
    print("NAO COBERTO: sem referencia BF16 deste modelo, entao nada aqui afirma FIDELIDADE")
    print("  do condicionamento -- so que ele carrega, roda e nao sai nan nem zero. Nao roda")
    print("  o modelo de video. Um prompt, uma placa sm86, um processo por braco.")


if __name__ == "__main__":
    main()
