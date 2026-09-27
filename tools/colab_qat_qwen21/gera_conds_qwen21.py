"""Condicionamentos de texto do Qwen-Image-2.1 para o QAT por bloco, pelo mesmo caminho do node TextEncodeQwenImage21.

O QAT roda na VM sem o encoder (Qwen3-VL 8B): aqui o ComfyUI local codifica os prompts e grava so os tensores de
contexto. Treino e holdout em arquivos de texto, um prompt por linha. O holdout e a bateria de 26/09 (nunca vista no
treino), para o render final comparar com as outras quants nas mesmas imagens.

    python_embeded\\python.exe -s tools/colab_qat_qwen21/gera_conds_qwen21.py --treino T.txt --holdout H.txt --out conds.pt
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "ComfyUI"))


def le(p):
    return [l.strip() for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--clip", default="qwen3vl_8b_w4a8.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"recusado: {a.out} existe")
    import torch
    import comfy.sd
    import folder_paths
    import utils.extra_config
    extra = RAIZ / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        utils.extra_config.load_extra_path_config(str(extra))
    clip = comfy.sd.load_clip(ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", a.clip)],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    saida = {"clip": a.clip}
    for nome, arq in (("treino", a.treino), ("holdout", a.holdout)):
        lista = []
        for p in le(arq):
            # mesmo tokenize do TextEncodeQwenImage21 sem imagens de referencia
            toks = clip.tokenize(p, images=[], keep_vision=True, prevent_empty_text=True)
            c = clip.encode_from_tokens_scheduled(toks)
            lista.append(c[0][0].detach().to("cpu", torch.bfloat16).clone())
        saida[nome] = lista
        saida[nome + "_prompts"] = le(arq)
        print(nome, len(lista), [tuple(t.shape) for t in lista[:3]], flush=True)
    parcial = a.out.with_suffix(a.out.suffix + ".partial")
    torch.save(saida, parcial)
    parcial.replace(a.out)
    print("gravado", a.out, a.out.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
