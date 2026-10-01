"""Ids de token de todos os tokenizadores do core do ComfyUI, em CPU (sem GPU).

Uso: python_embeded/python.exe -s tokens.py saida.json
Compara-se o JSON de antes e de depois do upgrade: tem de sair idêntico.
"""
import importlib
import inspect
import json
import os
import pkgutil
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
saida = sys.argv[1]
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, r"F:\COMFY_PORTABLE\ComfyUI")

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.sd1_clip  # noqa: E402
import comfy.text_encoders  # noqa: E402

PROMPTS = [
    "a photo of a cat sitting on a red chair, 35mm, golden hour",
    "Uma mulher de vestido azul caminha na praia ao pôr do sol, câmera lenta.",
    "(masterpiece:1.2), [blurry], embedding:foo, 日本語のテキスト 🐱 \"quoted\" <image1>",
    "",
    "  multiple   spaces\tand\nnewlines  ",
]

res = {}
mods = ["comfy.sd1_clip"] + [f"comfy.text_encoders.{m.name}" for m in pkgutil.iter_modules(comfy.text_encoders.__path__)]
for nome in sorted(mods):
    try:
        mod = importlib.import_module(nome)
    except Exception as e:
        res[nome] = {"import_error": f"{type(e).__name__}: {e}"}
        continue
    for cls_nome, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != nome or not cls_nome.endswith("Tokenizer") or not hasattr(cls, "tokenize_with_weights"):
            continue
        chave = f"{nome}.{cls_nome}"
        try:
            tok = cls(embedding_directory=None, tokenizer_data={})
        except TypeError:
            try:
                tok = cls()
            except Exception as e:
                res[chave] = {"init_error": f"{type(e).__name__}: {e}"}
                continue
        except Exception as e:
            res[chave] = {"init_error": f"{type(e).__name__}: {e}"}
            continue
        saidas = []
        for p in PROMPTS:
            try:
                out = tok.tokenize_with_weights(p)
                saidas.append(json.loads(json.dumps(out, default=lambda o: repr(o))))
            except Exception as e:
                saidas.append({"erro": f"{type(e).__name__}: {e}", "tb": traceback.format_exc().splitlines()[-3:]})
        res[chave] = saidas

import transformers  # noqa: E402
json.dump({"transformers": transformers.__version__, "tokenizadores": res}, open(saida, "w", encoding="utf-8"),
          ensure_ascii=False, indent=0, sort_keys=True)
ok = sum(1 for v in res.values() if isinstance(v, list) and not any(isinstance(x, dict) and "erro" in x for x in v))
print(f"transformers {transformers.__version__}: {len(res)} entradas, {ok} sem erro")
for k, v in res.items():
    if isinstance(v, dict):
        print(" ", k, v)
    elif any(isinstance(x, dict) and "erro" in x for x in v):
        print(" ", k, [x["erro"] for x in v if isinstance(x, dict) and "erro" in x][:1])
