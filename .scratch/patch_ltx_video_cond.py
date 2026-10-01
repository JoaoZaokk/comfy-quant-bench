"""ltx_video.py: --encode-only (gemma -> LTXVSaveConditioning) e --cond-from (LTXVLoadConditioning),
para o braco BF16 nao ter o encoder de 22,7 GiB residente em RAM enquanto o transformer de 39 GiB
carrega. O que matou o servidor tres vezes nao foi RAM fisica: foi COMMIT -- 1,6 GiB livres de
127,7 no momento da terceira morte, com o proprio servidor em 59,8 GiB de commit ainda carregando."""
import ast
import io

p = 'tools/ltx_video.py'
s = io.open(p, encoding='utf-8').read()

# 1) clip nodes: either real encoder or loaded conditioning
old = """    g.update({
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["2", 0]}},
"""
new = """    if a.cond_from:
        # Condicionamento pre-computado (ver --encode-only): o encoder nem e carregado. Os
        # arquivos moram em models/embeddings/, que e onde LTXVSaveConditioning grava.
        g.pop("2", None)
        cond_nodes = {
            "3": {"class_type": "LTXVLoadConditioning",
                  "inputs": {"file_name": f"{a.cond_from}_pos.safetensors", "device": "gpu"}},
            "4": {"class_type": "LTXVLoadConditioning",
                  "inputs": {"file_name": f"{a.cond_from}_neg.safetensors", "device": "gpu"}},
        }
    else:
        cond_nodes = {
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["2", 0]}},
        }
    if a.encode_only:
        # So o encoder: codifica os dois textos e grava. Depois disto o chamador faz POST /free
        # e roda o sampler com --cond-from, sem o encoder na memoria.
        g = {k: v for k, v in g.items() if k in ("2",)}
        g.update(cond_nodes)
        g["30"] = {"class_type": "LTXVSaveConditioning",
                   "inputs": {"conditioning": ["3", 0], "filename": f"{a.saida}_pos", "dtype": "bfloat16"}}
        g["31"] = {"class_type": "LTXVSaveConditioning",
                   "inputs": {"conditioning": ["4", 0], "filename": f"{a.saida}_neg", "dtype": "bfloat16"}}
        return g
    g.update(cond_nodes)
    g.update({
"""
assert old in s
s = s.replace(old, new)

# 2) in encode-only mode the model/vae loaders must not be built: skip them when encode_only
old2 = """    if a.checkpoint:
        # ---- LTX 2.3: checkpoint unico ----"""
new2 = """    if a.encode_only and a.checkpoint:
        # so o encoder de texto; nada de modelo, VAEs ou LoRA
        proj_ck = a.proj_checkpoint or a.checkpoint
        g["2"] = {"class_type": "LTXAVTextEncoderLoader",
                  "inputs": {"text_encoder": a.encoder, "ckpt_name": proj_ck, "device": "default"}}
    elif a.encode_only:
        g["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": a.encoder, "type": "ltxv"}}
    elif a.checkpoint:
        # ---- LTX 2.3: checkpoint unico ----"""
assert old2 in s
s = s.replace(old2, new2)

# 3) args
old3 = """    p.add_argument("--lora", default=None, help="arquivo em loras/ (segundo eixo, so para teste de LoRA)")"""
new3 = """    p.add_argument("--encode-only", action="store_true",
                   help="so codifica prompt e negativo com o encoder e grava em models/embeddings/ como "
                        "<saida>_pos/<saida>_neg (LTXVSaveConditioning). Nada e amostrado")
    p.add_argument("--cond-from", default=None,
                   help="prefixo gravado por --encode-only: amostra com LTXVLoadConditioning e NAO carrega "
                        "o encoder. E o que deixa o transformer BF16 de 39 GiB caber no commit da maquina")
    p.add_argument("--lora", default=None, help="arquivo em loras/ (segundo eixo, so para teste de LoRA)")"""
assert old3 in s
s = s.replace(old3, new3)

# 4) validation + prints + record
old4 = """    if not a.checkpoint and not a.transformer:
        raise SystemExit("modo 2.5 exige --transformer; modo 2.3 exige --checkpoint")"""
new4 = """    if a.encode_only and a.cond_from:
        raise SystemExit("--encode-only e --cond-from sao passos opostos; um por chamada")
    if not a.checkpoint and not a.transformer and not a.encode_only:
        raise SystemExit("modo 2.5 exige --transformer; modo 2.3 exige --checkpoint")"""
assert old4 in s
s = s.replace(old4, new4)

old5 = """    prompt = monta_prompt(a)
    modelo = a.gguf or a.transformer or a.checkpoint"""
new5 = """    prompt = monta_prompt(a)
    modelo = a.gguf or a.transformer or a.checkpoint
    if a.encode_only:
        print(f"SO ENCODER  : {a.encoder} -> models/embeddings/{a.saida}_pos|_neg.safetensors", flush=True)
    if a.cond_from:
        print(f"condicion.  : de models/embeddings/{a.cond_from}_pos|_neg.safetensors (encoder NAO carregado)", flush=True)"""
assert old5 in s
s = s.replace(old5, new5)

old6 = """           "lora": a.lora, "lora_strength": a.lora_strength if a.lora else None,"""
new6 = """           "encode_only": bool(a.encode_only), "cond_from": a.cond_from,
           "lora": a.lora, "lora_strength": a.lora_strength if a.lora else None,"""
assert old6 in s
s = s.replace(old6, new6)

# encode-only: frames validation must not block (frames irrelevant) -- keep, harmless; but the
# 'nenhum arquivo de audio' warning would fire: guard it
old7 = """    if not arq["audio"]:
        print("  ATENCAO: nenhum arquivo de audio na saida -- o ramo de audio nao foi gravado",
              flush=True)"""
new7 = """    if not arq["audio"] and not a.encode_only:
        print("  ATENCAO: nenhum arquivo de audio na saida -- o ramo de audio nao foi gravado",
              flush=True)"""
assert old7 in s
s = s.replace(old7, new7)

ast.parse(s)
io.open(p, 'w', encoding='utf-8').write(s)
print('ltx_video.py: --encode-only / --cond-from added; syntax ok')
