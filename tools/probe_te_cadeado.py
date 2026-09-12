r"""O cadeado do text encoder virou uma flag. Quanto custa solta-la, quanto rende, e o controle.

DE ONDE VEM
-----------
Ate 2026-09-12 o ComfyUI carregava um text encoder quantizado e rodava a matematica
**dequantizada**, por duas travas independentes:

    comfy/sd.py       set_model_compute_dtype(torch.float32)  -> comfy_force_cast_weights=True
    comfy/sd1_clip.py mixed_precision_ops(..., full_precision_mm=True)

Qualquer uma sozinha basta para o kernel nunca ser alcancado. O resultado era peso de 4 ou 8 bits
na VRAM e conta em BF16: **economiza memoria, nao economiza tempo**. Medido em 2026-08-31 contando
chamadas, e confirmado de novo em 2026-09-12: 336 de 336 `dequantize`, zero forwards quantizados.

As duas agora sao soltas juntas quando o checkpoint carrega camadas quantizadas de verdade, e
`--disable-quantized-text-encoder` repoe o comportamento antigo. Este probe mede o que essa troca
faz, porque "o kernel dispara" nao e a mesma afirmacao que "vale a pena".

O DESENHO
---------
Um eixo varia: a flag. Mesmo arquivo, mesmo prompt, mesma placa, mesmo processo-filho limpo por
braco -- limpo porque `model_patcher.py:1016` reescreve `comfy_force_cast_weights` em cada modulo
toda vez que o modelo sobe para a GPU, entao estado de VRAM herdado de um braco contamina o outro.

Tres medidas por braco:

    forwards quantizados / dequantize   o kernel disparou? (conta, nao infere)
    tempo de encode, mediana de N       o que se ganha, se e que se ganha
    condicionamento                     o que se perde: erro relativo e cosseno contra o
                                        braco travado, que e a saida que o ComfyUI produzia antes

O CONTROLE QUE TEM DE PASSAR
----------------------------
Um encoder **nao quantizado** tem de continuar exatamente como era: `comfy_force_cast_weights`
ligado, upcast para float32 aplicado. Sem esse braco, uma mudanca que destravasse TODO encoder
passaria neste probe com louvor -- e quebraria todo workflow BF16 da maquina em silencio.

NAO COBERTO
-----------
Um prompt por chamada (o comprimento importa: medido em 2026-08-31, a virada fica entre 75 e 199
tokens num encoder de 4 B). Uma placa. Nenhuma imagem: isto mede o condicionamento, e o que uma
diferenca de condicionamento faz com a imagem final e outra pergunta.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

FILHO = r'''
import json, statistics, sys, time, traceback
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.cli_args

TRAVADO = %(TRAVADO)s
comfy.cli_args.args.disable_quantized_text_encoder = TRAVADO

CKPT = %(CKPT)r; CLIP_TYPE = %(CLIP_TYPE)r; PROMPT = %(PROMPT)r; REPETE = %(REPETE)d
rep = {"travado": TRAVADO, "ckpt": CKPT}
try:
    clip = comfy.sd.load_clip(
        ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CKPT)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, CLIP_TYPE))

    quantizadas = [m for _, m in clip.cond_stage_model.named_modules()
                   if getattr(m, "layout_type", None) is not None]
    rep["n_quantizadas"] = len(quantizadas)
    rep["force_cast_weights_patcher"] = bool(clip.patcher.force_cast_weights)
    if quantizadas:
        rep["full_precision_mm"] = bool(getattr(quantizadas[0], "_full_precision_mm", True))

    fichas = clip.tokenize(PROMPT)
    # O comprimento do prompt decide o SINAL do ganho, entao um `None` aqui torna a tabela
    # ilegivel. Os tokenizadores devolvem formas diferentes (dict de listas de listas, ou lista
    # direta), entao conta-se pela estrutura em vez de assumir uma delas.
    def _conta(v):
        if isinstance(v, dict):
            return sum(_conta(x) for x in v.values())
        if isinstance(v, list):
            return sum(_conta(x) for x in v) if v and isinstance(v[0], list) else len(v)
        return 0
    rep["tokens"] = _conta(fichas)
    tempos = []
    saida = None
    for _ in range(REPETE):
        torch.cuda.synchronize()
        t = time.perf_counter()
        cond = clip.encode_from_tokens_scheduled(fichas)
        torch.cuda.synchronize()
        tempos.append(time.perf_counter() - t)
        saida = cond[0][0]
    rep["ms"] = round(statistics.median(tempos) * 1000, 1)
    rep["ms_todos"] = [round(x * 1000, 1) for x in tempos]
    # O condicionamento vai para disco: comparar dois tensores exige os dois, e cada braco
    # roda no seu proprio processo justamente para nao herdar estado de VRAM do outro.
    torch.save(saida.detach().float().cpu(), %(SAIDA)r)
    rep["shape"] = list(saida.shape)
    rep["ok"] = True
except Exception:
    rep["ok"] = False
    rep["traceback"] = traceback.format_exc()[-2000:]
print("@@JSON@@" + json.dumps(rep))
'''


def roda(ckpt: str, clip_type: str, prompt: str, repete: int, travado: bool,
         saida: Path, device: int) -> dict:
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    src = FILHO % {"TRAVADO": "True" if travado else "False", "CKPT": ckpt,
                   "CLIP_TYPE": clip_type, "PROMPT": prompt, "REPETE": repete,
                   "SAIDA": str(saida)}
    proc = subprocess.run([str(RAIZ / "python_embeded/python.exe"), "-s", "-c", src],
                          capture_output=True, text=True, env=env, cwd=str(RAIZ))
    for linha in proc.stdout.splitlines():
        if linha.startswith("@@JSON@@"):
            return json.loads(linha[len("@@JSON@@"):])
    print(proc.stdout[-3000:], proc.stderr[-3000:])
    raise SystemExit(f"o braco {'travado' if travado else 'destravado'} nao emitiu relatorio")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="arquivo dentro de models/text_encoders")
    p.add_argument("--clip-type", default="LUMINA2")
    p.add_argument("--prompt", default="a crowded night market street at dusk, food stalls under "
                                       "paper lanterns, people walking, wet pavement reflecting "
                                       "the signs, shot on a 35mm lens with shallow depth of field")
    p.add_argument("--repete", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--referencia", default=None,
                   help="o BF16 original deste encoder. Sem ele os dois bracos so se comparam "
                        "entre si, o que diz o TAMANHO da diferenca e nao QUAL dos dois esta "
                        "certo -- e os dois usam o mesmo peso quantizado, entao a pergunta que "
                        "decide e a distancia ate o BF16.")
    p.add_argument("--controle", default=None,
                   help="um encoder NAO quantizado, que tem de sair igual nos dois bracos. Sem "
                        "ele o probe nao distingue 'soltou so o quantizado' de 'soltou todos'.")
    a = p.parse_args()

    tmp = RAIZ / ".scratch"
    tmp.mkdir(exist_ok=True)
    destravado = roda(a.ckpt, a.clip_type, a.prompt, a.repete, False, tmp / "te_destravado.pt", a.device)
    travado = roda(a.ckpt, a.clip_type, a.prompt, a.repete, True, tmp / "te_travado.pt", a.device)

    for r in (destravado, travado):
        if not r.get("ok"):
            print(r.get("traceback", "sem traceback"))
            return 1

    print(f"\n{a.ckpt}   {destravado.get('n_quantizadas', 0)} camadas quantizadas   "
          f"{destravado.get('tokens')} tokens\n")
    print(f"{'braco':14s} {'force_cast':>11s} {'fp_mm':>7s} {'ms':>9s} {'ganho':>9s}")
    for nome, r in (("travado", travado), ("destravado", destravado)):
        ganho = "-" if nome == "travado" else f"{travado['ms'] / r['ms']:.2f}x"
        print(f"{nome:14s} {str(r['force_cast_weights_patcher']):>11s} "
              f"{str(r.get('full_precision_mm')):>7s} {r['ms']:9.1f} {ganho:>9s}")
    print(f"   medicoes cruas travado {travado['ms_todos']}  destravado {destravado['ms_todos']}")

    import torch
    a_t = torch.load(tmp / "te_travado.pt")
    b_t = torch.load(tmp / "te_destravado.pt")
    if a_t.shape != b_t.shape:
        print(f"\nFORMAS DIFERENTES {list(a_t.shape)} contra {list(b_t.shape)}: nao comparaveis")
    else:
        # float64 na comparacao, e nao por zelo: o condicionamento do Gemma tem ~192 M elementos,
        # e o primeiro resultado deste probe saiu com **cosseno 1,002611** -- impossivel por
        # definicao, e puro erro de acumulacao em float32. Um numero fora do dominio da propria
        # metrica e defeito do instrumento, nao medida.
        x = a_t.double().flatten()
        y = b_t.double().flatten()
        rel = ((y - x).norm() / x.norm()).item()
        cos = (torch.dot(x, y) / (x.norm() * y.norm())).item()
        print(f"\ncondicionamento destravado contra travado: rel-RMSE {rel:.4e}   cosseno {cos:.6f}")
        print("   o braco travado e a saida que o ComfyUI produzia antes da flag; esta linha diz")
        print("   o quanto a saida muda, NAO qual das duas esta mais perto do BF16 original.")

    if a.referencia:
        print(f"\n--- referencia BF16: {a.referencia} ---")
        # `--repete` tambem aqui, e nao 1: com uma corrida so o mesmo BF16 mediu 282,2 ms num
        # braco e 434,1 ms no outro -- 1,54x de diferenca no MESMO arquivo com o MESMO prompt,
        # que e aquecimento, nao velocidade. Um numero de referencia que varia mais que o efeito
        # medido nao e referencia.
        ref = roda(a.referencia, a.clip_type, a.prompt, a.repete, False, tmp / "te_ref.pt", a.device)
        if not ref.get("ok"):
            print(ref.get("traceback", "sem traceback"))
            return 1
        r_t = torch.load(tmp / "te_ref.pt")
        if r_t.shape != a_t.shape:
            print(f"  FORMAS DIFERENTES {list(r_t.shape)} contra {list(a_t.shape)}: nao comparaveis")
        else:
            r = r_t.double().flatten()
            print(f"  {'braco':14s} {'rel-RMSE vs BF16':>18s} {'cosseno':>10s} {'ms':>9s}")
            piores = {}
            for nome, t, ms in (("travado", a_t, travado["ms"]),
                                ("destravado", b_t, destravado["ms"])):
                v = t.double().flatten()
                e = ((v - r).norm() / r.norm()).item()
                c = (torch.dot(v, r) / (v.norm() * r.norm())).item()
                piores[nome] = e
                print(f"  {nome:14s} {e:18.4e} {c:10.6f} {ms:9.1f}")
            razao = piores["destravado"] / piores["travado"]
            direcao = "MENOS fiel" if razao > 1 else "mais fiel"
            print(f"  destravar deixa a saida {max(razao, 1 / razao):.2f}x {direcao}, "
                  f"e {travado['ms'] / destravado['ms']:.2f}x mais rapida.")
            print(f"  BF16 de referencia levou {ref['ms']:.1f} ms com {ref.get('n_quantizadas')} "
                  "camadas quantizadas.")

    if a.controle:
        print(f"\n--- controle, encoder nao quantizado: {a.controle} ---")
        c_des = roda(a.controle, a.clip_type, a.prompt, 1, False, tmp / "te_ctrl_d.pt", a.device)
        c_tra = roda(a.controle, a.clip_type, a.prompt, 1, True, tmp / "te_ctrl_t.pt", a.device)
        for nome, r in (("travado", c_tra), ("destravado", c_des)):
            print(f"  {nome:12s} camadas quantizadas {r.get('n_quantizadas')}   "
                  f"force_cast {r['force_cast_weights_patcher']}")
        if c_des.get("force_cast_weights_patcher") and c_tra.get("force_cast_weights_patcher"):
            print("  CONTROLE PASSOU: um encoder sem camadas quantizadas continua com o upcast.")
        else:
            print("  CONTROLE FALHOU: a flag alcancou um encoder que nao e quantizado.")
            return 1
    else:
        print("\nSEM CONTROLE. Rode com --controle <encoder bf16> antes de acreditar que a flag")
        print("alcanca so o que e quantizado.")

    print("\nNAO COBERTO: um prompt (o comprimento decide o sinal do ganho -- a virada ficou "
          "entre 75 e 199 tokens\nnum encoder de 4 B, medido 2026-08-31), uma placa, nenhuma "
          "imagem. Isto mede o condicionamento,\nnao o que ele faz com a foto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
