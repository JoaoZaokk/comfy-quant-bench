"""Quais `convrot_groupsize` o comfy-kitchen REALMENTE aceita, op por op?

DE ONDE VEIO
------------
Em 2026-09-01, tentando construir um Z-Image com rotacao mais grossa para achar onde ele quebra, o
`quant_mixed` morreu com:

    RuntimeError: convrot rotate kernel only supports group_size 256

Isso contradiz duas coisas escritas nesta bancada:

  * `quant_w4a4.py --convrot-groupsize` diz, no proprio texto de ajuda: *"O loader le isso de volta
    do sidecar, entao um arquivo convertido em 64 e executado em 64."* -- o que so faz sentido se
    64 for um valor utilizavel.
  * `CLAUDE.md` registra, de `tools/probe_backend_resolution.py`: *"todas as quatro combinacoes --
    cg 64 e 256, tensores dummy e reais -- resolvem para `comfy_kitchen.backends.cuda`, e as duas
    chamadas reais funcionam."*

Resolver e executar sao coisas diferentes, e uma delas pode ter mudado entre comfy-kitchen 0.2.23
(quando aquilo foi medido) e 0.2.31 (hoje). Este arquivo mede as DUAS pontas, por valor, com o
`RuntimeError` capturado em vez de derrubar a varredura -- porque a lista de valores que falham e
justamente o resultado.

NAO COBERTO
-----------
Uma versao de comfy-kitchen, uma placa (sm86), uma forma de peso por chamada. Nao diz o que
acontece com um checkpoint JA escrito num cg nao suportado -- se ele carrega e quebra no forward,
ou se nem carrega; isso e outro teste. E nao mede qualidade: aceitar nao e ser bom.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

VALORES = [16, 64, 128, 256, 512, 1024]


def main() -> int:
    import torch
    import comfy_kitchen as ck

    if not torch.cuda.is_available():
        print("sem CUDA -- este probe mede o kernel, nao ha o que medir", file=sys.stderr)
        return 2

    print(f"comfy_kitchen {getattr(ck, '__version__', '?')}   torch {torch.__version__}")
    print(f"placa {torch.cuda.get_device_name(0)}\n")
    print(f"{'cg':>6}  {'quantize_convrot_w4a4_weight':<32} {'convrot_w4a4_linear':<32} "
          f"{'quantize_w4a8_int8_weight':<32}")
    print("-" * 108)

    K = 2048  # divisivel por todos os valores testados, para que a falha NAO seja de divisibilidade
    w = torch.randn(512, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)

    aceitos = []
    for cg in VALORES:
        try:
            q, s = ck.quantize_convrot_w4a4_weight(w, cg, 64)
            r_q = "ok"
        except Exception as e:
            r_q, q, s = f"{type(e).__name__}: {str(e)[:44]}", None, None
        if q is None:
            r_l = "(nao chegou a testar)"
        else:
            try:
                ck.convrot_w4a4_linear(x, q, s, None, cg)
                r_l = "ok"
            except Exception as e:
                r_l = f"{type(e).__name__}: {str(e)[:44]}"
        # O W4A8 tambem recebe um convrot_groupsize, e o `quant_mixed` mede os DOIS formatos por
        # camada. Se so este recusar, o limite nao e "do kernel convrot" -- e do caminho W4A8, e a
        # mensagem generica leva a conclusao errada. Foi o que aconteceu comigo.
        try:
            ck.quantize_w4a8_int8_weight(w, group_size=16, convrot_groupsize=cg, symmetric=True,
                                         scale_dtype=torch.float8_e4m3fn, codebook=True,
                                         codebook_tensor=None, stochastic_rounding=0)
            r_8 = "ok"
        except Exception as e:
            r_8 = f"{type(e).__name__}: {str(e)[:44]}"
        if r_q == "ok" and r_l == "ok":
            aceitos.append(cg)
        print(f"{cg:>6}  {r_q:<32} {r_l:<32} {r_8:<32}")

    print()
    print(f"ACEITOS de ponta a ponta: {aceitos}")
    if aceitos == [256]:
        print("  -> 256 e o UNICO valor utilizavel neste build. Toda documentacao que sugere")
        print("     escolher outro esta desatualizada, e `--convrot-groupsize` so tem um valor.")
    print()
    print("NAO COBERTO: uma versao do comfy-kitchen, uma placa, uma forma de peso. K=2048 e")
    print("  divisivel por todos os valores testados, entao nenhuma falha aqui e por")
    print("  divisibilidade. Aceitar nao e ser bom: isto mede o que roda, nao o que presta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
