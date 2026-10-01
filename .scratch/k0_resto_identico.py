"""K0 do criterio b6/b7: os tensores FORA do corpo ternario do exportado sao byte a byte os do BF16 original?
uso: k0_resto_identico.py <braco_bfl> <bf16_bfl>"""
import sys
sys.path.insert(0, "tools")
from compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais
from transplanta_klein import bruto
B, O = Leitor(sys.argv[1]), Leitor(sys.argv[2])
pil = pilhas_reais(B.h)
corpo = {k for k, m in B.h.items() if len(m["shape"]) == 2 and (x := BLOCO.match(k)) and x.group("pilha") in pil}
resto = sorted(set(B.h) - corpo)
dif = [k for k in resto if bruto(B, k) != bruto(O, k)]
print(f"K0: resto {len(resto)} tensores, {len(resto) - len(dif)} identicos byte a byte, {len(dif)} diferentes")
for k in dif[:10]:
    print("   diferente:", k)
print("K0", "PASSOU" if not dif else "FALHOU")
