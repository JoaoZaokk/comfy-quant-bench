"""Regressao de `avaliar.checar_backend`: as DUAS formas do campo `backend`, e o caso que tem de FALHAR.

Por que este arquivo existe. Em 2026-09-21 a camada 1 devolveu `backend_sem_registro` para o
`10Eros_v1.5_bf16_w4a8` com a mensagem *"sidecar presente e SEM campo `backend`"* -- e o sidecar
trazia `"backend": "comfy_kitchen.backends.cuda"`, la, legivel. O veredito contradizia a evidencia
que estava no arquivo, que e a forma mais caruma de erro: ele parece uma medicao.

A causa: `checar_backend` exigia `isinstance(backend, dict)`, mas **quatro dos cinco escritores
gravam uma string** -- `quant_w4a4.py:480`, `quant_w4a8.py:366`, `quant_int8.py:239` e
`quant_w4a4_smooth.py:330`, todos `backend["resolved"][<um op>]` -- e so `quant_mixed.py:946` grava
o dicionario `{op: impl}`. Medido nos quatro roots no mesmo dia: **14 dos 46 sidecars** em disco
traziam string, incluindo builds ja publicados (`capybara_v0.1_w4a8`,
`gemma_3_12B_it_heretic_w4a8`, `hv15_w4a8`).

**O CASO QUE TEM DE FALHAR e o unico que nao pode ser removido deste arquivo.** Um conserto que so
faz o falso negativo parar de aparecer pode ter sido feito desligando a checagem; o que prova que ela
ainda morde e um backend NAO nativo, na forma de string, disparando `backend_nao_nativo`. Sem esse
caso, um `checar_backend` que apenas `return`-asse passaria em todo o resto daqui.

Rodar: `.\\python_embeded\\python.exe -s .\\tools\\test_avaliar_backend.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import avaliar  # noqa: E402

NATIVO = "comfy_kitchen.backends.cuda"
EAGER = "comfy_kitchen.backends.eager"


def ck(sidecar: dict, camadas: int = 8, formato: str = "asym_w4a8_int8") -> avaliar.Checkpoint:
    return avaliar.Checkpoint(
        caminho=Path("sintetico.safetensors"), tamanho=1,
        tensores={}, metadata={},
        quant={"layers": {f"l{i}": {"format": formato} for i in range(camadas)}},
        sidecar=sidecar, candidatas=[], sobra_bytes=0, dialeto="sintetico")


def achados(sidecar: dict, **kw) -> list[avaliar.Achado]:
    return list(avaliar.checar_backend(ck(sidecar, **kw)))


def codigos(sidecar: dict, **kw) -> list[str]:
    return [a.codigo for a in achados(sidecar, **kw)]


CASOS: list[tuple[str, dict, list[str], str]] = [
    # --- as duas formas nativas: nenhum achado
    ("string nativa, como quant_w4a8/quant_w4a4/quant_int8/smooth gravam",
     {"backend": NATIVO}, [],
     "e a forma de 14 dos 46 sidecars em disco; era o falso negativo"),
    ("dicionario nativo, como quant_mixed grava",
     {"backend": {"quantize_convrot_w4a4_weight": NATIVO, "convrot_w4a4_linear": NATIVO}}, [],
     "forma que sempre funcionou; nao pode regredir"),
    ("string nativa + backend_linear, como quant_w4a4_smooth grava as duas",
     {"backend": NATIVO, "backend_linear": NATIVO}, [],
     "smooth grava dois campos de string"),

    # --- O CASO QUE TEM DE FALHAR. Nao remover.
    ("CONTROLE QUE TEM DE FALHAR: string NAO nativa",
     {"backend": EAGER}, ["backend_nao_nativo"],
     "se isto parar de disparar, o conserto desligou a checagem"),
    ("CONTROLE QUE TEM DE FALHAR: dicionario com um op fora",
     {"backend": {"quantize_w4a8_int8_weight": NATIVO, "w4a8_int8_linear": EAGER}},
     ["backend_nao_nativo"], "a forma dict tem de continuar mordendo op por op"),
    ("CONTROLE QUE TEM DE FALHAR: backend_linear nao nativo com backend nativo",
     {"backend": NATIVO, "backend_linear": EAGER}, ["backend_nao_nativo"],
     "o segundo campo do smooth nao pode passar batido"),

    # --- ausencia e isencao: continuam avisando, com motivos DIFERENTES
    ("sem sidecar nenhum -- terceiro",
     {}, ["backend_sem_registro"], "nao e defeito do arquivo, e falta de registro"),
    ("sidecar sem a chave backend",
     {"quantization": "asym_w4a8_int8"}, ["backend_sem_registro"],
     "conversao nossa antiga, sem registro"),
    ("backend None de proposito, como quant_int8 --no-convrot grava",
     {"backend": None}, ["backend_sem_registro"],
     "isencao declarada; avisa, mas a mensagem tem de dizer que a chave EXISTE"),

    # --- sem camadas quantizadas: a checagem nao se aplica
    ("arquivo sem camada quantizada nenhuma",
     {"backend": EAGER}, [], "checar_backend sai antes; nao ha o que registrar"),
]


def main() -> int:
    falhas = 0
    total = 0
    print(f"{'caso':66s} {'esperado':22s} {'obtido':22s} ok")
    for nome, sidecar, esperado, _porque in CASOS:
        kw = {"camadas": 0} if "sem camada quantizada" in nome else {}
        obtido = codigos(sidecar, **kw)
        ok = obtido == esperado
        total += 1
        falhas += not ok
        print(f"{nome[:66]:66s} {str(esperado)[:22]:22s} {str(obtido)[:22]:22s} "
              f"{'OK' if ok else 'FALHOU'}")

    # A mensagem tambem e testada, porque o defeito original era a MENSAGEM mentindo sobre o arquivo.
    print()
    msg_none = achados({"backend": None})[0].mensagem
    ok_none = "SEM a chave" not in msg_none
    print(f"{'mensagem de backend=None nao afirma que a chave falta':66s} "
          f"{'':22s} {'':22s} {'OK' if ok_none else 'FALHOU'}")
    if not ok_none:
        print(f"    mensagem: {msg_none}")
    total += 1
    falhas += not ok_none

    msg_ausente = achados({"quantization": "x"})[0].mensagem
    ok_ausente = "SEM a chave" in msg_ausente
    print(f"{'mensagem de chave ausente DIZ que a chave falta':66s} "
          f"{'':22s} {'':22s} {'OK' if ok_ausente else 'FALHOU'}")
    total += 1
    falhas += not ok_ausente

    print(f"\n{total - falhas}/{total} passaram")
    print("\nNAO COBERTO: isto testa `checar_backend` com Checkpoints sintetizados a mao. Nao le")
    print("nenhum arquivo real, nao confere se os escritores continuam gravando as formas que os")
    print("casos assumem, e nao diz nada sobre o backend que roda HOJE ao carregar -- para isso e a")
    print("camada 2, `tools/avaliar_despacho.py`.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
