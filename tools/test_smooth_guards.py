"""As guardas de SELECAO do quant_w4a4_smooth.py, exercitadas como subprocesso.

DE ONDE VEIO
------------
Na janela de GPU de 2026-09-01, que fechava a migracao do ticket 08, o `smooth` era o unico dos
sete escritores sem par no disco -- nunca escreveu nada que ainda esteja aqui, entao nao ha
sha256 contra o que comparar. A aceitacao possivel era exercitar seus pontos de chamada ao
nucleo, e para isso escreveu-se um teste de recusas com um CASO DE CONTROLE: argumentos validos
tem de PASSAR da guarda e morrer depois, provando que as recusas vieram da guarda e nao de um
erro qualquer anterior a ela.

O controle falhou, e foi ele que achou os dois defeitos que este arquivo agora tranca:

    guarda vazia   Apontado para um Wan 2.1, o `smooth` imprimia `Layers: 0   quantized: 0` e
                   saia com rc=0. `LAYER_RE` nao casa nada, `selected` e `norm_keys` saem vazias,
                   e a checagem de `missing` -- que compara duas listas vazias -- aprova. Um
                   `--dry-run`, que e exatamente o que se roda ANTES de gastar horas, respondia
                   SUCESSO para uma conversao sem nada a converter. Os outros quatro escritores
                   ja recusavam (`quant_w4a4.py:386`, `quant_w4a8.py:248`, `quant_int8.py:139`,
                   `quant_mixed.py:581`); so este nao.

    guarda dtype   Apontado para o gemma_3_12B_it_heretic_fp8_e4m3fn, passava por TODA a
                   `refuse_unsafe`, passava pelo preflight de backend, rodava os seis prompts de
                   calibragem ate o fim (96 normas, ~3 min de 3090) e so entao morria com
                   `KeyError: 'F8_E4M3'` la dentro de `quant_w4a8.read_tensor` -- rastro que
                   aponta para outro arquivo, nao para a fonte que a pessoa escolheu. A
                   informacao para recusar estava no cabecalho o tempo todo.

O CASO 7 E O QUE FAZ OS OUTROS SEIS VALEREM
-------------------------------------------
Os casos 1-6 afirmam que o conversor RECUSA. Um conversor quebrado que morresse em toda
invocacao passaria nos seis. O caso 7 exige que uma fonte Gemma-3 BF16 legitima ATRAVESSE as
duas guardas novas e chegue ao relatorio -- e o unico caso que pode falhar por excesso de zelo.
Foi assim que os defeitos apareceram; e o caso que nao pode ser removido.

O CASO 6 MEDE TEMPO, DE PROPOSITO
---------------------------------
Nao basta recusar o fp8: tem de recusar ANTES de calibrar, que era o custo real do defeito. O
limite de 60 s e frouxo contra os ~3 min que a calibragem sozinha levava na 3090, largo o
bastante para nao piscar em maquina carregada e apertado o bastante para pegar a regressao de
mover a checagem para depois da calibragem.

NAO COBERTO POR ESTE ARQUIVO
----------------------------
- `conv.guard()` e `conv.commit()` do smooth (:295-296) NAO sao alcancados aqui: ficam depois de
  `calibrate()`, que exige um Gemma W4A4 em `text_encoders`. Medido em 2026-09-01 com busca
  recursiva `-Force` nos dois roots: nao existe nenhum nesta maquina, nem o BF16 de 12B que
  seria a fonte. **O caminho de ESCRITA do smooth continua sem ter rodado depois da migracao**,
  e nenhuma coluna de OK aqui muda isso.
- Nenhum byte e escrito por este arquivo. Ele testa recusas, nao conversao.
- Nada aqui diz que a matematica do SmoothQuant esta certa.
- Exige CUDA presente: `main()` do smooth checa `torch.cuda.is_available()` antes de tudo, entao
  numa maquina sem placa TODO caso vira "recusa" e os casos 1-6 passariam pelo motivo errado. O
  caso 7 e quem detecta isso, porque falharia.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PY = RAIZ / "python_embeded" / "python.exe"
SMOOTH = RAIZ / "tools" / "quant_w4a4_smooth.py"
MODELOS = RAIZ / "ComfyUI" / "models" / "diffusion_models"
ENCODERS = RAIZ / "ComfyUI" / "models" / "text_encoders"

# Fontes reais. Um caso cuja fonte sumiu vira PULADO, nunca OK -- ausencia nao e aprovacao.
SEM_CAMADAS = MODELOS / "wan2.1_vace_1.3B_fp16.safetensors"
JA_QUANTIZADO = MODELOS / "zimage-v2-w4a4.safetensors"
SAIDA_EXISTENTE = MODELOS / "wan21-vace-13b-misto005.safetensors"
DTYPE_ILEGIVEL = ENCODERS / "gemma_3_12B_it_heretic_fp8_e4m3fn.safetensors"
GEMMA3_BF16 = ENCODERS / "gemma-3-1b-it-heretic-extreme-uncensored-abliterated.safetensors"

LIMITE_RECUSA_RAPIDA = 60.0


def roda(args: list[str]) -> tuple[int, str, float]:
    inicio = time.perf_counter()
    p = subprocess.run([str(PY), "-s", str(SMOOTH)] + args,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr), time.perf_counter() - inicio


def main() -> int:
    trab = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / ".scratch" / "smooth_guards"
    trab.mkdir(parents=True, exist_ok=True)
    parcial_alvo = trab / "parcial.safetensors"
    (trab / "parcial.safetensors.partial").write_bytes(b"restos de uma corrida morta")

    casos = [
        ("1. saida == fonte", [SEM_CAMADAS],
         ["--source", str(SEM_CAMADAS), "--output", str(SEM_CAMADAS), "--calibrate-with", "x"],
         "recusa", "overwrite the source"),
        ("2. saida ja existe", [SEM_CAMADAS, SAIDA_EXISTENTE],
         ["--source", str(SEM_CAMADAS), "--output", str(SAIDA_EXISTENTE), "--calibrate-with", "x"],
         "recusa", "existing output"),
        ("3. fonte ja quantizada", [JA_QUANTIZADO],
         ["--source", str(JA_QUANTIZADO), "--output", str(trab / "a.safetensors"),
          "--calibrate-with", "x"],
         "recusa", "already has quantization metadata"),
        ("4. parcial rancoso no caminho", [SEM_CAMADAS],
         ["--source", str(SEM_CAMADAS), "--output", str(parcial_alvo), "--calibrate-with", "x"],
         "recusa", "stale partial"),
        ("5. fonte sem `model.layers.N.`", [SEM_CAMADAS],
         ["--source", str(SEM_CAMADAS), "--output", str(trab / "b.safetensors"),
          "--calibrate-with", "x", "--dry-run"],
         "recusa", "no `model.layers.N.` tensors"),
        ("6. dtype que ele nao le, e RAPIDO", [DTYPE_ILEGIVEL],
         ["--source", str(DTYPE_ILEGIVEL), "--output", str(trab / "c.safetensors"),
          "--calibrate-with", "gemma_3_12B_it_heretic_w4a8.safetensors"],
         "recusa_rapida", "dtypes this converter cannot read"),
        ("7. CONTROLE: Gemma-3 BF16 atravessa as duas guardas", [GEMMA3_BF16],
         ["--source", str(GEMMA3_BF16), "--output", str(trab / "d.safetensors"),
          "--calibrate-with", "x", "--convrot-groupsize", "64", "--dry-run"],
         "passa", "quantized: 182"),
    ]

    falhas = pulados = 0
    controle_rodou = False
    for nome, exige, args, esperado, marca in casos:
        ausentes = [p.name for p in exige if not p.is_file()]
        if ausentes:
            print(f"[PULADO] {nome}  -- falta {', '.join(ausentes)}")
            pulados += 1
            continue
        rc, saida, segundos = roda(args)
        if esperado == "passa":
            controle_rodou = True
            ok = rc == 0 and marca in saida
        else:
            ok = rc != 0 and marca in saida
            if esperado == "recusa_rapida":
                ok = ok and segundos < LIMITE_RECUSA_RAPIDA
        falhas += 0 if ok else 1
        detalhe = next((ln.strip() for ln in saida.splitlines() if marca in ln), "")
        if not detalhe:
            detalhe = next((ln.strip() for ln in saida.splitlines() if ln.strip()), "(sem saida)")
        print(f"[{'OK  ' if ok else 'FALHA'}] {nome}  ({segundos:.1f}s, rc={rc})")
        print(f"         {detalhe[:140]}")

    (trab / "parcial.safetensors.partial").unlink(missing_ok=True)

    if not controle_rodou:
        print("\nFALHA ESTRUTURAL: o caso de CONTROLE nao rodou. Sem ele os casos de recusa nao")
        print("  distinguem uma guarda que funciona de um conversor que morre em toda invocacao.")
        falhas += 1

    total = len(casos)
    print(f"\n{total - falhas - pulados}/{total} casos, {falhas} falharam, {pulados} pulados")
    print("\nNAO COBERTO POR ESTA EXECUCAO:")
    print("  - `conv.guard()` e `conv.commit()` do smooth NAO sao alcancados: ficam depois de")
    print("    `calibrate()`, que exige um Gemma W4A4 em text_encoders. Nao existe nesta")
    print("    maquina (busca recursiva -Force, dois roots, 2026-09-01). O caminho de ESCRITA")
    print("    do smooth continua sem ter rodado depois da migracao do ticket 08.")
    print("  - Nenhum byte foi escrito aqui. Isto testa recusas, nao conversao.")
    print("  - Nada aqui diz que a matematica do SmoothQuant esta certa.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
