"""Fila de download da varredura de 2026-08-30. Um arquivo por vez, sem GPU.

POR QUE ESTES E NAO OUTROS. A varredura achou catorze checkpoints; esta fila tem cinco.
O corte foi feito por hardware e por pergunta:

  - NENHUM nvfp4. Medido em `nunchaku/utils.py:311-313` e confirmado pelo README do
    ConvRot oficial: NVFP4 exige Blackwell. Nesta bancada (sm86) e peso morto. Isso
    descarta 46 GiB de arquivos que parecem relevantes pelo nome.
  - O conjunto CASADO do joeygambino primeiro: w4a4, w4a8 e mix4x8 do MESMO LTX-2.5, pelo
    MESMO produtor. E o A/B/C da pergunta que esta bancada acabou de medir em Z-Image
    (ramo INT4 nativo contra ramo INT8), feito por outra pessoa e em outro modelo.
  - O text encoder junto, senao os DiT acima nao geram nada.
  - Um ConvRot misto do Abiray, que e o checkpoint de 791k downloads cujo metadado declara
    `w4a4_int4mm_layers: 0`. Baixado para ser CARREGADO, nao so lido: o header ja foi lido
    por Range, o que falta e ver o que o dispatch faz com ele.

Resumivel: `hf_parallel_get.py` grava um sidecar `.parts.json` por chunk, entao uma queda
custa no maximo os chunks em voo, e rodar de novo continua de onde parou -- inclusive
depois de reboot.

Nada aqui toca a GPU. Pode rodar junto com o treino do dono.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from hf_parallel_get import download  # noqa: E402

DIFF = ROOT / "ComfyUI" / "models" / "diffusion_models"
TE = ROOT / "ComfyUI" / "models" / "text_encoders"

FILA = [
    # (repo, arquivo, destino, GiB aproximado, por que)
    ("joeygambino/LTX-2.5-Quantized", "LTX25-distilled-DiT-comfy-w4a4.safetensors", DIFF,
     10.46, "W4A4 puro, o par direto do que esta bancada produz"),
    ("joeygambino/LTX-2.5-Quantized", "LTX25-distilled-DiT-comfy-w4a8.safetensors", DIFF,
     11.66, "W4A8 do mesmo modelo e produtor: o outro braco do A/B"),
    ("joeygambino/LTX-2.5-Quantized", "gemma4-12b-ltx25-comfy-w4a8.safetensors", TE,
     9.88, "text encoder; sem ele os DiT acima nao geram"),
    ("joeygambino/LTX-2.5-Quantized", "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors",
     DIFF, 12.86, "a mistura por camada decidida por error-removed-per-byte"),
    ("Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
     "MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors", DIFF,
     14.81, "791k downloads, declara w4a4_int4mm_layers: 0 -- carregar para conferir"),
]

total = sum(x[3] for x in FILA)
print(f"{len(FILA)} arquivos, ~{total:.1f} GiB no total. Um por vez, sem GPU.")
print(f"token HF: {'presente' if os.environ.get('HF_TOKEN') else 'AUSENTE'}\n", flush=True)

ok, falhou, pulados = [], [], []
for i, (repo, arquivo, destino, gib, motivo) in enumerate(FILA, 1):
    alvo = destino / Path(arquivo).name
    print(f"[{i}/{len(FILA)}] {repo}", flush=True)
    print(f"        {arquivo}  (~{gib:.2f} GiB)", flush=True)
    print(f"        motivo: {motivo}", flush=True)
    if alvo.exists() and not (alvo.with_suffix(alvo.suffix + ".parts.json")).exists():
        print(f"        JA EXISTE em {alvo}, pulando\n", flush=True)
        pulados.append(arquivo)
        continue
    t0 = time.perf_counter()
    resultado = {}
    try:
        rc = download(repo, arquivo, destino, outcome=resultado)
    except Exception as exc:  # noqa: BLE001
        print(f"        FALHOU: {type(exc).__name__}: {exc}\n", flush=True)
        falhou.append((arquivo, f"{type(exc).__name__}: {exc}"))
        continue
    el = time.perf_counter() - t0
    if rc == 0:
        v = resultado.get("verification", "sem digest oferecido pelo servidor")
        print(f"        pronto em {el / 60:.1f} min  ({gib * 1024 / el:.0f} MiB/s)  "
              f"verificacao: {v}\n", flush=True)
        ok.append((arquivo, resultado.get("verified")))
    else:
        print(f"        retornou {rc}\n", flush=True)
        falhou.append((arquivo, f"rc={rc}"))

print("=" * 70)
print(f"concluidos {len(ok)}   pulados {len(pulados)}   falharam {len(falhou)}")
for a, v in ok:
    print(f"  OK      {a}   verificado={v}")
for a in pulados:
    print(f"  PULADO  {a}")
for a, m in falhou:
    print(f"  FALHOU  {a}  :: {m}")
print()
print("NAO COBERTO: nada foi carregado nem executado -- isto so traz bytes para o disco.",
      "Um arquivo cujo servidor nao ofereceu digest fica com verificado=False e continua",
      "sem conferencia de conteudo. Rode de novo para retomar o que faltou.", file=sys.stderr)
