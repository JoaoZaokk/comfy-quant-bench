"""`avaliar_despacho.py` exige um lock VIVO antes de mandar checkpoints para a GPU (revisao
2026-09-29) e grava os controles da execucao em cada laudo.

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s tools/test_avaliar_despacho_lock.py

`avaliar_despacho.LOCK_PATH` aponta para um arquivo temporario; F:/GPU_BENCH.lock nao e tocado.
Nenhum probe roda: o unico caminho de `main()` exercitado e a recusa, que sai antes do subprocesso.

NAO COBERTO: o lote com lock vivo (carregaria modelos na GPU).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import avaliar_despacho as ad  # noqa: E402


def _com_lock(conteudo: str | None, fn):
    with tempfile.TemporaryDirectory() as td:
        salvo = ad.LOCK_PATH
        ad.LOCK_PATH = Path(td) / "GPU_BENCH.lock"
        if conteudo is not None:
            ad.LOCK_PATH.write_text(conteudo, encoding="ascii")
        try:
            return fn(Path(td))
        finally:
            ad.LOCK_PATH = salvo


def _vivo(dono="comfy:avaliar_despacho", pid=None, hb=None):
    return (f"dono={dono}\npid={pid or os.getpid()}\ndesde=x\n"
            f"hb={int(hb if hb is not None else time.time())}\nowner_kind=ps\n")


def test_sem_lock_recusa():
    ok, motivo, _ = _com_lock(None, lambda td: ad.lock_vivo())
    assert not ok and "sem lock" in motivo, motivo


def test_lock_vivo_passa():
    ok, motivo, _ = _com_lock(_vivo(), lambda td: ad.lock_vivo())
    assert ok, motivo


def test_heartbeat_velho_recusa():
    ok, motivo, _ = _com_lock(_vivo(hb=time.time() - 3600), lambda td: ad.lock_vivo())
    assert not ok and "nao esta vivo" in motivo, motivo


def test_pid_morto_recusa():
    ok, motivo, _ = _com_lock(_vivo(pid=999999), lambda td: ad.lock_vivo())
    assert not ok and "nao esta vivo" in motivo, motivo


def test_dono_errado_recusa():
    ok, motivo, _ = _com_lock(_vivo(dono="outro"), lambda td: ad.lock_vivo("comfy:avaliar_despacho"))
    assert not ok and "outro dono" in motivo, motivo


def test_controles_tem_versoes_e_lock():
    c = ad.controles_da_execucao(1, "owner=x")
    assert c["device"] == 1 and c["lock"] == "owner=x"
    assert set(c["versoes"]) == {"torch", "comfy-kitchen", "triton-windows"}
    assert "comfyui_head" in c and "quando" in c


def test_main_recusa_sem_lock_antes_de_qualquer_probe():
    def caso(td):
        laudos = td / "laudos"
        laudos.mkdir()
        (laudos / "x.json").write_text(json.dumps({
            "arquivo": str(td / "diffusion_models" / "falso_w4a4.safetensors"),
            "fatos": {"camadas_quantizadas": 3, "formatos": {}}}), encoding="utf-8")
        chamou = []
        salvo_probe = ad.rodar_probe
        ad.rodar_probe = lambda *a, **k: chamou.append(a) or (None, "nao devia rodar")
        salvo_argv = sys.argv
        sys.argv = ["avaliar_despacho.py", "--laudos", str(laudos), "--saida", str(td / "saida")]
        try:
            rc = ad.main()
        finally:
            sys.argv = salvo_argv
            ad.rodar_probe = salvo_probe
        return rc, chamou, list((td / "saida").glob("*.json"))
    rc, chamou, gravados = _com_lock(None, caso)
    assert rc == 3 and not chamou and not gravados, (rc, chamou, gravados)


if __name__ == "__main__":
    falhas = 0
    for nome, fn in sorted(globals().items()):
        if not nome.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {nome}")
        except AssertionError as exc:
            falhas += 1
            print(f"FAIL  {nome}: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"FAIL  {nome}: unexpected {exc!r}")
    print("\nNAO COBERTO: lote com lock vivo; F:/GPU_BENCH.lock nunca e tocado.")
    raise SystemExit(1 if falhas else 0)
