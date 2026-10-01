"""`bench_server.py` le e toma o lock da GPU pela API de `gpu_lock.py` (revisao 2026-09-29).

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s tools/test_bench_server_lock.py

Nada aqui toca F:/GPU_BENCH.lock: `bench_server.LOCK_PATH` aponta para um arquivo temporario antes
de cada caso. Nenhum conversor sobe: o unico caminho de `_converter` exercitado e a RECUSA por lock
ocupado, que devolve antes de criar subprocesso.

NAO COBERTO: a conversao real com o lock livre (subiria um conversor de horas), a pagina HTML.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import bench_server as bs  # noqa: E402
import gpu_lock  # noqa: E402

REAL = gpu_lock.LOCK_PATH


def _com_lock_temporario(fn):
    with tempfile.TemporaryDirectory() as td:
        salvo = bs.LOCK_PATH
        bs.LOCK_PATH = Path(td) / "GPU_BENCH.lock"
        try:
            return fn(bs.LOCK_PATH)
        finally:
            bs.LOCK_PATH = salvo


def test_caminho_padrao_e_o_do_protocolo():
    assert bs.LOCK_PATH == REAL == Path("F:/GPU_BENCH.lock"), bs.LOCK_PATH
    assert bs.LOCK_PATH != bs.RAIZ / "GPU_BENCH.lock"


def test_livre_sem_arquivo():
    assert _com_lock_temporario(lambda p: bs._lock())["livre"] is True


def test_ocupado_le_a_chave_dono():
    def caso(p):
        p.write_text("dono=cortiq:teste\npid=4242\ndesde=x\nhb=0\nowner_kind=ps\n", encoding="ascii")
        return bs._lock()
    r = _com_lock_temporario(caso)
    assert r["livre"] is False and r["dono"] == "cortiq:teste" and r["pid"] == "4242", r
    assert "cortiq:teste" in r["descricao"]


def test_toma_lock_recusa_quando_ocupado_e_nao_mexe_no_arquivo():
    def caso(p):
        p.write_text("dono=outro\npid=4242\nhb=0\n", encoding="ascii")
        trava, recusa = bs._toma_lock("w4a4")
        return trava, recusa, p.read_text(encoding="ascii")
    trava, recusa, conteudo = _com_lock_temporario(caso)
    assert trava is None and recusa["recusado"] and "outro" in recusa["porque"], recusa
    assert conteudo.startswith("dono=outro"), conteudo


def test_toma_lock_livre_grava_este_pid_e_solta():
    def caso(p):
        trava, recusa = bs._toma_lock("w4a4")
        assert recusa is None and trava is not None
        estado = gpu_lock.read_state(p)
        trava.__exit__(None, None, None)
        return estado, p.exists()
    estado, existe_depois = _com_lock_temporario(caso)
    assert estado["dono"] == "bench_server:w4a4" and estado["pid"] == str(os.getpid()), estado
    assert not existe_depois


def test_converter_real_recusa_com_lock_ocupado_sem_subir_nada():
    def caso(p):
        p.write_text("dono=medindo\npid=4242\nhb=0\n", encoding="ascii")
        salvo = bs.PERMITIR_ESCRITA
        bs.PERMITIR_ESCRITA = True
        try:
            antes = dict(bs.TRABALHOS)
            r = bs._converter({"subcomando": "w4a4", "flags": {}})
            return r, antes == bs.TRABALHOS
        finally:
            bs.PERMITIR_ESCRITA = salvo
    r, sem_trabalho_novo = _com_lock_temporario(caso)
    assert r.get("recusado") and "medindo" in r["porque"], r
    assert sem_trabalho_novo


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
    print("\nNAO COBERTO: conversao real com lock livre; F:/GPU_BENCH.lock nunca e tocado.")
    raise SystemExit(1 if falhas else 0)
