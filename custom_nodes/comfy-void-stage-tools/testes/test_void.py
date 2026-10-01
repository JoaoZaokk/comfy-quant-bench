"""Testes de CPU dos nos VOID (sem GPU: as funcoes CUDA sao substituidas por falsas).

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s custom_nodes/comfy-void-stage-tools/testes/test_void.py

Como script (ele chama o pytest): coletado direto, o pytest importaria o `__init__.py` do no -- e com ele o
`comfy.model_management`, que sonda a GPU -- antes de este arquivo ligar o `args.cpu`.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NO = Path(__file__).resolve().parents[1]
for raiz in (NO.parents[1] / "ComfyUI", NO.parents[1]):
    if (raiz / "comfy" / "model_management.py").is_file():
        sys.path.insert(0, str(raiz))
        break
from comfy.cli_args import args

args.cpu = True
assert not torch.cuda.is_available(), "rodar com CUDA_VISIBLE_DEVICES=-1"

spec = importlib.util.spec_from_file_location("void_stage_tools", NO / "__init__.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def _cuda_falso(monkeypatch, reservado, nvml=None):
    chamadas = []
    props = [types.SimpleNamespace(name=f"falsa{i}", total_memory=24 << 30, uuid=f"u{i}") for i in range(2)]
    monkeypatch.setattr(V.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(V.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(V.torch.cuda, "get_device_properties", lambda i: props[i])
    monkeypatch.setattr(V.torch.cuda, "memory_allocated", lambda i: reservado[i] // 2)
    monkeypatch.setattr(V.torch.cuda, "memory_reserved", lambda i: reservado[i])
    monkeypatch.setattr(V.torch.cuda, "mem_get_info", lambda i: chamadas.append(i) or (1 << 30, 24 << 30))
    monkeypatch.setattr(V, "_nvml", lambda: nvml)
    return chamadas


def test_sem_nvml_so_consulta_placa_ja_inicializada(monkeypatch):
    chamadas = _cuda_falso(monkeypatch, reservado=[1 << 30, 0])
    g = V._gpus()
    assert chamadas == [0]  # a placa 1 (sem memoria deste processo) nao ganha contexto
    assert g[0]["fonte"] == "mem_get_info" and g[1]["fonte"] == "indisponivel" and g[1]["global_free"] is None


def test_com_nvml_nao_chama_mem_get_info(monkeypatch):
    info = types.SimpleNamespace(free=5, total=10)
    nvml = types.SimpleNamespace(nvmlDeviceGetHandleByUUID=lambda u: u, nvmlDeviceGetMemoryInfo=lambda h: info)
    chamadas = _cuda_falso(monkeypatch, reservado=[1 << 30, 0], nvml=nvml)
    g = V._gpus()
    assert chamadas == [] and all(r["fonte"] == "nvml" and r["global_free"] == 5 for r in g)


def test_caminho_de_auditoria(monkeypatch, tmp_path):
    monkeypatch.delenv("VOID_AUDIT_DIR", raising=False)
    assert V._audit_dir("").name == "void_audit"
    monkeypatch.setenv("VOID_AUDIT_DIR", str(tmp_path / "env"))
    assert V._audit_dir("") == tmp_path / "env"
    assert V._audit_dir(str(tmp_path / "no")) == tmp_path / "no"
    assert V._audit_dir("off") is None


def test_tres_nos_uma_implementacao(monkeypatch):
    vistos = []
    monkeypatch.setattr(V, "_unload", lambda m, s, a="": vistos.append((m, s, a)))
    casos = [(V.VoidUnloadModelImage, "image", "IMAGE", "model"), (V.VoidUnloadModelLatent, "latent", "LATENT", "model"),
             (V.VoidUnloadOpticalFlowLatent, "latent", "LATENT", "optical_flow")]
    for cls, passa, tipo, alvo in casos:
        req = cls.INPUT_TYPES()["required"]
        assert list(req) == [passa, alvo, "stage"] and cls.RETURN_TYPES == (tipo,) and cls.RETURN_NAMES == (passa,)
        assert cls().unload(stage="s", **{passa: "dado", alvo: "mp"}) == ("dado",)
    assert vistos == [("mp", "s", "")] * 3
    assert set(V.NODE_CLASS_MAPPINGS) >= {"VoidUnloadModelImage", "VoidUnloadModelLatent", "VoidUnloadOpticalFlowLatent"}


def test_unload_recusa_objeto_que_nao_e_patcher():
    with pytest.raises(TypeError):
        V._unload(object(), "x", "off")


if __name__ == "__main__":
    # O pytest importa o `__init__.py` do no ao coletar este arquivo; rodando por aqui o `args.cpu` ja esta
    # ligado (acima) quando o `comfy.model_management` for importado.
    raise SystemExit(pytest.main([__file__, "-q", "--import-mode=importlib", "-p", "no:cacheprovider"]))
