"""O patcher do Qwen-Image 2.1 Nunchaku recusa LoRA/patches de peso em vez de aceitar e ignorar (CPU, sem modelo).

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s custom_nodes/comfy-qwen21-nunchaku/testes/test_qwen21_patcher.py

Como script pelo mesmo motivo do teste do comfy-void-stage-tools: o `args.cpu` tem de ligar antes do
`comfy.model_management` ser importado pelo `__init__.py` do nó.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

NO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NO.parents[1] / "ComfyUI"))
from comfy.cli_args import args

args.cpu = True
assert not torch.cuda.is_available(), "rodar com CUDA_VISIBLE_DEVICES=-1"
spec = importlib.util.spec_from_file_location("qwen21_nunchaku", NO / "__init__.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def _patcher():
    m = torch.nn.Module()
    m.lin = torch.nn.Linear(4, 4)
    m.diffusion_model = m.lin  # o detach() do __del__ do patcher move `diffusion_model`
    return N.Qwen21NunchakuPatcher(m, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def test_recusa_lora():
    p = _patcher()
    with pytest.raises(RuntimeError, match="LoRA"):
        p.add_patches({"lin.weight": (torch.zeros(4, 4),)}, 1.0)
    assert p.add_patches({}) == []
    with pytest.raises(RuntimeError):
        p.add_weight_wrapper("lin.weight", lambda w: w)


def test_clone_mantem_a_classe():
    assert type(_patcher().clone()) is N.Qwen21NunchakuPatcher


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--import-mode=importlib", "-p", "no:cacheprovider"]))
