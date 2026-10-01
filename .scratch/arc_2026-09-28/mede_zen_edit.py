"""Pico de VRAM do encode do zen com 0, 1 e 2 imagens de referência (1024²), fora do servidor."""
import sys, time
sys.path[:0] = ["/home/arc/ComfyUI", "/home/arc/ComfyUI/custom_nodes/zen-image-edit-comfyui"]
sys.argv = [sys.argv[0]]
import torch
from PIL import Image
import zen_nodes
M = "/mnt/comfy-models/text_encoders"
a = zen_nodes.load_adapter("", "bf16", f"{M}/qwen3.5_0.8b", f"{M}/zen/adapter_v12.safetensors")
imgs = [Image.open("/home/arc/ComfyUI/input/edit_pescador.png").convert("RGB"), Image.open("/home/arc/ComfyUI/input/edit_frasco.png").convert("RGB")]
for n in (0, 1, 2, 2):
    a.load(); torch.xpu.synchronize(); base = torch.xpu.memory_allocated(); torch.xpu.reset_peak_memory_stats()
    t = time.time()
    a.encode("test <image1>", imgs[:n], [(1024, 1024)] * n)
    torch.xpu.synchronize()
    print(n, "imagens: pico extra", round((torch.xpu.max_memory_allocated() - base) / 2**30, 2), "GiB; pesos", round(base / 2**30, 2), "GiB;", round(time.time() - t, 1), "s", flush=True)
    a.offload()
