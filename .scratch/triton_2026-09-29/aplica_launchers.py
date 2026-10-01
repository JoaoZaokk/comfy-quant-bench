"""Põe --enable-triton-backend nos launchers de GPU e repassa as flags aos workers do multigpu-orchestrator.

    python aplica_launchers.py [--aplicar]
Sem --aplicar só mostra o diff. Cópia dos originais em bat_antes/. Preserva o fim de linha de cada arquivo.
"""
import difflib
import pathlib
import shutil
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
ANTES = pathlib.Path(__file__).resolve().parent / "bat_antes"
FLAG = "--enable-triton-backend"
# launcher -> flags que o worker do orquestrador deve receber (None: o launcher já desliga o orquestrador)
LAUNCHERS = {
    "run_nvidia_gpu.bat": FLAG,
    "run_nvidia_gpu_8190.bat": f"--use-sage-attention {FLAG}",
    "run_nvidia_gpu_8190_flash.bat": f"--use-flash-attention {FLAG}",
    "run_nvidia_gpu_8190_loopback.bat": f"--use-sage-attention {FLAG}",
    "run_8190_limpo.bat": f"--use-sage-attention --reserve-vram 1.5 {FLAG}",
    "run_nvidia_gpu_fast_fp16_accumulation.bat": f"--fast fp16_accumulation {FLAG}",
    "advanced/run_nvidia_gpu_disable_api_nodes.bat": f"--disable-api-nodes {FLAG}",
    "run_nvidia_gpu_8190_dual_component.bat": None,
    "run_nvidia_gpu_8190_ultra_image.bat": None,
    "run_nvidia_gpu_8190_ultra_image - sem dynamic.bat": None,
    "run_nvidia_gpu_8190_ultra_video.bat": None,
    "run_nvidia_gpu_8190_void_cache_none.bat": None,
}
BLOCO_WORKER = [
    "REM multigpu-orchestrator: os workers que executam os prompts sobem com main.py sem -s e sem as flags",
    "REM desta linha de comando (so recebem COMFYUI_MGPU_WORKER_FLAGS). Medido em 2026-09-29.",
]


def edita(texto: str, flags_worker: str | None) -> str:
    nl = "\r\n" if "\r\n" in texto else "\n"
    linhas = texto.split(nl)
    saida = []
    tem_cvd = any(li.strip().lower().startswith("set cuda_visible_devices") for li in linhas)
    for linha in linhas:
        if FLAG in linha:
            raise SystemExit("flag já presente; nada a fazer neste arquivo")
        if flags_worker and not tem_cvd and "main.py" in linha:
            saida += BLOCO_WORKER + [f'set "COMFYUI_MGPU_WORKER_FLAGS={flags_worker}"', 'set "PYTHONNOUSERSITE=1"']
        if "main.py" in linha and "--windows-standalone-build" in linha:
            linha = linha.replace("--windows-standalone-build", f"--windows-standalone-build {FLAG}", 1)
        saida.append(linha)
        if linha.strip() == "--windows-standalone-build ^":
            recuo = linha[: len(linha) - len(linha.lstrip())]
            saida.append(f"{recuo}{FLAG} ^")
        if flags_worker and linha.strip().lower().startswith("set cuda_visible_devices"):
            saida += BLOCO_WORKER + [f'set "COMFYUI_MGPU_WORKER_FLAGS={flags_worker}"', 'set "PYTHONNOUSERSITE=1"']
    novo = nl.join(saida)
    if novo.count(FLAG) < 1:
        raise SystemExit("linha do main.py não encontrada")
    return novo


def main():
    aplicar = "--aplicar" in sys.argv
    for nome, flags in LAUNCHERS.items():
        p = RAIZ / nome
        texto = p.read_bytes().decode("utf-8")
        novo = edita(texto, flags)
        print("".join(difflib.unified_diff(texto.splitlines(True), novo.splitlines(True), nome, nome, n=0)))
        if aplicar:
            destino = ANTES / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            if not destino.exists():
                shutil.copy2(p, destino)
            p.write_bytes(novo.encode("utf-8"))


if __name__ == "__main__":
    main()
