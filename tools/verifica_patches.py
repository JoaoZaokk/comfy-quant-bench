"""Confere se cada patch local em patches/ está aplicado onde deve estar.

    python_embeded/python.exe -s tools/verifica_patches.py [--base v0.37.4]

Para cada patches/*.patch com alvo conhecido roda `git apply --check -R` (aplicado = o reverso
aplica limpo). Os patches do ComfyUI também precisam aplicar sozinhos sobre a versão base limpa
(índice temporário, nada no checkout muda), que é o que permite regerá-los depois de atualizar.
`git apply` pula em silêncio arquivos fora do repositório onde roda; a saída `-v` é lida e um
"Skipped patch" conta como falha, para que um patch pulado não passe por aplicado.

Só lê: não aplica, não reverte, não escreve no checkout. Sai com 1 se algum patch com alvo
conhecido não estiver aplicado.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "patches"

# prefixo do arquivo -> (diretório onde rodar git apply, argumentos extras)
TARGETS = {
    "comfyui_": (ROOT / "ComfyUI", []),
    "comfy_kitchen_": (ROOT, ["-p1", "--directory=python_embeded/Lib/site-packages"]),
    "dasiwa_": (ROOT, ["--directory=ComfyUI/custom_nodes/ComfyUI-DaSiWa-Nodes"]),
    # clones de terceiros com patch local: git apply roda dentro do próprio clone
    "nunchaku_": (ROOT / "ComfyUI/custom_nodes/ComfyUI-nunchaku", []),
    "anomalous_model_browser_": (ROOT / "ComfyUI/custom_nodes/Anomalous_Model_Browser", []),
    "fishspeech_s2wrapper_": (ROOT / "ComfyUI/custom_nodes/ComfyUI-FishSpeechS2Wrapper", []),
    "ltx2_multigpu_": (ROOT / "ComfyUI/custom_nodes/ComfyUI-LTX2-MultiGPU", []),
    "ltxvideo_": (ROOT / "ComfyUI/custom_nodes/ComfyUI-LTXVideo", []),
    "multigpu_": (ROOT / "ComfyUI/custom_nodes/ComfyUI-MultiGPU", []),
    "diffueraser_": (ROOT / "ComfyUI/custom_nodes/ComfyUI_DiffuEraser", []),
}
# patches aplicados em outra máquina: só informados, conferir lá
REMOTE = {
    "zen_image_edit_": "Arc (ssh arc): ~/ComfyUI/custom_nodes/zen-image-edit-comfyui, conferir com `git diff | md5sum`",
}


def git_apply(cwd: Path, args: list[str], patch: Path, env: dict | None = None) -> tuple[bool, str]:
    proc = subprocess.run(["git", "apply", "--check", "-v", *args, str(patch)], cwd=cwd,
                          capture_output=True, text=True, env=env, check=False)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0 and "Skipped patch" not in output, output


def applies_alone_on_base(patch: Path, base: str) -> tuple[bool, str]:
    comfy = ROOT / "ComfyUI"
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        read = subprocess.run(["git", "read-tree", base], cwd=comfy, capture_output=True, text=True, env=env, check=False)
        if read.returncode != 0:
            return False, read.stderr.strip()
        return git_apply(comfy, ["--cached"], patch, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="v0.37.4", help="versão do ComfyUI sobre a qual os patches comfyui_* são gerados")
    args = parser.parse_args()

    failed = 0
    for patch in sorted(PATCHES.glob("*.patch")):
        target = next((t for prefix, t in TARGETS.items() if patch.name.startswith(prefix)), None)
        remote = next((r for prefix, r in REMOTE.items() if patch.name.startswith(prefix)), None)
        if remote is not None:
            print(f"REMOTO     {patch.name}: {remote}")
            continue
        if target is None:
            print(f"SEM ALVO   {patch.name}: prefixo sem alvo registrado em TARGETS; nao conferido")
            continue
        cwd, extra = target
        applied, output = git_apply(cwd, ["-R", *extra], patch)
        if applied:
            status = "APLICADO"
        else:
            failed += 1
            clean, _ = git_apply(cwd, extra, patch)
            status = "AUSENTE (aplica limpo)" if clean else "DIVERGENTE"
        line = f"{status:10} {patch.name}"
        if patch.name.startswith("comfyui_"):
            alone, alone_output = applies_alone_on_base(patch, args.base)
            if not alone:
                failed += 1
                line += f" | NAO aplica sozinho sobre {args.base}: {alone_output.splitlines()[-1] if alone_output else '?'}"
            else:
                line += f" | aplica sozinho sobre {args.base}"
        print(line)
        if not applied:
            print("           " + output.replace("\n", "\n           "))
    print(f"\n{failed} problema(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
