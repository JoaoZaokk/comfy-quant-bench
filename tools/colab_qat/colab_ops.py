"""Operações repetidas das células do Colab, num lugar só. Sobe para `/content/qat/colab_ops.py`.

    exige(caminhos)              recusa (SystemExit) se faltar arquivo, listando todos
    snapshot_klein()             baixa o pipeline do klein (apache-2.0, sem token) -> (raiz, professor)
    lanca(cmd, log, cwd, probe)  processo em background (setsid nohup) + `/content/qat/probe.json` opcional
    para(padroes, espera_s)      SIGTERM, ESPERA sair (o QAT grava checkpoint ao fim do passo; o antigo matava
                                 com SIGKILL 8 s depois e perdia até 25 min), só então SIGKILL nos que sobram
    limpa_braco_sem_ckpt(d)      braço que morreu antes do 1o checkpoint: apaga só journal/melhor/status (os
                                 shards do professor e o ref_sens ficam -- o antigo `rmtree` apagava tudo)

Localmente: `python_embeded\\python.exe -s tools/colab_qat/colab_ops.py monta` regrava os probes gerados a
partir de `probe_job.py` e o `arquivos_qat.txt` (origem:destino de `ARQUIVOS_QAT`) que o
`.scratch/sobe_qat_comum.sh` usa para subir o QAT; o teste confere que os dois estão em dia.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

Q = Path("/content/qat")
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
PADROES_KLEIN = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*", "transformer/*"]

# Arquivos que a VM precisa para o QAT/fila/gerador (destino relativo a /content/qat). FONTE ÚNICA: as células
# conferem com `exige` e os scripts de subida leem `arquivos_qat.txt`, gerado daqui por `monta`.
ARQUIVOS_QAT = ["qat_ternario_klein.py", "qat_klein/__init__.py", "qat_klein/config.py", "qat_klein/modulos.py",
                "qat_klein/professor.py", "qat_klein/estado.py", "qat_klein/hf_sync.py", "qat_klein/avalia.py",
                "qat_klein/exporta.py", "qat_klein/status.py", "qat_klein/treino.py", "qat_klein/util.py",
                "lowbit_canon.py", "ajusta_denso_diffusers.py", "colab_ops.py", "probe_job.py", "lowbit_kernel.py"]
# origem (relativa à raiz da bancada) quando não é `tools/<destino>`
ORIGEM = {"colab_ops.py": "tools/colab_qat/colab_ops.py", "probe_job.py": "tools/colab_qat/probe_job.py",
          "lowbit_kernel.py": "custom_nodes/comfy-lowbit-loader/kernel.py"}
MANIFESTO = "arquivos_qat.txt"


def manifesto() -> list[tuple[str, str]]:
    """[(origem na bancada, destino em /content/qat)] de `ARQUIVOS_QAT`."""
    return [(ORIGEM.get(d, f"tools/{d}"), d) for d in ARQUIVOS_QAT]


def texto_manifesto() -> str:
    cab = ("# GERADO de colab_ops.ARQUIVOS_QAT por `colab_ops.py monta` -- nao editar a mao.\n"
           "# origem (relativa a raiz da bancada):destino (relativo a /content/qat); lido por .scratch/sobe_qat_comum.sh\n")
    return cab + "".join(f"{o}:{d}\n" for o, d in manifesto())


def exige(caminhos) -> None:
    faltam = [str(p) for p in caminhos if not Path(p).exists()]
    if faltam:
        for p in faltam:
            print(f"FALTA {p}")
        raise SystemExit(1)


def snapshot_klein(local_dir: str = "/content/klein4b") -> tuple[str, Path]:
    from huggingface_hub import snapshot_download
    raiz = snapshot_download(KLEIN, token=False, local_dir=local_dir, allow_patterns=PADROES_KLEIN)
    prof = Path(raiz) / "transformer" / "diffusion_pytorch_model.safetensors"
    print("snapshot", raiz, "professor", prof.stat().st_size)
    return raiz, prof


def lanca(cmd: list, log: Path, cwd: Path = Q, probe: dict | None = None) -> int:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    if probe is not None:
        (Q / "probe.json").write_text(json.dumps(probe))
    print("CMD", " ".join(map(str, cmd)))
    with open(log, "ab") as f:
        p = subprocess.Popen(["setsid", "nohup", *map(str, cmd)], stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd),
                             start_new_session=True)
    print(f"LANCADO pid={p.pid} log={log}")
    return p.pid


def vivos(padroes) -> str:
    out = []
    for pad in padroes:
        r = subprocess.run(["pgrep", "-af", f"[{pad[0]}]{pad[1:]}"], capture_output=True, text=True, check=False)
        out += [ln for ln in r.stdout.splitlines() if ln.strip()]
    return "\n".join(sorted(set(out)))


def para(padroes, espera_s: int = 600) -> None:
    for pad in padroes:
        subprocess.run(["pkill", "-TERM", "-f", pad], check=False)
    t0 = time.time()
    while vivos(padroes) and time.time() - t0 < espera_s:
        time.sleep(5)
    resto = vivos(padroes)
    if resto:
        print(f"ainda vivos apos {espera_s} s de SIGTERM; SIGKILL:\n{resto}")
        for pad in padroes:
            subprocess.run(["pkill", "-KILL", "-f", pad], check=False)
        time.sleep(5)
    else:
        print(f"parados com SIGTERM em {time.time() - t0:.0f} s")
    print("vivos:", vivos(padroes) or "nenhum")


def limpa_braco_sem_ckpt(d: Path) -> list[str]:
    d = Path(d)
    apagados = []
    if d.is_dir() and not (d / "ckpt" / "ultimo.pt").is_file():
        for nome in ("journal.jsonl", "melhor.json", "status.json"):
            if (d / nome).is_file():
                (d / nome).unlink()
                apagados.append(nome)
    return apagados


# ------------------------------------------------------------------ probes gerados (roda em casa)

PROBES = {
    "probe_qat.py": {"config": "/content/qat/config.json", "log": "qat.log", "processo": "qat_ternario_klein.py"},
    "probe_ds.py": {"dir": "/content/ds", "log": "ds.log", "processo": "gera_dataset_klein.py"},
    "../colab_qat_qwen21/probe_qat_qwen21.py": {"config": "/content/qatq/config.json", "log": "qat.log",
                                                "processo": "qat_qwen21_blocos.py", "corta_em": "carregando",
                                                "conta_glob": "blocos/bloco_*.pt", "conta_nome": "blocos_prontos"},
}


def texto_probe(job: dict) -> str:
    molde = (Path(__file__).with_name("probe_job.py")).read_text(encoding="utf-8")
    if molde.count("\nJOB = None\n") != 1:
        raise ValueError("probe_job.py sem a linha `JOB = None`")
    cab = "# GERADO de probe_job.py por `colab_ops.py monta_probes` -- editar o molde, nao este arquivo.\n"
    return cab + molde.replace("\nJOB = None\n", f"\nJOB = {json.dumps(job, sort_keys=True)}\n")


def monta_probes() -> None:
    aqui = Path(__file__).parent
    for nome, job in PROBES.items():
        (aqui / nome).write_bytes(texto_probe(job).encode("utf-8"))  # LF, tambem no Windows
        print("gravado", (aqui / nome).resolve())


def monta_manifesto() -> None:
    arq = Path(__file__).with_name(MANIFESTO)
    arq.write_bytes(texto_manifesto().encode("utf-8"))  # LF: lido pelo bash no WSL
    print("gravado", arq.resolve())


if __name__ == "__main__" and sys.argv[1:] in (["monta"], ["monta_probes"]):
    monta_probes()
    monta_manifesto()
