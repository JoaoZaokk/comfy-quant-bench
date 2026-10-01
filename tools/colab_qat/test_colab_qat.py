"""Testes locais (CPU, sem rede, sem Colab) do probe único, do colab_ops e do índice do dataset.

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s -m pytest tools/colab_qat/test_colab_qat.py -q --import-mode=importlib
"""
import importlib.util
import json
import py_compile
import sys
import time
from pathlib import Path

import pytest
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

import colab_ops
import probe_job


def test_probes_gerados_estao_em_dia():
    for nome, job in colab_ops.PROBES.items():
        assert (AQUI / nome).read_text(encoding="utf-8") == colab_ops.texto_probe(job), f"rodar monta_probes ({nome})"


@pytest.mark.parametrize("st,vivo,esperado", [
    ({"estado": "FIM"}, False, "DONE"),
    ({"estado": "PARADO", "motivo": "sinal 15"}, False, "DONE"),
    ({"estado": "FALHOU", "motivo": "x"}, True, "FAILED"),
    ({"estado": "RECUSADO"}, False, "FAILED"),
    ({"estado": "RODANDO", "fase": "treino", "heartbeat": 100.0}, True, "RUNNING"),
    ({"estado": "RODANDO", "fase": "treino", "heartbeat": 100.0}, False, "FAILED"),
])
def test_probe_decide_pelo_status(st, vivo, esperado):
    assert probe_job.decide(st, "Traceback no log velho", vivo, agora=200.0)[0] == esperado


@pytest.mark.parametrize("tail,vivo,esperado", [
    ("[10:00:00] passo 5\nFIM 10:00:01", False, "DONE"),
    ("Traceback (most recent call last)", True, "FAILED"),
    ("[10:00:00] passo 5", True, "RUNNING"),
    ("[10:00:00] passo 5", False, "FAILED"),
])
def test_probe_cai_no_log_sem_status(tail, vivo, esperado):
    assert probe_job.decide(None, tail, vivo)[0] == esperado


def test_probe_mede_pasta(tmp_path, monkeypatch):
    (tmp_path / "qat.log").write_text("[10:00:00] passo 1\n")
    (tmp_path / "journal.jsonl").write_text('{"passo": 1}\n')
    (tmp_path / "status.json").write_text(json.dumps({"estado": "RODANDO", "fase": "treino", "pid": 999999,
                                                      "heartbeat": time.time(), "passo": 1}))
    monkeypatch.setattr(probe_job, "processo_vivo", lambda padrao, pid=None: True)
    monkeypatch.setattr(probe_job.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "gpu"})())
    p = probe_job.mede({"dir": str(tmp_path), "log": "qat.log", "processo": "qat_ternario_klein.py"})
    assert p["status"] == "RUNNING" and p["job_status"]["passo"] == 1 and p["ultimo_journal"] == '{"passo": 1}'


def test_limpa_braco_so_journal_e_melhor(tmp_path):
    (tmp_path / "professor").mkdir()
    (tmp_path / "professor" / "p0000_s1.pt").write_bytes(b"x")
    for n in ("journal.jsonl", "melhor.json", "ref_sens.json"):
        (tmp_path / n).write_text("{}")
    assert sorted(colab_ops.limpa_braco_sem_ckpt(tmp_path)) == ["journal.jsonl", "melhor.json"]
    assert (tmp_path / "professor" / "p0000_s1.pt").is_file() and (tmp_path / "ref_sens.json").is_file()
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "ultimo.pt").write_bytes(b"x")
    (tmp_path / "journal.jsonl").write_text("{}")
    assert colab_ops.limpa_braco_sem_ckpt(tmp_path) == []  # com checkpoint a corrida retoma: nada sai


def test_arquivos_da_vm_compilam():
    for f in list(AQUI.glob("*.py")) + list((AQUI.parent / "colab_qat_qwen21").glob("*.py")):
        py_compile.compile(str(f), doraise=True)


# ------------------------------------------------------------------ índice do dataset

def _gera():
    spec = importlib.util.spec_from_file_location("gera_dataset_klein", AQUI / "gera_dataset_klein.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Api:
    def __init__(self, arquivos):
        self.arquivos = arquivos

    def list_repo_files(self, repo, repo_type=None):
        return list(self.arquivos)


def _baixa_de(tmp_path, conteudo):
    def baixa(repo, caminho, repo_type=None, force_download=False):
        from huggingface_hub.errors import EntryNotFoundError
        if caminho not in conteudo:
            raise EntryNotFoundError(caminho)
        if isinstance(conteudo[caminho], Exception):
            raise conteudo[caminho]
        p = tmp_path / caminho.replace("/", "_")
        p.write_text("".join(json.dumps(r) + "\n" for r in conteudo[caminho]))
        return str(p)
    return baixa


def test_metadados_por_vm_nao_reescrevem_os_outros(tmp_path):
    g = _gera()
    conteudo = {"metadata.jsonl": [{"chave": "a", "camadas": {}}],
                "metadata/vm2.jsonl": [{"chave": "b", "camadas": {}}],
                "metadata/vm1.jsonl": [{"chave": "c", "camadas": {}}]}
    meta = g.Metadados(_Api(conteudo), "u/ds", "vm1", _baixa_de(tmp_path, conteudo))
    meta.rele()
    assert set(meta.todos) == {"a", "b", "c"} and set(meta.meus) == {"c"}
    txt = meta.acrescenta([{"chave": "d", "camadas": {}}])
    assert [json.loads(ln)["chave"] for ln in txt.splitlines()] == ["c", "d"]  # só as linhas desta VM


def test_metadados_recusam_erro_que_nao_e_ausencia(tmp_path):
    g = _gera()
    conteudo = {"metadata.jsonl": ConnectionError("rede")}
    meta = g.Metadados(_Api(["metadata.jsonl"]), "u/ds", "vm1", _baixa_de(tmp_path, conteudo))
    with pytest.raises(SystemExit):
        meta.rele()
    vazio = g.Metadados(_Api(["metadata/vm9.jsonl"]), "u/ds", "vm1", _baixa_de(tmp_path, {}))
    vazio.rele()  # listado mas ausente (EntryNotFound): vazio, sem erro
    assert vazio.todos == {}


def test_sobra_local_entra_no_envio(tmp_path, monkeypatch):
    g = _gera()
    monkeypatch.setattr(g, "BASE", tmp_path)
    meta = g.Metadados(_Api([]), "u/ds", "vm1", _baixa_de(tmp_path, {}))
    ger = g.Gerador({"repo": "u/ds", "passos": 2, "size": 64}, None, meta, lambda m: None)
    d = tmp_path / "holdout_s1"
    d.mkdir()
    torch.save({"prompt": "um gato", "semente": 1, "passos": [0, 0], "estat": [{"l": [0, 1, 2, 1]}] * 2},
               d / "p0003_s1.pt")  # nome legado: o indice nao diz o prompt
    torch.save({"prompt": "uma casa", "semente": 1, "passos": [0, 0]}, d / "p0004_s1.pt")
    assert ger.sobras(d, "holdout") == 1
    c = ger.chave("um gato", 1)
    assert (d / f"{c}.pt").is_file() and ger.lote[0][0] == c and ger.lote[0][2]["camadas"]["l"]["maximo"] == 2
    assert (d / "_sem_estat" / "p0004_s1.pt").is_file()


# ------------------------------------------------------------------ subida para o Colab (manifesto + .sh)

RAIZ = AQUI.parents[1]
SCRATCH = RAIZ / ".scratch"


def _bash():
    """Bash do Git para Windows (ou do sistema no Linux); nunca o bash.exe do System32, que abre o WSL."""
    import shutil
    b = shutil.which("bash")
    if b is None or "system32" in b.lower() or "windowsapps" in b.lower():
        pytest.skip("sem bash do Git/sistema")
    return b


def test_manifesto_em_dia_e_arquivos_existem():
    assert (AQUI / colab_ops.MANIFESTO).read_text(encoding="utf-8") == colab_ops.texto_manifesto(), "rodar colab_ops.py monta"
    destinos = [d for _, d in colab_ops.manifesto()]
    assert destinos == colab_ops.ARQUIVOS_QAT and len(set(destinos)) == len(destinos)
    for origem, _ in colab_ops.manifesto():
        assert (RAIZ / origem).is_file(), origem
    assert ("custom_nodes/comfy-lowbit-loader/kernel.py", "lowbit_kernel.py") in colab_ops.manifesto()


def _scripts_de_subida():
    if not SCRATCH.is_dir():
        pytest.skip(".scratch ausente")
    return sorted(p for p in SCRATCH.glob("*.sh") if "sobe_qat" in p.read_text(encoding="utf-8", errors="replace"))


def test_scripts_de_subida_passam_bash_n_e_usam_o_manifesto():
    import subprocess
    b = _bash()
    scripts = _scripts_de_subida()
    assert len(scripts) >= 10  # sobe_qat_comum.sh + os 9 scripts de subida do QAT
    for p in scripts:
        r = subprocess.run([b, "-n", str(p)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, (p.name, r.stderr)
        txt = p.read_text(encoding="utf-8")
        # o codigo do QAT so' sobe pelo manifesto: nenhum upload avulso dele sobrou
        assert "tools/qat_ternario_klein.py" not in txt and "tools/ajusta_denso_diffusers.py" not in txt, p.name
        if p.name != "sobe_qat_comum.sh":
            assert "sobe_qat_comum.sh" in txt and "sobe_qat " in txt, p.name


def test_sobe_qat_seco_envia_exatamente_o_manifesto():
    """`colab` falso (echo): nada sai da maquina; confere cada upload que o script faria."""
    import subprocess
    b = _bash()
    if not (SCRATCH / "sobe_qat_comum.sh").is_file():
        pytest.skip(".scratch/sobe_qat_comum.sh ausente")
    raiz = RAIZ.as_posix()
    cmd = f'. "{raiz}/.scratch/sobe_qat_comum.sh"; C="echo COLAB"; W=/mnt/f/COMFY_PORTABLE; L="{raiz}"; sobe_qat sessao-x'
    r = subprocess.run([b, "-c", cmd], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    ups = [ln.split() for ln in r.stdout.splitlines() if ln.startswith("COLAB upload")]
    esperado = [["COLAB", "upload", "-s", "sessao-x", f"/mnt/f/COMFY_PORTABLE/{o}", f"/content/qat/{d}"]
                for o, d in colab_ops.manifesto()]
    assert ups == esperado
    assert "COLAB exec -s sessao-x --timeout 60 -f /mnt/f/COMFY_PORTABLE/tools/colab_qat/celula_pastas.py" in r.stdout
    assert f"sobe_qat: {len(esperado)} arquivos" in r.stdout
    # arquivo faltando: recusa antes de qualquer upload
    r2 = subprocess.run([b, "-c", cmd.replace(f'L="{raiz}"', 'L=/nao/existe')], capture_output=True, text=True, check=False)
    assert r2.returncode != 0 and "COLAB upload" not in r2.stdout and "FALTA" in r2.stderr
