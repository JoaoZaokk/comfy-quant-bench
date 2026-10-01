"""Testes de CPU do QAT com um transformer sintético minúsculo (sem diffusers, sem GPU, sem rede).

    CUDA_VISIBLE_DEVICES=-1 python_embeded\\python.exe -s -m pytest tools/qat_klein/test_qat_klein.py -q --import-mode=importlib
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
TOOLS = AQUI.parent
sys.path.insert(0, str(TOOLS))

import lowbit_canon

from qat_klein import estado as estado_mod
from qat_klein import hf_sync, professor
from qat_klein.treino import RC_PARADO, main

assert not torch.cuda.is_available(), "rodar com CUDA_VISIBLE_DEVICES=-1"

C, D, E = 32, 64, 32


# ------------------------------------------------------------------ referências congeladas (código antigo)

def _antigo_codigo_e_escala(w, grupo, niveis):
    n, k = w.shape
    g = w.float().reshape(n, k // grupo, grupo)
    if niveis == 1:
        d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
    else:
        d = (g.abs().amax(dim=2, keepdim=True) / niveis).clamp(min=1e-30)
    t = (g / d).clamp(-niveis, niveis).round()
    num = (g * t).sum(dim=2, keepdim=True)
    den = (t * t).sum(dim=2, keepdim=True)
    s = torch.where(den > 0, num / den, torch.zeros_like(num))
    return t.reshape(n, k).to(torch.int8), s


def _antigo_ternariza_constroi(w, grupo):
    n, k = w.shape
    g = w.float().reshape(n, k // grupo, grupo)
    d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
    t = (g / d).clamp(-1, 1).round()
    num = (g * t).sum(dim=2, keepdim=True)
    den = (t * t).sum(dim=2, keepdim=True)
    s = torch.where(den > 0, num / den, torch.zeros_like(num))
    return (t * s).reshape(n, k).to(w.dtype)


def _igual_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("niveis,grupo", [(1, 128), (1, 32), (7, 32)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantizador_identico_ao_antigo(niveis, grupo, dtype):
    w = (torch.randn(96, 256, generator=torch.Generator().manual_seed(niveis * 7 + grupo)) * 0.02).to(dtype)
    q = lowbit_canon.Quantizador(niveis, grupo)
    t, s = q.codigo_e_escala(w)
    t0, s0 = _antigo_codigo_e_escala(w, grupo, niveis)
    assert torch.equal(t, t0) and _igual_bits(s, s0)
    antigo = (t0.float().reshape(96, 256 // grupo, grupo) * s0).reshape(96, 256).to(dtype)
    assert _igual_bits(q.quantiza(w), antigo)
    if niveis == 1:
        assert _igual_bits(lowbit_canon.ternariza(w, grupo), _antigo_ternariza_constroi(w, grupo))


def test_pilhas_e_corpo():
    nomes = ["transformer_blocks.0.attn.to_q.weight", "transformer_blocks.1.attn.to_q.weight",
             "adaLN_modulation.1.weight", "single_blocks.3.linear1.weight"]
    assert lowbit_canon.pilhas_reais(nomes) == {"transformer_blocks"}
    ps = [(n, torch.zeros(4, 4)) for n in nomes] + [("transformer_blocks.0.norm.weight", torch.zeros(4))]
    assert lowbit_canon.corpo(ps) == {nomes[0], nomes[1]}


def test_rtn_simetrico_igual_ao_mistura():
    spec = importlib.util.spec_from_file_location("mistura_klein", TOOLS / "mistura_klein.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    w = torch.randn(16, 64, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    for bits in (2, 3, 4):
        raw = w.contiguous().view(torch.uint8).numpy().tobytes()
        esperado = torch.frombuffer(bytearray(mk.rtn4(raw, "BF16", [16, 64], 32, bits)), dtype=torch.bfloat16)
        got = lowbit_canon.rtn_simetrico(w, 32, bits).to(torch.bfloat16).reshape(-1)
        assert _igual_bits(got, esperado.reshape(-1))


@pytest.mark.parametrize("niveis,grupo", [(1, 128), (1, 32), (7, 32)])
def test_lowbit_desquantiza_bit_a_bit(niveis, grupo):
    q = lowbit_canon.Quantizador(niveis, grupo)
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(11)) * 0.03
    w[3, :grupo] = 0  # grupo todo zero: escala 0
    t, s = q.codigo_e_escala(w)
    ref = q.reconstroi(t, s, torch.float32).to(torch.bfloat16)
    pack = q.empacota(t, s)
    assert pack.bits == (2 if niveis == 1 else 4) and pack.qdata.dtype == torch.uint8
    n, zeros = lowbit_canon.confere_exato(ref, pack)
    assert n == 0, (n, zeros)


def _importa_formats_do_loader():
    import types
    here = TOOLS.parent / "custom_nodes" / "comfy-lowbit-loader"
    sys.path.insert(0, str(TOOLS.parent / "ComfyUI"))
    from comfy.cli_args import args
    args.cpu = True
    pkg = types.ModuleType("lb_teste")
    pkg.__path__ = [str(here)]
    sys.modules["lb_teste"] = pkg
    mods = {}
    for n in ("kernel", "layout", "formats"):
        spec = importlib.util.spec_from_file_location(f"lb_teste.{n}", here / f"{n}.py")
        mods[n] = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mods[n]
        spec.loader.exec_module(mods[n])
    return mods["formats"]


def test_contrato_igual_ao_loader():
    try:
        formats = _importa_formats_do_loader()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ComfyUI/loader nao importou em CPU: {type(e).__name__}")
    q = lowbit_canon.Quantizador(1, 32)
    t, s = q.codigo_e_escala(torch.randn(8, 64))
    p = q.empacota(t, s)
    lb = formats.LowBit(p.qdata, p.scale, p.zero, p.bits, p.group_size)
    a = lowbit_canon.state_dict_lowbit({"x": torch.ones(2)}, {"blk.0.l": p})
    b = formats.to_comfy_state_dict({"x": torch.ones(2)}, {"blk.0.l": lb})
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert formats.shape_params(p.qdata, p.scale, 64) == (2, 32)


# ------------------------------------------------------------------ modelo e shards sintéticos

class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(D, D)


class _Bloco(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = torch.nn.Linear(D, D, bias=False)


class Minusculo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = torch.nn.Linear(C, D)
        self.context_embedder = torch.nn.Linear(E, D)
        self.time_in = torch.nn.Linear(1, D)
        self.transformer_blocks = torch.nn.ModuleList([_Bloco(), _Bloco()])
        self.norm_out = torch.nn.LayerNorm(D)
        self.proj_out = torch.nn.Linear(D, C)

    def enable_gradient_checkpointing(self):
        pass

    def forward(self, hidden_states, encoder_hidden_states, timestep, return_dict=False):
        h = self.x_embedder(hidden_states) + self.context_embedder(encoder_hidden_states).mean(1, keepdim=True)
        h = h + self.time_in(timestep.reshape(-1, 1, 1).to(h.dtype))
        for b in self.transformer_blocks:
            h = h + b.ff(F.gelu(b.attn.to_q(h)))
        return (self.proj_out(self.norm_out(h)),)


def carrega(_cfg=None):
    g = torch.Generator().manual_seed(1234)
    m = Minusculo()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return m.to(torch.bfloat16)


def _shard(prof, prompt, semente, passos=2, legado=False):
    g = torch.Generator().manual_seed(int(professor.chave(prompt, semente, passos, 64)[:8], 16))
    comum = {"encoder_hidden_states": torch.randn(1, 5, E, generator=g).to(torch.bfloat16)}
    ps = []
    with torch.no_grad():
        for j in range(passos):
            kw = {"hidden_states": torch.randn(1, 6, C, generator=g).to(torch.bfloat16),
                  "timestep": torch.tensor([1.0 - j / passos])}
            ps.append({"kw": kw, "out": prof(**comum, **kw)[0].detach().clone()})
    sh = {"prompt": prompt, "semente": semente, "comum": comum, "passos": ps}
    if not legado:
        sh.update(passos_n=passos, size=64)
    return sh


def monta_pasta(tmp: Path, legado=False) -> tuple[Path, list[str]]:
    tmp.mkdir(parents=True, exist_ok=True)
    tr = [f"um gato numero {i}" for i in range(4)]
    ho = [f"uma casa numero {i}" for i in range(3)]
    (tmp / "treino.txt").write_text("\n".join(tr), encoding="utf-8")
    (tmp / "holdout.txt").write_text("\n".join(ho), encoding="utf-8")
    prof = carrega()
    d = tmp / "run"
    for lista, sub in ((tr, "professor"), (ho, "professor_holdout")):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(lista):
            nome = f"p{i:04d}_s1.pt" if legado else f"{professor.chave(p, 1, 2, 64)}.pt"
            torch.save(_shard(prof, p, 1, legado=legado), d / sub / nome)
    return d, [str(tmp / "treino.txt"), str(tmp / "holdout.txt")]


def args_de(d: Path, prompts, *extra, max_passos=6):
    return ["--raiz", str(d.parent), "--professor", "x", "--prompts", prompts[0], "--prompts-holdout", prompts[1],
            "--sementes", "1", "--passos", "2", "--size", "64", "--dir", str(d), "--otim", "adam-fp32",
            "--lr", "3e-3", "--max-passos", str(max_passos), "--grupo", "32", "--log-cada", "1",
            "--holdout-cada", "2", "--sem-grad-ckpt", "--device", "cpu", "--sens-passos", "0,1", *extra]


def _perdas(d: Path) -> dict:
    out = {}
    for ln in (d / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if "perda" in r:
            out[r["passo"]] = r["perda"]
    return out


def _mestre(d: Path) -> dict:
    return torch.load(d / "ckpt" / "ultimo.pt", map_location="cpu", weights_only=False)["mestre"]


# ------------------------------------------------------------------ treino, retomada, exportação

def test_passo_de_treino_e_codigos_ternarios(tmp_path):
    d, pr = monta_pasta(tmp_path)
    assert main(args_de(d, pr, max_passos=2), carrega=carrega) == 0
    st = json.loads((d / "status.json").read_text(encoding="utf-8"))
    assert st["estado"] == "FIM" and st["rc"] == 0
    perdas = _perdas(d)
    assert set(perdas) == {1, 2} and all(math.isfinite(v) and v < 1e3 for v in perdas.values())
    from safetensors.torch import load_file
    sd = load_file(str(d / "aluno_ternario_diffusers.safetensors"))
    w = sd["transformer_blocks.0.attn.to_q.weight"].float().reshape(D, D // 32, 32)
    for g in w.reshape(-1, 32):  # cada grupo: valores em {-s, 0, s}
        assert len({abs(float(x)) for x in g} - {0.0}) <= 1


def test_retomada_reproduz_a_trajetoria(tmp_path):
    a, pr = monta_pasta(tmp_path / "a")
    b, pr_b = monta_pasta(tmp_path / "b")
    assert main(args_de(a, pr, "--exporta", "ambos"), carrega=carrega) == 0
    assert main(args_de(b, pr_b, "--exporta", "ambos", "--para-no-passo", "3"), carrega=carrega) == RC_PARADO
    st = json.loads((b / "status.json").read_text(encoding="utf-8"))
    assert st["estado"] == "PARADO" and st["rc"] == RC_PARADO
    est = torch.load(b / "ckpt" / "ultimo.pt", map_location="cpu", weights_only=False)
    assert est["passo"] == 3 and "estado" in est and {"mestre", "otim", "rng_py", "rng_torch"} <= set(est)
    assert main(args_de(b, pr_b, "--exporta", "ambos"), carrega=carrega) == 0
    assert _perdas(a) == _perdas(b)
    ma, mb = _mestre(a), _mestre(b)
    assert ma.keys() == mb.keys() and all(_igual_bits(ma[k], mb[k]) for k in ma)
    from safetensors import safe_open
    for nome in ("aluno_ternario_diffusers.safetensors", "aluno_ternario_diffusers_lowbit.safetensors"):
        # bytes do arquivo nao: o safetensors grava o __metadata__ em ordem de HashMap (varia por execucao)
        with safe_open(str(a / nome), "pt") as fa, safe_open(str(b / nome), "pt") as fb:
            assert fa.metadata() == fb.metadata() and list(fa.keys()) == list(fb.keys())
            assert all(_igual_bits(fa.get_tensor(k), fb.get_tensor(k)) for k in fa.keys()), nome  # noqa: SIM118 -- safe_open nao itera
    assert json.loads((a / "melhor.json").read_text()) == json.loads((b / "melhor.json").read_text())
    # o lowbit desquantiza bit a bit no desempacotado
    from safetensors.torch import load_file
    ds, lb = load_file(str(a / "aluno_ternario_diffusers.safetensors")), \
        load_file(str(a / "aluno_ternario_diffusers_lowbit.safetensors"))
    k = lowbit_canon.kernel()
    for nome in ("transformer_blocks.0.attn.to_q", "transformer_blocks.1.ff"):
        got = k.dequantize_torch(lb[nome + ".weight"], lb[nome + ".weight_scale"], lb[nome + ".weight_zeros"], 2, 32,
                                 torch.bfloat16)
        assert _igual_bits(got, ds[nome + ".weight"])
    assert _igual_bits(lb["x_embedder.weight"], ds["x_embedder.weight"])


def test_checkpoint_antigo_retoma(tmp_path, capsys):
    b, pr = monta_pasta(tmp_path)
    assert main(args_de(b, pr, "--para-no-passo", "3"), carrega=carrega) == RC_PARADO
    ck = b / "ckpt" / "ultimo.pt"
    est = torch.load(ck, map_location="cpu", weights_only=False)
    del est["estado"]  # exatamente as chaves que o código antigo gravava
    est["extra"] = {"otim": "adam-fp32"}
    torch.save(est, ck)
    capsys.readouterr()
    assert main(args_de(b, pr), carrega=carrega) == 0
    saida = capsys.readouterr().out
    assert "formato antigo" in saida and "RETOMADO do passo 3" in saida
    assert torch.load(ck, map_location="cpu", weights_only=False)["estado"]["passo"] == 6


def test_retomada_com_outra_config_recusa(tmp_path):
    b, pr = monta_pasta(tmp_path)
    assert main(args_de(b, pr, "--para-no-passo", "2"), carrega=carrega) == RC_PARADO
    with pytest.raises(SystemExit) as e:
        main(args_de(b, pr[:1] + pr[1:], "--lr", "1e-2"), carrega=carrega)
    assert "outra configuracao" in str(e.value.code)
    assert json.loads((b / "status.json").read_text())["estado"] == "RECUSADO"
    assert main(args_de(b, pr, "--lr", "1e-2", "--retoma-config-diferente"), carrega=carrega) == 0


def test_config_recusa_antes_de_carregar(tmp_path):
    d, pr = monta_pasta(tmp_path)

    def nao_carrega(_cfg):
        raise AssertionError("carregou o modelo")
    assert main(args_de(d, pr, "--so-escalas", "--l1-corpo", "1.0"), carrega=nao_carrega) == 2
    assert main(args_de(d, pr, "--cruzado-frac", "1.5"), carrega=nao_carrega) == 2


def test_so_escalas_roda_e_exporta_lowbit(tmp_path):
    d, pr = monta_pasta(tmp_path)
    assert main(args_de(d, pr, "--so-escalas", "--exporta", "lowbit", max_passos=2), carrega=carrega) == 0
    assert (d / "aluno_ternario_diffusers_lowbit.safetensors").is_file()
    assert not (d / "aluno_ternario_diffusers.safetensors").is_file()


def test_sem_globais_de_modo():
    import re
    atrib = re.compile(r"^\s*(NIVEIS|STE_LIGADO)\s*=|^\s*global\s+(?!_KERNEL\b)", re.MULTILINE)
    for arq in list((TOOLS / "qat_klein").glob("*.py")) + [TOOLS / "qat_ternario_klein.py", TOOLS / "lowbit_canon.py"]:
        if arq.name != "test_qat_klein.py":
            assert not atrib.search(arq.read_text(encoding="utf-8")), arq


# ------------------------------------------------------------------ shards

def test_shard_legado_treina_e_valida(tmp_path):
    d, pr = monta_pasta(tmp_path, legado=True)
    assert main(args_de(d, pr, max_passos=2), carrega=carrega) == 0
    arq = min((d / "professor").glob("*.pt"))
    sh = torch.load(arq, map_location="cpu", weights_only=False)
    professor.valida_shard(sh, arq, prompt=sh["prompt"], semente=1, passos=2, size=64)  # legado: size ausente
    with pytest.raises(ValueError):
        professor.valida_shard(sh, arq, prompt="outro prompt")
    with pytest.raises(ValueError):
        professor.valida_shard(sh, arq, passos=8)
    assert (d / "professor" / professor.INDICE).is_file()


def test_legado_com_indice_trocado_nao_pula_o_prompt(tmp_path):
    d, _ = monta_pasta(tmp_path, legado=True)

    class A:
        sementes = (1,)
    # lista nova: o prompt certo do indice 0 e' outro -> tem de faltar (antes o arquivo p0000_s1 o pulava)
    novos = ["prompt que nao existe"] + [f"um gato numero {i}" for i in range(1, 4)]
    falt = professor.faltantes(A, novos, d / "professor")
    assert falt == [(0, "prompt que nao existe", 1)]


def test_vazamento_pelos_shards_recusa(tmp_path):
    d, _ = monta_pasta(tmp_path)
    alvo = next((d / "professor").glob("*.pt"))
    shutil.copy(alvo, d / "professor_holdout" / alvo.name)
    with pytest.raises(SystemExit) as e:
        professor.recusa_vazamento(d / "professor", d / "professor_holdout", (1,))
    assert "RECUSADO" in str(e.value.code)


# ------------------------------------------------------------------ HF (API falsa, sem rede)

class ApiFalsa:
    def __init__(self):
        self.commits = []

    def create_repo(self, *a, **k):
        pass

    def upload_file(self, **k):
        pass

    def delete_file(self, *a, **k):
        pass

    def create_commit(self, **k):
        self.commits.append(k)

    def super_squash_history(self, **k):
        pass


def test_hf_nao_recomeca_do_zero_por_erro_transitorio(tmp_path):
    from huggingface_hub.errors import EntryNotFoundError
    hf = hf_sync.EmpurraHF("u/r", api=ApiFalsa())
    hf._baixa = lambda nome, local_dir=None: (_ for _ in ()).throw(ConnectionError("rede caiu"))
    with pytest.raises(SystemExit) as e:
        hf.baixa_se_faltar(tmp_path / "ckpt" / "ultimo.pt")
    assert "recusando comecar do zero" in str(e.value.code)

    def ausente(nome, local_dir=None):
        raise EntryNotFoundError("sem arquivo")
    hf._baixa = ausente
    hf.baixa_se_faltar(tmp_path / "ckpt" / "ultimo.pt")  # repo sem checkpoint: segue do zero


def test_hf_nunca_sobe_passo_menor(tmp_path):
    api = ApiFalsa()
    hf = hf_sync.EmpurraHF("u/r", api=api)
    pj = tmp_path / "passo.json"
    pj.write_text(json.dumps({"passo": 10}))
    hf._baixa = lambda nome, local_dir=None: str(pj)
    ck = tmp_path / "ultimo.pt"
    ck.write_bytes(b"x")
    hf.envia(ck, 5)
    hf.espera()
    assert api.commits == []
    hf.envia(ck, 12)
    hf.espera()
    ops = [o.path_in_repo for o in api.commits[0]["operations"]]
    assert ops == ["ckpt/ultimo.pt", "ckpt/passo.json"]


def test_parada_por_passo():
    p = estado_mod.Parada(no_passo=3)
    assert not p.confere(2) and p.confere(3) and p.pedida


def test_entrada_reexporta_a_api_antiga():
    spec = importlib.util.spec_from_file_location("qat_entrada", TOOLS / "qat_ternario_klein.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    w = torch.randn(8, 64, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    t0, s0 = _antigo_codigo_e_escala(w, 32, 1)
    assert _igual_bits(q.ternariza(w, 32), (t0.float().reshape(8, 2, 32) * s0).reshape(8, 64).to(torch.bfloat16))
    assert q.pilhas_reais(["a.0.w", "a.1.w"]) == {"a"} and q.BLOCO.match("a.0.w")
    assert callable(q.grava_professor) and callable(q.le_prompts) and callable(q.log)


def test_ref_sens_antigo_reusado_e_outro_remedido(tmp_path, capsys):
    d, pr = monta_pasta(tmp_path)
    (d / "ref_sens.json").write_text(json.dumps({"d_ref": 0.5, "pares": 3, "passos": [0, 1]}))  # formato antigo
    assert main(args_de(d, pr, max_passos=1), carrega=carrega) == 0
    assert "ref_sens.json antigo" in capsys.readouterr().out
    assert json.loads((d / "ref_sens.json").read_text())["d_ref"] == 0.5  # reusado, nao regravado
    d2, pr2 = monta_pasta(tmp_path / "b")
    (d2 / "ref_sens.json").write_text(json.dumps({"d_ref": 0.5, "pares": 9, "passos": [0, 1]}))  # outros pares
    assert main(args_de(d2, pr2, max_passos=1), carrega=carrega) == 0
    r = json.loads((d2 / "ref_sens.json").read_text())
    assert r["d_ref"] != 0.5 and r["pares"] == 3 and "impressao" in r
