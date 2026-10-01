"""Escrita ponta a ponta por formato, com safetensors sintetico e quantizador FALSO em CPU.

    python_embeded\\python.exe -s tools\\test_formats_e2e.py

O que prende (revisao de 2026-09-29):

  - cada formato de `_formats` planeja o header SO das formas, quantiza dentro do laco de escrita e
    o arquivo sai com os tensores, dtypes e formas que `tensors()` prometeu -- e passa na
    `validate_structure` e na comparacao byte a byte de preservados do `verify_w4a4` (a saida do
    escritor e SUBCONJUNTO do que o verificador aceita);
  - um produtor que devolve outra forma/dtype levanta, sem deixar saida, sidecar nem `.partial`;
  - o sidecar entra no commit atomico: falha no meio nao deixa nenhum dos dois, sidecar ou saida
    criados por outro processo depois da recusa inicial sao recusados e nada deles e tocado;
  - `_ram_guard.check_commit` recusa com numero quando falta commit;
  - `quant_w4a4 --dry-run` recusa saida existente (antes o dry-run voltava antes da recusa);
  - `_profiles`: as regex dos conversores sao as de arquivo + `.weight`, e a autodeteccao de LTX;
  - `quant_mixed.decide`: a decisao por camada e o orcamento, por tabela, sem GPU.

O quantizador falso devolve tensores deterministicos da forma e do dtype reais do comfy-kitchen
(conferidos contra o backend eager em 2026-09-29). Nenhum kernel roda aqui.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402
import _profiles as P  # noqa: E402
import _ram_guard as R  # noqa: E402

ROWS, COLS = 16, 256


# ---------------------------------------------------------------- fakes e fixtures

class FakeCK:
    """As duas funcoes do comfy-kitchen que `_formats` chama com `ck`, com as formas reais."""

    @staticmethod
    def quantize_convrot_w4a4_weight(weight, convrot_groupsize=256, quant_group_size=64,
                                     stochastic_rounding=0):
        n, k = weight.shape
        q = (weight[:, : k // 2] * 100).round().clamp(-128, 127).to(torch.int8)
        return q, weight.abs().amax(dim=1).float() + convrot_groupsize

    @staticmethod
    def quantize_w4a8_int8_weight(weight, group_size, convrot_groupsize, symmetric, scale_dtype,
                                  codebook, codebook_tensor, stochastic_rounding):
        n, k = weight.shape
        q = (weight[:, : k // 2] * 50).round().clamp(-128, 127).to(torch.int8)
        s_rel = weight.abs().reshape(n, k // group_size, group_size).amax(-1).to(scale_dtype)
        cb = torch.linspace(-1, 1, 16) if codebook else None
        return q, s_rel, weight.abs().amax(dim=1).float(), None, cb


def instala_fake_int8() -> None:
    """`Int8Tensorwise.quantize` importa o comfy-kitchen ele mesmo; aqui ele recebe um falso."""
    def rowwise(w):
        s = w.abs().amax(dim=1, keepdim=True).float() / 127
        return (w / s.clamp(min=1e-12)).round().clamp(-127, 127).to(torch.int8), s

    quant = types.ModuleType("comfy_kitchen.backends.eager.quantization")
    quant.quantize_int8_rowwise = rowwise
    registry = types.SimpleNamespace(
        get_implementation=lambda name, kwargs=None: (lambda w, group_size: rowwise(w)))
    ck = types.ModuleType("comfy_kitchen")
    ck.registry = registry
    for nome, mod in (("comfy_kitchen", ck), ("comfy_kitchen.backends", types.ModuleType("b")),
                      ("comfy_kitchen.backends.eager", types.ModuleType("e")),
                      ("comfy_kitchen.backends.eager.quantization", quant)):
        sys.modules[nome] = mod
    sys.modules["comfy_kitchen.backends"].eager = sys.modules["comfy_kitchen.backends.eager"]
    sys.modules["comfy_kitchen.backends.eager"].quantization = quant


def escreve_fonte(path: Path, nome_extra: str = "") -> dict:
    g = torch.Generator().manual_seed(7)
    tensores = {
        "model.layers.0.self_attn.q_proj.weight": (torch.randn(ROWS, COLS, generator=g) * 0.05).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": (torch.randn(ROWS, COLS, generator=g) * 0.05).to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.rand(COLS, generator=g).to(torch.bfloat16),
        "model.norm.weight": torch.rand(COLS, generator=g),
    }
    entradas_header, blobs, off = {}, [], 0
    for k, t in tensores.items():
        raw = C.as_bytes(t).tobytes()
        entradas_header[k] = {"dtype": C.header_dtype(t), "shape": list(t.shape),
                              "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps({"__metadata__": {"format": "pt", "extra": nome_extra}, **entradas_header},
                    separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with path.open("wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for b in blobs:
            f.write(b)
    return tensores


def converte(fonte: Path, saida: Path, fmt, manifesto=None) -> C.Conversion:
    conv = C.Conversion(fonte, saida, saida.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    selected = P.select_layers(conv.header, "gemma", fmt.accepts)
    formats = {n: fmt for n in selected}
    meta = F.quant_metadata(conv.metadata, F.layer_configs(formats), fmt.name)
    with conv.tensors() as t:
        entradas = F.plan_model(conv.header, formats, lambda n: t[n].float(), FakeCK())
        conv.commit(entradas, meta, sidecar=manifesto or {"formato": fmt.name})
    return conv


FORMATOS = [F.ConvrotW4A4(64), F.ConvrotW4A4(256), F.AsymW4A8(16, 256, True),
            F.AsymW4A8(32, 256, False), F.Int8Tensorwise(True, 256), F.Int8Tensorwise(False, 256),
            F.AwqW4A16(32)]


# ---------------------------------------------------------------- testes

def test_cada_formato_escreve_o_que_planejou_e_passa_no_verify(tmp: Path) -> None:
    from verify_w4a4 import validate_preserved_bytes, validate_structure

    fonte = tmp / "fonte.safetensors"
    original = escreve_fonte(fonte)
    for i, fmt in enumerate(FORMATOS):
        saida = tmp / f"saida_{i}.safetensors"
        conv = converte(fonte, saida, fmt)
        header, meta = C.read_header(saida)
        layers = json.loads(meta["_quantization_metadata"])["layers"]
        assert set(layers) == {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"}, layers
        for nome in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight"):
            for chave, dtype, forma in fmt.tensors(nome, [ROWS, COLS]):
                assert header[chave]["dtype"] == dtype and header[chave]["shape"] == forma, (fmt, chave)
        assert saida.stat().st_size == conv.output_size, fmt
        lido = C.LazyTensors(saida)
        for k in ("model.layers.0.input_layernorm.weight", "model.norm.weight"):
            assert torch.equal(lido[k].view(torch.uint8) if lido[k].dtype == torch.bfloat16 else lido[k],
                               original[k].view(torch.uint8) if original[k].dtype == torch.bfloat16 else original[k]), k
        side = json.loads(saida.with_suffix(".quant.json").read_text(encoding="utf-8"))
        assert side == {"formato": fmt.name}, side
        src_header, _ = C.read_header(fonte)
        erros = validate_structure(header, meta, src_header)
        assert not erros, (fmt, erros)
        assert not validate_preserved_bytes(saida, fonte, header, src_header, layers), fmt
        assert not saida.with_suffix(".safetensors.partial").exists()
        assert not saida.with_suffix(".quant.json.partial").exists()


def test_produtor_com_forma_errada_levanta_e_nao_deixa_nada(tmp: Path) -> None:
    fonte = tmp / "fonte.safetensors"
    escreve_fonte(fonte)
    saida = tmp / "errada.safetensors"
    conv = C.Conversion(fonte, saida, saida.with_suffix(".quant.json"))
    entradas = [C.plan_lazy("x", "F32", [2, 8], 64, lambda: torch.zeros(4, 4))]
    try:
        conv.commit(entradas, None, sidecar={"a": 1})
        raise AssertionError("commit aceitou forma diferente da planejada")
    except RuntimeError as exc:
        assert "[2, 8]" in str(exc) and "[4, 4]" in str(exc), str(exc)
    assert not saida.exists() and not saida.with_suffix(".quant.json").exists()
    assert not conv.partial.exists() and not conv.sidecar_partial.exists()
    # mesmo numero de bytes, dtype diferente: tambem levanta
    entradas = [C.plan_lazy("x", "F32", [4, 4], 64, lambda: torch.zeros(4, 8, dtype=torch.int16))]
    try:
        conv.commit(entradas)
        raise AssertionError("commit aceitou dtype diferente do planejado")
    except RuntimeError:
        pass
    assert not saida.exists()


def test_falha_no_meio_nao_deixa_saida_nem_sidecar(tmp: Path) -> None:
    fonte = tmp / "fonte.safetensors"
    escreve_fonte(fonte)
    saida = tmp / "meio.safetensors"
    conv = C.Conversion(fonte, saida, saida.with_suffix(".quant.json"))

    def explode():
        raise ValueError("kernel caiu")

    entradas = [C.plan_copy("model.norm.weight", conv.header["model.norm.weight"]),
                C.plan_lazy("y", "F32", [2], 8, explode)]
    try:
        conv.commit(entradas, None, sidecar={"a": 1})
        raise AssertionError("commit engoliu a falha do produtor")
    except ValueError:
        pass
    for p in (saida, conv.sidecar, conv.partial, conv.sidecar_partial):
        assert not p.exists(), p


def test_sidecar_criado_depois_da_recusa_e_recusado_e_preservado(tmp: Path) -> None:
    fonte = tmp / "fonte.safetensors"
    escreve_fonte(fonte)
    saida = tmp / "corrida.safetensors"
    conv = C.Conversion(fonte, saida, saida.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    conv.sidecar.write_text("de outro processo", encoding="utf-8")
    try:
        conv.commit([C.plan_copy("model.norm.weight", conv.header["model.norm.weight"])], None,
                    sidecar={"meu": True})
        raise AssertionError("commit sobrescreveu um sidecar existente")
    except SystemExit as exc:
        assert "existing sidecar" in str(exc), str(exc)
    assert conv.sidecar.read_text(encoding="utf-8") == "de outro processo"
    assert not saida.exists() and not conv.partial.exists() and not conv.sidecar_partial.exists()


def test_saida_criada_depois_da_recusa_e_recusada_e_o_sidecar_sai_junto(tmp: Path) -> None:
    fonte = tmp / "fonte.safetensors"
    escreve_fonte(fonte)
    saida = tmp / "corrida2.safetensors"
    conv = C.Conversion(fonte, saida, saida.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    saida.write_bytes(b"de outro processo")

    try:
        conv.commit([C.plan_copy("model.norm.weight", conv.header["model.norm.weight"])], None,
                    sidecar={"meu": True})
        raise AssertionError("commit sobrescreveu uma saida existente")
    except SystemExit as exc:
        assert "existing output" in str(exc), str(exc)
    assert saida.read_bytes() == b"de outro processo"
    assert not conv.sidecar.exists(), "o sidecar desta conversao ficou para tras sem o modelo"
    assert not conv.partial.exists() and not conv.sidecar_partial.exists()


def test_write_sidecar_avulso_e_exclusivo(tmp: Path) -> None:
    alvo = tmp / "avulso.quant.json"
    C.write_json_exclusive(alvo, {"a": 1})
    assert json.loads(alvo.read_text(encoding="utf-8")) == {"a": 1}
    try:
        C.write_json_exclusive(alvo, {"a": 2})
        raise AssertionError("write_json_exclusive sobrescreveu")
    except SystemExit:
        pass
    assert json.loads(alvo.read_text(encoding="utf-8")) == {"a": 1}
    assert not alvo.with_suffix(".json.partial").exists()


def test_check_commit_recusa_com_numero(tmp: Path) -> None:  # noqa: ARG001
    recusa = R.check_commit(10 * R.GIB, headroom_gib=2.0, label="x", free_bytes=5 * R.GIB)
    assert recusa and "12.00 GiB" in recusa and "5.00 GiB" in recusa, recusa
    assert R.check_commit(1 * R.GIB, headroom_gib=2.0, label="x", free_bytes=100 * R.GIB) is None
    livre = R.commit_free_bytes()
    if sys.platform == "win32":
        assert isinstance(livre, int) and livre > 0, livre
    else:
        assert livre is None


def test_dry_run_do_w4a4_recusa_saida_existente(tmp: Path) -> None:
    fonte = tmp / "gemma_teste.safetensors"
    escreve_fonte(fonte)
    existente = tmp / "ja_existe.safetensors"
    existente.write_bytes(b"x")
    py = HERE.parent / "python_embeded" / "python.exe"
    p = subprocess.run([str(py), "-s", str(HERE / "quant_w4a4.py"), "--input", str(fonte),
                        "--output", str(existente), "--profile", "gemma", "--dry-run"],
                       capture_output=True, text=True, env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"})
    assert p.returncode != 0 and "existing output" in (p.stdout + p.stderr), (p.returncode, p.stderr[-300:])


def test_perfis_dos_conversores_sao_os_de_arquivo_com_weight(tmp: Path) -> None:  # noqa: ARG001
    assert tuple(P.PROFILE_PATTERNS) == P.CONVERTER_PROFILES
    for nome, pat in P.FILE_PATTERNS.items():
        assert P.WEIGHT_PATTERNS[nome].pattern == pat.pattern[:-1] + r"\.weight$", nome
    ltx = ["transformer_blocks.0.audio_to_video_attn.to_q.weight",
           "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"]
    assert P.detect_profile(Path("qualquer.safetensors"), ltx) == "ltx_2_5"
    assert P.detect_profile(Path("x.safetensors"), ["model." + n for n in ltx]
                            + ["model.diffusion_model." + n for n in ltx]) == "ltx_2_5"
    # a segunda rede corta mesmo quando a allowlist deixa passar
    header = {"model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8, 256]}}
    assert P.select_layers(header, "gemma") == ["model.layers.0.self_attn.q_proj.weight"]
    P.EXCLUSIONS["gemma"] += ("q_proj",)
    try:
        assert P.select_layers(header, "gemma") == []
    finally:
        P.EXCLUSIONS["gemma"] = P.EXCLUSIONS["gemma"][:-1]


def _linhas(n: int, sem: set[int], e4, e8) -> dict:
    by = {}
    for i in range(n):
        if i in sem:
            by[f"l{i}"] = {"layer": f"l{i}", "calibrated": False}
        else:
            by[f"l{i}"] = {"layer": f"l{i}", "calibrated": True, "err_w4a4": e4(i), "err_w4a8": e8(i)}
    return by


def test_decide_do_mixed_por_tabela(tmp: Path) -> None:  # noqa: ARG001
    import quant_mixed as qm

    sel = [f"l{i}.weight" for i in range(6)]
    by = _linhas(6, {5}, lambda i: 0.1 * i, lambda i: 0.02 * i)
    base = dict(uncalibrated="w4a8", somente_w4a4=False, keep_bf16_error=None, promote_error=0.25,
                budget=1.0)
    d, notas = qm.decide(sel, by, **base)
    assert d == {"l0": "convrot_w4a4", "l1": "convrot_w4a4", "l2": "convrot_w4a4",
                 "l3": "asym_w4a8_int8", "l4": "asym_w4a8_int8", "l5": "asym_w4a8_int8"}, d
    assert notas == []
    d, _ = qm.decide(sel, by, **(base | {"keep_bf16_error": 0.07}))
    assert d["l4"] == "bf16" and d["l3"] == "asym_w4a8_int8", d
    d, _ = qm.decide(sel, by, **(base | {"somente_w4a4": True, "uncalibrated": "bf16"}))
    assert set(d.values()) == {"convrot_w4a4", "bf16"} and d["l5"] == "bf16", d
    try:
        qm.decide(sel, by, **(base | {"uncalibrated": "fail"}))
        raise AssertionError("--uncalibrated fail nao recusou")
    except SystemExit:
        pass
    # O defeito do `inf`: 170 camadas, 12 sem medicao, --budget 0.05. O corte nunca pode rebaixar
    # uma camada que nao foi medida.
    sel = [f"l{i}.weight" for i in range(170)]
    by = _linhas(170, set(range(12)), lambda i: 0.5, lambda i: 0.001 * i)
    d, notas = qm.decide(sel, by, **(base | {"budget": 0.05}))
    assert all(d[f"l{i}"] == "asym_w4a8_int8" for i in range(12)), "rebaixou camada sem medicao"
    assert sum(v == "asym_w4a8_int8" for v in d.values()) == 12
    # 12 sem medicao contra um orcamento de int(170 * 0.05) = 8: excede, e diz que excede.
    assert any("capped" in n for n in notas) and any("exceed the budget of 8" in n for n in notas), notas


# ---------------------------------------------------------------- runner

def main() -> int:
    instala_fake_int8()
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    falhas = 0
    with tempfile.TemporaryDirectory(prefix="formats_e2e_") as raw:
        for teste in testes:
            tmp = Path(raw) / teste.__name__
            tmp.mkdir()
            try:
                teste(tmp)
                print(f"PASS {teste.__name__}")
            except Exception:  # noqa: BLE001
                falhas += 1
                print(f"FAIL {teste.__name__}")
                traceback.print_exc()
            except SystemExit as exc:
                falhas += 1
                print(f"FAIL {teste.__name__}: SystemExit {exc}")
    print(f"\n{len(testes) - falhas}/{len(testes)} passaram")
    print("\nNAO COBERTO: nenhum kernel real roda aqui (quantizador falso, exceto o awq, que usa o gguf-py "
          "real em CPU). A equivalencia de BYTES com o codigo antigo, com o comfy-kitchen eager real, "
          "e a prova de .scratch/revisao_2026-09-29/conv/prova/; o backend CUDA so a janela de GPU prova. "
          "O smooth nao tem caso aqui: depende de calibragem (ComfyUI) e esta so na prova.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
