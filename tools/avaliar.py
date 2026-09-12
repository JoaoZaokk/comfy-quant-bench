"""Avaliador em lote de checkpoints quantizados. Camada 1: cabecalho, sidecar, analise.

O QUE ESTA FERRAMENTA FAZ E O QUE ELA RECUSA A FAZER
----------------------------------------------------
Ela **reprova**, **aponta onde olhar** e **preve antes de gastar GPU**. Ela nunca aprova, e
`APROVADO` nao existe no conjunto de vereditos por decisao, nao por falta de tempo.

O motivo esta medido nesta bancada e nao e uma cautela generica: **nenhum corte separa usavel de
inutilizavel**, em nenhum dos dois eixos que temos.

    erro por camada     0,1837 correta   contra   0,2147 destruida    (HunyuanVideo 1.5)
    divergencia latente 0,7173 boa       contra   0,8255 destruida

Quinze por cento de distancia nos dois casos. Um numero desses reprova sozinho quando esta do lado
errado da linha *daquele modelo*, e nao promete nada quando esta do lado certo. So uma renderizacao
olhada por alguem diz que um checkpoint presta.

PROVENIENCIA DE CADA CHECAGEM
-----------------------------
Toda checagem aqui nasceu de um erro que custou trabalho real nesta bancada, nao de uma auditoria
teorica. Cada uma declara, no proprio codigo, o que ela **nao** cobre (campo `cego`), e essas
declaracoes sao impressas em toda execucao -- passe ou falhe. Uma coluna de linhas verdes lida como
"verificado" por quem cola o resultado na proxima sessao; ver CLAUDE.md, "Say which one it was".

Esta camada **le**, nunca executa: nada aqui toca a GPU, carrega modelo ou roda kernel. Onde essa
limitacao morde, o laudo diz `SEM VEREDITO` naquele eixo em vez de calar.

A **camada 2 existe** desde 2026-09-01 e mora em `tools/avaliar_despacho.py`, tomando o lugar do
`--dispatch` que este texto prometia. Ela e um programa separado de proposito: adicionar GPU aqui
custaria o unico argumento desta camada, que e rodar 163 checkpoints em 0,68 s sem torch. Ela le a
SAIDA desta -- so visita quem aqui apareceu com camada quantizada -- carrega pelo caminho normal do
ComfyUI e conta forward quantizado contra `dequantize`. E o unico jeito de saber se o campo
`backend` do sidecar, que registra uma conversao passada, ainda descreve o que acontece hoje.

A **camada 3 existe** e mora em `tools/avaliar_referencia.py`: ela usa GPU e pergunta se o arquivo
NAO quantizado responde ao proprio condicionamento. E a guarda que teria abortado a rodada do Wan
na primeira imagem em vez da quarta. Rode-a antes de acreditar em qualquer numero que sai daqui,
porque todo numero desta camada e medido *contra* uma referencia que ela nao confere.

USO
---
    python.exe -s tools\\avaliar.py ComfyUI\\models\\diffusion_models --saida .scratch\\avaliacao
    python.exe -s tools\\avaliar.py um_arquivo.safetensors --json

Roda desacompanhado: escreve um `.json` por checkpoint e um `digest.md` ordenado por
"olha aqui primeiro".
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPROVA = "REPROVA"
OLHAR = "OLHAR"

VEREDITO_REPROVADO = "REPROVADO"
VEREDITO_OLHAR = "OLHAR"
VEREDITO_SEM = "SEM VEREDITO"


# ---------------------------------------------------------------------------------------------
# A linha por modelo
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Banda:
    """A faixa de erro por camada que UM modelo tolera. Nao ha faixa do formato.

    Medido 2026-08-31/09-01 em tres familias de arquitetura, uma linha por familia. A coluna
    `tolerado` e monotona no tamanho do modelo, o que e uma hipotese sobre tres pontos e nao uma
    lei. `nao_tolerado=None` significa medido-que-funciona ate `tolerado` e **nunca testado acima
    disso** -- e uma celula honestamente vazia, nao um teto alto.
    """

    familia: str
    parametros: str
    tolerado: float
    nao_tolerado: float | None
    nota: str = ""


BANDAS: dict[str, Banda] = {
    "wan_2_1": Banda("Wan 2.1 VACE", "1,3 B", 0.0546, 0.0793,
                     "o 0,1602 do W4A4 puro saiu destruido em tres sementes"),
    "zimage": Banda("Z-Image v2", "~6 B", 0.1421, 0.1848,
                    "medido em 2026-09-03 movendo o convrot_groupsize, unico eixo que ainda "
                    "andava: cg 64 (0,1421) rende imagem boa em 3 sementes e cg 16 (0,1848) "
                    "destroi. O 0,1241 do cg 256 continua o mais conservador que se sabe bom"),
    "hunyuan_video_15": Banda("HunyuanVideo 1.5", "~13 B", 0.1837, 0.2147,
                              "duas quebras independentes: 0,2147 e 0,2163 no capybara_v0.1"),
}

# `architecture` do sidecar e `profile` da analise usam nomes proprios; um mapa em vez de
# adivinhacao por substring, que casaria "zimage" dentro de qualquer coisa.
FAMILIA_POR_NOME: dict[str, str] = {
    "wan_2_1": "wan_2_1", "wan21": "wan_2_1", "wan_2_2": "wan_2_1",
    "zimage": "zimage", "z_image": "zimage",
    "hunyuan_video_15": "hunyuan_video_15", "hunyuan15": "hunyuan_video_15",
}


@dataclass(frozen=True)
class Formato:
    """O que UM formato exige no arquivo. Lido de arquivos reais desta bancada, nao de um spec.

    `empacota_int4` e o campo que separa as regras: so nesses o container I8 guarda duas colunas
    por byte e a largura precisa dividir o `convrot_groupsize`. Sem essa separacao as regras de
    int4 reprovaram um FP8 publico e valido -- ver `checar_forma`.

    `escala` diz que forma o `weight_scale` tem, porque ela varia entre formatos que de resto se
    parecem, e num deles varia dentro do proprio formato. Leia o aviso em `FORMAS_DE_ESCALA`
    antes de apertar qualquer regra daqui: apertar essa foi errado tres vezes seguidas.
    """

    exigidos: tuple[str, ...]
    empacota_int4: bool
    escala: str      # chave em FORMAS_DE_ESCALA


# Formato ausente daqui e desconhecido para esta camada, e ela diz isso em vez de inventar regra.
FORMATOS: dict[str, Formato] = {
    "convrot_w4a4":    Formato(("weight_scale",), True, "por_linha"),
    "asym_w4a8_int8":  Formato(("weight_s_rel", "weight_s_channel", "weight_codebook"),
                               True, "nenhuma"),
    "int8_tensorwise": Formato(("weight_scale",), False, "por_linha_ou_escalar"),
    "float8_e4m3fn":   Formato(("weight_scale",), False, "escalar"),
}

# AVISO, e o aviso e o conteudo: a regra de escala do `int8_tensorwise` foi escrita errada TRES
# vezes seguidas ao construir esta ferramenta, cada vez a partir do primeiro arquivo que eu tinha
# lido, e cada vez ela REPROVOU um arquivo bom:
#
#   1. `[linhas]`        vindo do ConvRot            -> acusou o Dasiwa WAN 2.2 inteiro
#   2. `[linhas, 1]`     vindo do Dasiwa             -> acusou `model.embed_tokens` do encoder
#                                                       MiniMax, que esta bancada ja mediu rodando
#                                                       kernel nativo em 15 de 15 saidas
#   3. `[linhas, 1]` se `per_row`, senao `[]`        -> acusou as 561 camadas do Gemma 4 E2B, que
#                                                       tem `[linhas, 1]` sem declarar `per_row`
#
# Os dois formatos existem no mundo real, de produtores diferentes, e nada no JSON da camada
# permite dizer qual e o certo para AQUELA camada. Entao esta camada **para de afirmar**: aceita
# os dois e declara no campo `cego` que nao pega escala torta nesse formato. Um quarto palpite
# seria o mesmo erro pela quarta vez.
FORMAS_DE_ESCALA = {
    "por_linha": lambda linhas, cfg: [[linhas]],  # noqa: ARG005
    "por_linha_ou_escalar": lambda linhas, cfg: [[linhas, 1], []],  # noqa: ARG005
    "escalar": lambda linhas, cfg: [[]],  # noqa: ARG005
    "nenhuma": lambda linhas, cfg: None,  # noqa: ARG005
}

BACKEND_NATIVO = "comfy_kitchen.backends.cuda"


# ---------------------------------------------------------------------------------------------
# Achados e laudo
# ---------------------------------------------------------------------------------------------

@dataclass
class Achado:
    nivel: str
    codigo: str
    mensagem: str
    evidencia: dict = field(default_factory=dict)

    def para_json(self) -> dict:
        return {"nivel": self.nivel, "codigo": self.codigo,
                "mensagem": self.mensagem, "evidencia": self.evidencia}


@dataclass
class Laudo:
    caminho: Path
    veredito: str
    achados: list[Achado]
    fatos: dict
    cego: list[str]
    erro_leitura: str | None = None

    @property
    def ordem(self) -> int:
        """Menor vem primeiro no digest: olha aqui primeiro."""
        if self.erro_leitura:
            return 0
        return {VEREDITO_REPROVADO: 1, VEREDITO_OLHAR: 2, VEREDITO_SEM: 3}[self.veredito]

    def para_json(self) -> dict:
        return {
            "arquivo": str(self.caminho),
            "veredito": self.veredito,
            "erro_leitura": self.erro_leitura,
            "achados": [a.para_json() for a in self.achados],
            "fatos": self.fatos,
            "nao_coberto": self.cego,
        }


# ---------------------------------------------------------------------------------------------
# O checkpoint lido uma vez
# ---------------------------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """Tudo que a camada 1 consegue ler, lido uma vez e passado a cada checagem.

    Nada aqui carrega peso: o cabecalho safetensors e um JSON no inicio do arquivo, e o sidecar e
    a analise sao arquivos de texto ao lado. Um checkpoint de 24 GiB custa alguns milissegundos.
    """

    caminho: Path
    tamanho: int
    tensores: dict[str, dict]
    metadata: dict[str, str]
    quant: dict          # `_quantization_metadata` decodificado, ou {}
    sidecar: dict        # `<stem>.quant.json`, ou {}
    candidatas: list[tuple[dict, Path]]   # analises da mesma fonte, a escolhida primeiro
    sobra_bytes: int     # bytes depois do fim do ultimo tensor
    dialeto: str         # de onde vieram as camadas quantizadas; ver `ler_dialeto_por_tensor`

    @property
    def analise(self) -> dict:
        return self.candidatas[0][0] if self.candidatas else {}

    @property
    def analise_origem(self) -> Path | None:
        return self.candidatas[0][1] if self.candidatas else None

    @property
    def camadas(self) -> dict[str, dict]:
        camadas = self.quant.get("layers")
        return camadas if isinstance(camadas, dict) else {}

    @property
    def formatos(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for cfg in self.camadas.values():
            nome = str(cfg.get("format", "?")) if isinstance(cfg, dict) else str(cfg)
            contagem[nome] = contagem.get(nome, 0) + 1
        return dict(sorted(contagem.items()))

    @property
    def familia(self) -> str | None:
        for nome in (self.sidecar.get("architecture"), self.analise.get("profile"),
                     self.sidecar.get("profile")):
            if nome and str(nome) in FAMILIA_POR_NOME:
                return FAMILIA_POR_NOME[str(nome)]
        return None


def ler_cabecalho(caminho: Path) -> tuple[dict, dict[str, str], int, int]:
    with caminho.open("rb") as h:
        tamanho_cabecalho = struct.unpack("<Q", h.read(8))[0]
        bruto = json.loads(h.read(tamanho_cabecalho))
    metadata = dict(bruto.pop("__metadata__", {}) or {})
    inicio_dados = 8 + tamanho_cabecalho
    ultimo = max((v["data_offsets"][1] for v in bruto.values()
                  if isinstance(v, dict) and "data_offsets" in v), default=0)
    sobra = caminho.stat().st_size - (inicio_dados + ultimo)
    return bruto, metadata, sobra, inicio_dados


def ler_dialeto_por_tensor(caminho: Path, tensores: dict, inicio_dados: int) -> dict[str, dict]:
    """O segundo dialeto: um tensor `<camada>.comfy_quant` de bytes UTF-8 com o JSON da camada.

    **Esta funcao existe porque a ferramenta era cega em silencio sem ela.** Na primeira varredura,
    `LTX25-distilled-DiT-comfy-w4a4` (riftcast, 1440 camadas de 4 bits de verdade) saiu como
    `SEM VEREDITO`, sem um unico achado -- que le como "nada a ver aqui". Ele nao carrega
    `_quantization_metadata` nenhum: carrega os tensores. `comfy/utils.py` converte um no outro ao
    carregar e `comfy/ops.py` despacha pelo JSON de cada camada, entao os dois dialetos sao
    igualmente validos e um arquivo pode trazer so o segundo.

    Custa um seek e ~50 bytes por camada, sem torch e sem GPU.
    """
    quantizadas: dict[str, dict] = {}
    alvos = [(t, i) for t, i in tensores.items()
             if t.endswith(".comfy_quant") and isinstance(i, dict) and "data_offsets" in i]
    if not alvos:
        return {}
    with caminho.open("rb") as h:
        for tensor, info in alvos:
            inicio, fim = info["data_offsets"]
            h.seek(inicio_dados + inicio)
            try:
                cfg = json.loads(h.read(fim - inicio).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(cfg, dict):
                quantizadas[tensor[: -len(".comfy_quant")]] = cfg
    return quantizadas


def indexar_analises(pastas: Iterable[Path]) -> list[tuple[dict, Path]]:
    """Todo `.analysis.json` legivel das pastas dadas, para casar depois com cada checkpoint."""
    achados: list[tuple[dict, Path]] = []
    for pasta in pastas:
        if not pasta.is_dir():
            continue
        for arquivo in sorted(pasta.glob("*.analysis.json")):
            try:
                achados.append((json.loads(arquivo.read_text(encoding="utf-8")), arquivo))
            except (OSError, json.JSONDecodeError):
                continue
    return achados


def casar_analise(sidecar: dict, indice: list[tuple[dict, Path]]) -> list[tuple[dict, Path]]:
    """Toda analise da mesma fonte, a mais confiavel primeiro. Nunca so uma.

    O sha vem antes do caminho de proposito: dois arquivos podem compartilhar caminho ao longo do
    tempo (a fonte foi substituida) e a analise antiga descreveria pesos que nao existem mais.

    **Devolver a lista inteira, e nao so a escolhida, e o ponto desta funcao.** Medido 2026-09-01
    contra `wan2.1_vace_1.3B_fp16`: duas calibragens da mesma fonte, diferindo so na semente
    (1234 contra 12345), dao mediana 0,051807 e 0,054631 -- 5,5% de diferenca no numero em que a
    linha por modelo inteira se apoia. Escolher uma em silencio e segurar um eixo sem dizer qual.
    """
    sha = sidecar.get("analysis_source_identity_sha256") or sidecar.get("source_identity_sha256")
    fonte = sidecar.get("source")

    def prioridade(par: tuple[dict, Path]) -> int:
        analise, _ = par
        casa_sha = bool(sha) and (analise.get("calibration") or {}).get("source_identity_sha256") == sha
        return 0 if casa_sha else 1

    candidatas = [(analise, arquivo) for analise, arquivo in indice
                  if (sha and (analise.get("calibration") or {}).get("source_identity_sha256") == sha)
                  or (fonte and analise.get("source") == fonte)]
    return sorted(candidatas, key=prioridade)


def carregar(caminho: Path, indice: list[tuple[dict, Path]]) -> Checkpoint:
    tensores, metadata, sobra, inicio_dados = ler_cabecalho(caminho)
    quant: dict = {}
    dialeto = "nenhum"
    bruto = metadata.get("_quantization_metadata")
    if bruto:
        try:
            quant = json.loads(bruto)
            dialeto = "_quantization_metadata"
        except json.JSONDecodeError:
            quant = {"_ilegivel": bruto[:200]}
            dialeto = "_quantization_metadata_ilegivel"
    if not (isinstance(quant.get("layers"), dict) and quant["layers"]):
        por_tensor = ler_dialeto_por_tensor(caminho, tensores, inicio_dados)
        if por_tensor:
            quant = {"layers": por_tensor}
            dialeto = "comfy_quant por tensor"

    sidecar: dict = {}
    caminho_sidecar = caminho.with_suffix(".quant.json")
    if caminho_sidecar.is_file():
        try:
            sidecar = json.loads(caminho_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sidecar = {}

    candidatas = casar_analise(sidecar, indice) if sidecar else []
    return Checkpoint(caminho=caminho, tamanho=caminho.stat().st_size, tensores=tensores,
                      metadata=metadata, quant=quant, sidecar=sidecar,
                      candidatas=candidatas, sobra_bytes=sobra, dialeto=dialeto)


# ---------------------------------------------------------------------------------------------
# Checagens
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Checagem:
    codigo: str
    olha: str
    cego: str          # o que ESTA checagem nao cobre. Obrigatorio; impresso toda execucao.
    funcao: object


CHECAGENS: list[Checagem] = []


def checagem(codigo: str, olha: str, cego: str):
    def registrar(funcao):
        CHECAGENS.append(Checagem(codigo, olha, cego, funcao))
        return funcao
    return registrar


@checagem(
    codigo="sobra",
    olha="bytes depois do fim do ultimo tensor",
    cego="nao le os bytes de sobra nem diz de onde vieram; so que estao la",
)
def checar_sobra(ck: Checkpoint) -> Iterator[Achado]:
    """Medido 2026-08-31 nos dois arquivos Abiray/Minimax-H3: 83 e 64 bytes de sobra.

    `safetensors.safe_open` recusa o arquivo inteiro com "incomplete metadata, file not fully
    covered"; o leitor de VRAM dinamica (`comfy_aimdo.model_mmap`) aceita. Como VRAM dinamica e o
    padrao, o arquivo funciona para quase todo mundo e falha exatamente em quem passa
    `--disable-dynamic-vram` -- que e o que esta bancada precisa para Nunchaku e LTX 2.5. Nao e
    download quebrado: o `Content-Length` do servidor bate byte a byte.
    """
    if ck.sobra_bytes > 0:
        yield Achado(OLHAR, "sobra",
                     f"{ck.sobra_bytes} bytes depois do ultimo tensor: `safe_open` recusa o "
                     f"arquivo, o leitor de VRAM dinamica aceita",
                     {"bytes": ck.sobra_bytes})
    elif ck.sobra_bytes < 0:
        yield Achado(REPROVA, "truncado",
                     f"o cabecalho declara {-ck.sobra_bytes} bytes a mais do que o arquivo tem",
                     {"faltando": -ck.sobra_bytes})


@checagem(
    codigo="escala",
    olha="cada camada declarada quantizada tem os tensores que o formato dela exige",
    cego="nao confere o VALOR das escalas, so a presenca e a forma",
)
def checar_escala(ck: Checkpoint) -> Iterator[Achado]:
    """A armadilha do qkv fundido: a camada carrega sem escala e **sem erro**.

    ComfyUI funde `attention.to_{q,k,v}` em `attention.qkv` ao carregar e so `.weight` esta no
    mapa de renome, entao `weight_scale` e `comfy_quant` passam sem renomear. `tools/to_native.py`
    existe por causa disso. O sintoma no arquivo e exatamente este: uma camada listada no
    `_quantization_metadata` cujo tensor de escala nao esta la.
    """
    desconhecidos: set[str] = set()
    faltando: list[str] = []
    for camada, cfg in ck.camadas.items():
        formato = str(cfg.get("format", "")) if isinstance(cfg, dict) else str(cfg)
        conhecido = FORMATOS.get(formato)
        if conhecido is None:
            desconhecidos.add(formato)
            continue
        if f"{camada}.weight" not in ck.tensores:
            faltando.append(f"{camada}.weight")
        for sufixo in conhecido.exigidos:
            if f"{camada}.{sufixo}" not in ck.tensores:
                faltando.append(f"{camada}.{sufixo}")

    if faltando:
        yield Achado(REPROVA, "escala_ausente",
                     f"{len(faltando)} tensores exigidos pelo formato declarado nao estao no "
                     f"arquivo; a camada carrega sem escala e sem erro",
                     {"total": len(faltando), "exemplos": faltando[:6]})
    if desconhecidos:
        yield Achado(OLHAR, "formato_desconhecido",
                     f"formato(s) que esta camada nao sabe conferir: "
                     f"{', '.join(sorted(desconhecidos))} -- sem veredito sobre as escalas deles",
                     {"formatos": sorted(desconhecidos)})


@checagem(
    codigo="forma",
    olha="o container int4 tem metade das colunas e a largura divide o groupsize",
    cego="nao pega escala torta em int8_tensorwise (ver FORMAS_DE_ESCALA) nem se os bits int4 decodificam certo",
)
def checar_forma(ck: Checkpoint) -> Iterator[Achado]:
    """As regras aqui sao de int4 empacotado e nao valem para qualquer formato quantizado.

    Aplicadas a todos, elas **reprovaram um FP8 publico e valido** na primeira varredura:
    `flux-2-klein-base-4b-fp8` declara `float8_e4m3fn`, guarda o peso em `F8_E4M3` sem empacotar
    e usa escala escalar `[]` -- tres violacoes das regras de int4 e nenhum defeito. Uma checagem
    que reprova arquivos bons ensina as pessoas a desligar checagens.
    """
    ruins: list[str] = []
    grupo_ruim: list[str] = []
    for camada, cfg in ck.camadas.items():
        if not isinstance(cfg, dict):
            continue
        formato = FORMATOS.get(str(cfg.get("format", "")))
        info = ck.tensores.get(f"{camada}.weight")
        if formato is None or not info or len(info.get("shape", [])) != 2:
            continue
        linhas, colunas = info["shape"]
        if formato.empacota_int4:
            if info.get("dtype") != "I8":
                ruins.append(f"{camada}: container {info.get('dtype')}, esperado I8")
                continue
            gs = cfg.get("convrot_groupsize")
            if gs and (colunas * 2) % int(gs):
                grupo_ruim.append(f"{camada}: {colunas * 2} colunas nao divide por {gs}")
        aceitas = FORMAS_DE_ESCALA[formato.escala](linhas, cfg)
        escala = ck.tensores.get(f"{camada}.weight_scale")
        if aceitas is not None and escala and escala.get("shape") not in aceitas:
            ruins.append(f"{camada}: weight_scale {escala.get('shape')}, esperado {aceitas[0]}")

    for lista, codigo, texto in ((ruins, "forma_errada", "forma inconsistente com o formato"),
                                 (grupo_ruim, "groupsize", "largura nao divide o convrot_groupsize")):
        if lista:
            yield Achado(REPROVA, codigo, f"{len(lista)} camadas com {texto}",
                         {"total": len(lista), "exemplos": lista[:6]})


@checagem(
    codigo="resumo",
    olha="o resumo no topo do metadata contra o que as camadas realmente dizem",
    cego="nao conta forwards; o formato declarado por camada pode nao ser o executado",
)
def checar_resumo(ck: Checkpoint) -> Iterator[Achado]:
    """Medido no `MiniMax_H3_FL2VA` da Abiray, 791k downloads: o resumo se contradiz.

    `convrot_w4a4_mixed` com `"linear_dtype": "int4"` no topo e `"w4a4_int4mm_layers": 0` tres
    chaves abaixo, enquanto os 117 tensores `comfy_quant` por camada dizem `"int8"`. **Leia as
    camadas, nao o resumo.**
    """
    declarado = ck.quant.get("linear_dtype") or ck.sidecar.get("linear_dtype")
    if not declarado:
        return
    largura = {"convrot_w4a4": "int4", "asym_w4a8_int8": "int8"}
    reais = {largura.get(f, f) for f in ck.formatos}
    if reais and str(declarado) not in reais:
        yield Achado(OLHAR, "resumo_contradiz",
                     f"o resumo diz `linear_dtype: {declarado}` e as camadas dizem "
                     f"{', '.join(sorted(reais))} -- leia as camadas",
                     {"resumo": str(declarado), "camadas": sorted(reais)})


@checagem(
    codigo="backend",
    olha="o sidecar registra que os kernels resolveram para o backend CUDA",
    cego="e o registro de uma conversao passada, nao uma execucao agora",
)
def checar_backend(ck: Checkpoint) -> Iterator[Achado]:
    """O backend eager declara as mesmas capacidades e produziria numeros de matematica dequantizada.

    Este campo e o unico registro do preflight que o conversor fez. Ele prova o que aconteceu na
    hora da conversao; nao prova o que acontece ao carregar hoje, com outra versao de
    comfy-kitchen. Por isso o cego acima -- e por isso `tools/avaliar_despacho.py` (camada 2)
    existe: ela carrega e conta, que e a unica forma de fechar esta lacuna.
    """
    if not ck.formatos:
        return
    backend = ck.sidecar.get("backend")
    if not isinstance(backend, dict) or not backend:
        motivo = ("sem sidecar `.quant.json` -- provavelmente de terceiros"
                  if not ck.sidecar else
                  "sidecar presente e SEM campo `backend` -- conversao nossa sem registro")
        yield Achado(OLHAR, "backend_sem_registro",
                     f"nada registra que o preflight nativo passou ({motivo}); o backend eager "
                     f"declara as mesmas capacidades e roda matematica dequantizada",
                     {"tem_sidecar": bool(ck.sidecar)})
        return
    fora = {op: impl for op, impl in backend.items() if not str(impl).startswith(BACKEND_NATIVO)}
    if fora:
        yield Achado(REPROVA, "backend_nao_nativo",
                     f"{len(fora)} op(s) resolveram fora do backend CUDA na conversao",
                     {"ops": fora})


PRECISAO_DA_BANDA = 4
"""Casas decimais em que as bandas sao conhecidas, e portanto em que faz sentido compara-las.

Nao e cosmetico. Na primeira execucao desta ferramenta o build BOM do Wan -- o que definiu o
proprio valor tolerado -- foi marcado suspeito porque a mediana crua 0,054631 e maior que o
0,0546 da tabela. Comparar um numero cru com uma tabela arredondada inventa uma quinta casa que
nenhuma medicao desta bancada tem.
"""


def classificar(mediana: float, banda: Banda) -> str:
    """De que lado da linha DAQUELE modelo o numero cai. Tres valores, nenhum deles "aprovado"."""
    if banda.nao_tolerado is not None and mediana >= banda.nao_tolerado:
        return "quebra"
    return "faixa_nao_testada" if mediana > banda.tolerado else "abaixo_do_tolerado"


def mediana_efetiva(ck: Checkpoint, analise: dict) -> tuple[float, int] | None:
    """Por camada, o erro do formato que aquela camada de fato recebeu. Mediana disso.

    O `convrot_groupsize` da analise tem de bater com o da camada, e esse foi um ponto cego real.
    Ate 2026-09-03 esta funcao casava a analise so pelo sha da FONTE, e o erro por camada foi
    medido *em* um groupsize -- entao tres builds do mesmo Z-Image diferindo so nesse valor
    recebiam a mesma mediana 0,1216, com o render mostrando o cg 16 DESTRUIDO e os outros dois
    bons. Uma analise de cg 256 nao descreve um arquivo cg 16, e devolver o numero dela como se
    descrevesse e pior que nao devolver nada: passa por medicao.
    """
    camadas_analise = {c["layer"]: c for c in analise.get("layers", [])
                       if isinstance(c, dict) and "layer" in c}
    chave = {"convrot_w4a4": "err_w4a4", "asym_w4a8_int8": "err_w4a8", "bf16": "err_bf16"}
    cg_analise = analise.get("convrot_groupsize")
    erros: list[float] = []
    for camada, cfg in ck.camadas.items():
        formato = str(cfg.get("format", "")) if isinstance(cfg, dict) else str(cfg)
        if cg_analise is not None and isinstance(cfg, dict):
            cg_camada = cfg.get("convrot_groupsize")
            if cg_camada is not None and cg_camada != cg_analise:
                continue
        medida = camadas_analise.get(camada)
        if medida and chave.get(formato) in medida:
            erros.append(float(medida[chave[formato]]))
    return (statistics.median(erros), len(erros)) if erros else None


def groupsizes_do_checkpoint(ck: Checkpoint) -> set:
    """Os `convrot_groupsize` que as camadas deste arquivo declaram."""
    saida = set()
    for cfg in ck.camadas.values():
        if isinstance(cfg, dict) and cfg.get("convrot_groupsize") is not None:
            saida.add(cfg["convrot_groupsize"])
    return saida


@checagem(
    codigo="erro",
    olha="a mediana do erro efetivo contra a faixa conhecida DAQUELE modelo",
    cego="preve o erro de predicao do modelo, nunca a imagem final; so uma renderizacao decide",
)
def checar_erro(ck: Checkpoint) -> Iterator[Achado]:
    """Erro efetivo = por camada, o erro do formato que aquela camada de fato recebeu.

    Sai sem GPU porque as duas metades ja estao em disco: o sidecar diz qual formato cada camada
    levou e o `.analysis.json` diz o erro medido de cada formato naquela camada, contra uma
    referencia float32, nas ativacoes reais que a camada viu amostrando.

    O criterio por camada foi dado como nao-preditivo nesta bancada em 2026-08-30 e **reabilitado
    no dia seguinte**: ele preve o erro de predicao do modelo (1,49x contra 1,33x, mesma direcao,
    mesmo vencedor). O que ele nao preve e a imagem livre -- e nada preve.
    """
    medidas = [(mediana_efetiva(ck, analise), arquivo) for analise, arquivo in ck.candidatas]
    medidas = [(m, arquivo) for m, arquivo in medidas if m]
    if not medidas:
        # Silencio aqui era a forma do ponto cego. Se ha analise da mesma fonte mas nenhuma no
        # groupsize deste arquivo, o motivo tem de aparecer -- senao o arquivo sai SEM VEREDITO
        # como se ninguem nunca tivesse calibrado a fonte dele.
        cgs_ck = groupsizes_do_checkpoint(ck)
        cgs_an = {a.get("convrot_groupsize") for a, _ in ck.candidatas
                  if a.get("convrot_groupsize") is not None}
        if ck.candidatas and cgs_ck and cgs_an and not (cgs_ck & cgs_an):
            yield Achado(OLHAR, "analise_de_outro_groupsize",
                         f"ha {len(ck.candidatas)} analise(s) da mesma fonte, mas medidas em "
                         f"convrot_groupsize {sorted(cgs_an)} e este arquivo usa "
                         f"{sorted(cgs_ck)}. O erro por camada foi medido NUM groupsize e nao "
                         f"vale nos outros: medido nesta bancada, cg 16 / 64 / 256 do mesmo "
                         f"Z-Image dao 0,1926 / 0,1516 / 0,1312 na mesma populacao de camadas, e "
                         f"so o cg 16 destroi a imagem. Recalibre em "
                         f"{sorted(cgs_ck)} para ter veredito neste eixo",
                         {"groupsize_do_arquivo": sorted(cgs_ck),
                          "groupsize_das_analises": sorted(cgs_an)})
        return

    (mediana, casadas), origem = medidas[0]
    arredondada = round(mediana, PRECISAO_DA_BANDA)
    banda = BANDAS.get(ck.familia or "")
    todas = {str(arquivo): round(m[0], PRECISAO_DA_BANDA) for m, arquivo in medidas}
    evidencia = {"mediana_erro_efetivo": arredondada, "camadas_casadas": casadas,
                 "camadas_declaradas": len(ck.camadas), "analise": str(origem)}
    if len(medidas) > 1:
        # O espalhamento viaja SEMPRE ao lado do numero, mesmo quando nao vira achado: um numero
        # de duas casas saido de uma calibragem so reivindica uma precisao que esta bancada nao
        # tem. Ver CLAUDE.md, "A single run is not a measurement".
        evidencia["por_calibragem"] = todas

    if banda is None:
        yield Achado(OLHAR, "erro_sem_faixa",
                     f"mediana do erro efetivo {arredondada}, e nao ha faixa medida para esta "
                     f"familia -- sem veredito neste eixo", evidencia)
        return

    evidencia |= {"familia": banda.familia, "tolerado": banda.tolerado,
                  "nao_tolerado": banda.nao_tolerado}

    # A divergencia entre calibragens da mesma fonte so e um achado quando ela muda o veredito.
    # Ela existe em quase todo arquivo desta bancada (a semente da calibragem move a mediana em
    # ~2-6%) e, disparando sempre, afogava o sinal que esta ferramenta serve para dar.
    classes = {classificar(valor, banda) for valor in todas.values()}
    if len(classes) > 1:
        yield Achado(OLHAR, "calibragens_discordam_do_veredito",
                     f"{len(medidas)} calibragens da mesma fonte cairiam em lados diferentes da "
                     f"linha de {banda.familia}: {todas}. O veredito abaixo usa a que casa pelo "
                     f"sha da fonte", evidencia | {"escolhida": str(origem)})

    classe = classificar(arredondada, banda)
    if classe == "quebra":
        yield Achado(REPROVA, "erro_acima_da_quebra",
                     f"mediana {arredondada} no ou acima do valor que quebrou {banda.familia} "
                     f"({banda.nao_tolerado}); {banda.nota}", evidencia)
    elif classe == "faixa_nao_testada":
        teto = ("e o teto desta familia nunca foi medido" if banda.nao_tolerado is None
                else f"e a quebra esta em {banda.nao_tolerado}")
        yield Achado(OLHAR, "erro_na_faixa_nao_testada",
                     f"mediana {arredondada} acima do unico valor que se sabe funcionar em "
                     f"{banda.familia} ({banda.tolerado}), {teto}. No Wan, o unico checkpoint "
                     f"medido na propria faixa nao testada saiu destruido", evidencia)


def _travas_do_encoder_soltas() -> tuple[bool, dict]:
    """As duas travas do text encoder estao soltas NESTA arvore do ComfyUI?

    LIDO NO FONTE, NAO EXECUTADO -- e o achado que usa isto diz isso na propria mensagem. A
    ferramenta nao importa torch nem carrega modelo (176 checkpoints em menos de um segundo e o
    motivo de ela existir), entao a unica pergunta barata que ela pode fazer e se o codigo que
    prende o kernel ainda esta la.

    Pergunta pelos TRES pontos, porque soltar so um nao adianta -- qualquer uma das travas sozinha
    ja manda a matematica para o caminho dequantizado, e isso custou uma medicao aqui: a primeira
    tentativa mexeu so em `sd1_clip.py` e a contagem de forwards continuou em zero.
    """
    raiz = Path(__file__).resolve().parent.parent / "ComfyUI" / "comfy"
    marcas = {
        "ops.py": "def quantized_text_encoder_math",
        "sd1_clip.py": "quantized_text_encoder_math(",
        "sd.py": "text_encoder_has_quantized_math(",
    }
    evidencia, todas = {}, True
    for arquivo, marca in marcas.items():
        caminho = raiz / arquivo
        try:
            tem = marca in caminho.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Sem a arvore do ComfyUI ao lado nao da para responder. Devolve o padrao de fabrica,
            # que e o estado em que a esmagadora maioria das instalacoes esta, e registra que a
            # resposta veio da ausencia do arquivo e nao de uma leitura.
            return False, {"comfy_nao_encontrado": str(caminho)}
        evidencia[arquivo] = tem
        todas = todas and tem
    evidencia["lido_no_fonte"] = True
    return todas, evidencia


@checagem(
    codigo="armadilha",
    olha="armadilhas de arquitetura visiveis nos nomes dos tensores",
    cego="so conhece as duas que ja custaram trabalho aqui; nao procura armadilhas novas",
)
def checar_armadilha(ck: Checkpoint) -> Iterator[Achado]:
    """As duas que ja custaram renderizacoes e uma sessao inteira nesta bancada.

    VACE: `WAN21_Vace.extra_conds` preenche `vace_frames` com zeros quando nao ha no de controle,
    passa por `process_latent_in` -- que subtrai a media do formato latente, entao zero vira **nao**
    nulo -- concatena mascara de uns e aplica em forca total. O FP16 sem quantizacao nenhuma sai
    destruido igual. Custou quatro renderizacoes ate alguem olhar o braco de referencia.

    Text encoder: `comfy/sd.py` chama `set_model_compute_dtype(torch.float32)` para todo CLIP,
    o que liga `comfy_force_cast_weights`, e `comfy/sd1_clip.py` fixa `full_precision_mm=True`.
    Duas travas independentes: o peso fica 4 bits na VRAM e a **matematica e dequantizada**.
    Memoria economizada, tempo nao, kernel nunca alcancado.

    Isso descreve o ComfyUI DE FABRICA, e em 2026-09-12 esta instalacao deixou de ser de fabrica:
    as duas travas foram soltas juntas (`patches/comfyui_text_encoder_quantized_math.patch`) e
    medidas -- 3,77x mais rapido com 1,11x menos fidelidade num encoder real. Enquanto este
    achado era incondicional, a ferramenta afirmava sobre ESTA maquina uma coisa que o patch ao
    lado dela ja refutava. `_travas_do_encoder_soltas()` pergunta a arvore em vez de assumir.
    """
    nomes = ck.tensores.keys()
    if any(".vace_blocks." in n for n in nomes):
        yield Achado(OLHAR, "vace_em_t2v",
                     "checkpoint VACE: em workflow T2V comum o ComfyUI aplica controle constante "
                     "em forca total e destroi a saida, sem erro. Confira o braco NAO quantizado "
                     "antes de acreditar em qualquer numero; use `--vace-strength 0`", {})
    if any(n.startswith("model.layers.") and ".self_attn." in n for n in nomes) and ck.formatos:
        soltas, evidencia = _travas_do_encoder_soltas()
        if soltas:
            yield Achado(OLHAR, "encoder_destravado_nao_medido",
                         "parece text encoder, e as duas travas do ComfyUI estao SOLTAS nesta "
                         "arvore, entao a matematica quantizada pode rodar -- o oposto do padrao "
                         "de fabrica. LIDO no fonte, nao executado: confirme com "
                         "`avaliar_despacho.py`, e note que soltar a trava TROCA fidelidade por "
                         "velocidade, entao um numero medido com a trava presa nao vale mais",
                         evidencia)
        else:
            yield Achado(OLHAR, "encoder_dequantizado",
                         "parece text encoder: de fabrica o ComfyUI roda encoders com a matematica "
                         "dequantizada (duas travas independentes). Economiza VRAM, nao economiza "
                         "tempo, e o kernel nunca e alcancado", evidencia)


@checagem(
    codigo="tamanho",
    olha="quanto o arquivo encolheu contra a fonte que o sidecar nomeia",
    cego="nao confere se a fonte no disco ainda e a mesma que foi quantizada",
)
def checar_tamanho(ck: Checkpoint) -> Iterator[Achado]:
    fonte = ck.sidecar.get("source_size")
    if not fonte or not ck.formatos:
        return
    razao = fonte / ck.tamanho
    if razao < 1.10:
        yield Achado(OLHAR, "quase_nao_encolheu",
                     f"declarado quantizado e so {razao:.2f}x mais leve que a fonte",
                     {"razao": round(razao, 3), "fonte_bytes": fonte, "saida_bytes": ck.tamanho})


# ---------------------------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------------------------

def fatos_de(ck: Checkpoint) -> dict:
    fatos = {
        "tamanho_bytes": ck.tamanho,
        "tensores": len(ck.tensores),
        "camadas_quantizadas": len(ck.camadas),
        "dialeto": ck.dialeto,
        "formatos": ck.formatos,
        "familia": ck.familia,
        "sidecar": ck.caminho.with_suffix(".quant.json").name
                   if ck.caminho.with_suffix(".quant.json").is_file() else None,
        "analise": str(ck.analise_origem) if ck.analise_origem else None,
    }
    if fonte := ck.sidecar.get("source_size"):
        fatos["razao_contra_fonte"] = round(fonte / ck.tamanho, 3)

    # A mediana e um FATO e sai sempre que puder ser calculada, mesmo -- principalmente -- quando
    # nenhum achado dispara nela. Na primeira versao ela so era reportada dentro do achado, entao
    # sumia exatamente dos checkpoints que passaram, que sao os que alguem vai querer comparar.
    medidas = [(mediana_efetiva(ck, analise), arquivo) for analise, arquivo in ck.candidatas]
    medidas = [(m, arquivo) for m, arquivo in medidas if m]
    if medidas:
        fatos["mediana_erro_efetivo"] = round(medidas[0][0][0], PRECISAO_DA_BANDA)
        fatos["camadas_com_erro_medido"] = medidas[0][0][1]
        if len(medidas) > 1:
            fatos["por_calibragem"] = {str(arquivo): round(m[0], PRECISAO_DA_BANDA)
                                       for m, arquivo in medidas}
    return fatos


def avaliar(caminho: Path, indice: list[tuple[dict, Path]]) -> Laudo:
    cego = [f"{c.codigo}: {c.cego}" for c in CHECAGENS]
    try:
        ck = carregar(caminho, indice)
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as erro:
        return Laudo(caminho, VEREDITO_OLHAR, [], {}, cego, erro_leitura=f"{type(erro).__name__}: {erro}")

    achados: list[Achado] = []
    for checagem_ in CHECAGENS:
        achados.extend(checagem_.funcao(ck))

    niveis = {a.nivel for a in achados}
    if REPROVA in niveis:
        veredito = VEREDITO_REPROVADO
    elif OLHAR in niveis:
        veredito = VEREDITO_OLHAR
    else:
        veredito = VEREDITO_SEM
    return Laudo(caminho, veredito, achados, fatos_de(ck), cego)


def coletar(alvos: list[Path]) -> tuple[list[Path], list[Path]]:
    """Devolve (arquivos, alvos_ausentes).

    Os ausentes voltam para o chamador em vez de so virarem aviso no stderr, porque esta
    ferramenta foi feita para rodar desacompanhada: um `.md` que nao diz que um alvo nao foi
    percorrido le como cobertura completa. Nesta bancada isso e concreto -- `D:` e um compartilhamento
    SMB de rede com ~408 GiB de modelos, e "o D: esta fora do ar" e um estado normal, nao quebrado.
    """
    arquivos: list[Path] = []
    ausentes: list[Path] = []
    for alvo in alvos:
        if alvo.is_dir():
            arquivos += sorted(alvo.rglob("*.safetensors"))
        elif alvo.is_file():
            arquivos.append(alvo)
        else:
            ausentes.append(alvo)
            print(f"aviso: nao existe, ignorado: {alvo}", file=sys.stderr)
    return arquivos, ausentes


def digest(laudos: list[Laudo], ausentes: list[Path] | None = None) -> str:
    linhas = [
        "# Avaliacao em lote, camada 1 (cabecalho, sidecar, analise)",
        "",
        "Ordenado por olha-aqui-primeiro. **Nenhum checkpoint aqui esta aprovado**: esta camada",
        "reprova, aponta e preve; ela nao aprova. Ver o cabecalho de `tools/avaliar.py` para o",
        "motivo medido.",
        "",
        "| veredito | arquivo | mediana erro | familia | achados |",
        "|---|---|---|---|---|",
    ]
    if ausentes:
        linhas[5:5] = [
            "",
            f"> **ATENCAO: {len(ausentes)} alvo(s) NAO foram percorridos** porque nao existem no "
            "disco agora. Esta lista esta incompleta:",
            "",
            *[f"> - `{alvo}`" for alvo in ausentes],
        ]
    for laudo in sorted(laudos, key=lambda x: (x.ordem, str(x.caminho))):
        if laudo.erro_leitura:
            linhas.append(f"| NAO LEU | `{laudo.caminho.name}` | - | - | {laudo.erro_leitura} |")
            continue
        erro = laudo.fatos.get("mediana_erro_efetivo")
        codigos = ", ".join(f"`{a.codigo}`" for a in laudo.achados) or "-"
        linhas.append(f"| {laudo.veredito} | `{laudo.caminho.name}` | "
                      f"{erro if erro is not None else '-'} | {laudo.fatos.get('familia') or '-'} "
                      f"| {codigos} |")

    for laudo in sorted(laudos, key=lambda x: (x.ordem, str(x.caminho))):
        if not laudo.achados and not laudo.erro_leitura:
            continue
        linhas += ["", f"## {laudo.caminho.name} -- {laudo.veredito}", ""]
        if laudo.erro_leitura:
            linhas.append(f"Nao foi possivel ler: `{laudo.erro_leitura}`")
            continue
        for achado in laudo.achados:
            linhas.append(f"- **{achado.nivel}** `{achado.codigo}` -- {achado.mensagem}")

    linhas += ["", "## Nao coberto por esta camada", ""]
    linhas += [f"- {linha}" for linha in (laudos[0].cego if laudos else [])]
    linhas += [
        "- dispatch: nada aqui carrega o modelo, entao ninguem contou forward quantizado nem",
        "  chamada a `dequantize`. Um arquivo pode passar tudo acima e rodar dequantizado.",
        "  Isto deixou de ser um buraco em 2026-09-01: `tools/avaliar_despacho.py` (camada 2, com",
        "  GPU) carrega pelo caminho normal do ComfyUI e conta. Rode-a antes de acreditar no campo",
        "  `backend` de qualquer sidecar -- ele registra a conversao, nao a execucao de hoje.",
        "- imagem: nenhuma renderizacao. Nenhum corte medido nesta bancada separa usavel de",
        "  inutilizavel -- 0,1837 correta contra 0,2147 destruida; 0,7173 boa contra 0,8255",
        "  destruida. So alguem olhando decide.",
        "- braco de referencia: o arquivo NAO quantizado nao e exercitado NESTA camada. Isto",
        "  deixou de ser um buraco em 2026-09-01: `tools/avaliar_referencia.py` (camada 3, com",
        "  GPU) mede se a referencia responde ao proprio condicionamento, e no par de verdade",
        "  conhecida do Wan separou o destruido do bom por 3,19x. Rode-a antes de acreditar em",
        "  qualquer numero desta tabela.",
    ]
    return "\n".join(linhas) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("alvos", nargs="+", type=Path, help="arquivos .safetensors ou pastas")
    p.add_argument("--saida", type=Path, default=None,
                   help="pasta para um .json por checkpoint e o digest.md")
    p.add_argument("--calib", type=Path, action="append", default=None,
                   help="pasta com .analysis.json (padrao: ./calib e ./bench)")
    p.add_argument("--json", action="store_true", help="imprime os laudos como JSON no stdout")
    a = p.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    pastas = a.calib or [raiz / "calib", raiz / "bench"]
    indice = indexar_analises(pastas)

    arquivos, ausentes = coletar([alvo.resolve() for alvo in a.alvos])
    if not arquivos:
        print("nenhum .safetensors encontrado", file=sys.stderr)
        return 2

    laudos = [avaliar(arquivo, indice) for arquivo in arquivos]

    if a.saida:
        a.saida.mkdir(parents=True, exist_ok=True)
        for laudo in laudos:
            destino = a.saida / f"{laudo.caminho.stem}.json"
            destino.write_text(json.dumps(laudo.para_json(), indent=2, ensure_ascii=False),
                               encoding="utf-8")
        (a.saida / "digest.md").write_text(digest(laudos, ausentes), encoding="utf-8")

    if a.json:
        print(json.dumps([laudo.para_json() for laudo in laudos], indent=2, ensure_ascii=False))
    else:
        print(digest(laudos, ausentes))

    contagem: dict[str, int] = {}
    for laudo in laudos:
        chave = "NAO LEU" if laudo.erro_leitura else laudo.veredito
        contagem[chave] = contagem.get(chave, 0) + 1
    print(f"\n{len(laudos)} checkpoints: " +
          ", ".join(f"{n} {k}" for k, n in sorted(contagem.items())), file=sys.stderr)
    if a.saida:
        print(f"escrito em {a.saida}", file=sys.stderr)
    print(f"{len(indice)} analises indexadas em {', '.join(str(x) for x in pastas)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
