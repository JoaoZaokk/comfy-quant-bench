"""QAT ternário do FLUX.2-klein por destilação contra o próprio BF16. Retomável, para rodar em casa ou no Colab.

DE ONDE VEM. O braço 1 (`ajusta_denso_diffusers.py`) congelou o corpo ternário e ajustou só 9 tensores
densos: fechou 54% da distância PTQ→Bonsai no epsilon fora da amostra e **não gerou imagem
utilizável** (`bench/render_braco1_resultado_2026-09-22.md`). O Bonsai trocou 2,7 M de sinais no
corpo — treinou os códigos. Este tool deixa o corpo treinar:

    professor   klein BF16, congelado; grava (latente ruidoso, t, condição) -> saída, na trajetória DELE
    aluno       inicializado do MESMO BF16 (não do ternário: o valor contínuo diz quem está perto
                do limiar); no forward cada Linear do corpo vira ternário por STE; o denso fica em
                precisão normal, como no braço 1
    perda       MSE em fp32 entre as duas saídas
    um backprop -> o mestre anda (contínuo) -> quando cruza o limiar, o código troca

TERNÁRIO = o mesmo do braço 0 e do Bonsai: absmean por grupo de 128 no eixo K, código
`round(clamp(w/d, -1, 1))`, escala ótima L2 por grupo dado o código (`constroi_ternario_ingenuo.py`).
A escala é RECALCULADA a cada forward a partir do mestre ("escala fixa" no sentido de não ser
parâmetro); escala aprendida é outro eixo, não este.

STE EXATO. `w + (q(w) - w).detach()` em bf16 NÃO dá exatamente q(w) — dois arredondamentos — e o
forward deixaria de ser ternário. Usa-se uma `autograd.Function` que devolve q(w) no forward e passa
o gradiente inteiro no backward.

PESO MESTRE, o eixo que já custou um braço: bf16 puro com lr 1e-5 deixou 63,4% dos elementos do braço
1 sem mudar um bit. Opções (`--otim`):
    adamw8bit-sr   peso bf16 + AdamW8bit do torchao com arredondamento estocástico   ~6 B/param
    adamw4bit-sr   peso bf16 + AdamW4bit com SR -- o 8-bit NAO coube na 3090: pico 23,47 GiB e o WDDM
                   paginou (20 s/passo, 190-230 W), smoke de 2026-09-22 22:08                  ~5 B/param
    adamw8bit      peso fp32 + AdamW8bit                                            ~10 B/param
    adam-fp32      peso fp32 + Adam do torch                                        ~16 B/param
Para 3.875.544.576 params: 21,7 / 36,1 / 57,7 GiB SÓ de peso+grad+estado, fora ativação.

RETOMÁVEL, porque o recurso some sem aviso (memória `intervalo-de-gravacao-e-a-perda-maxima`):
  * o professor grava UM shard por (prompt, semente) e pula os que já existem;
  * o aluno grava checkpoint atômico (`.partial` + `os.replace`) a cada `--ckpt-min` minutos e retoma
    do último, com o estado do otimizador. O intervalo É a perda máxima: não subir sem motivo.
  * `journal.jsonl` recebe uma linha por log; o log termina com `FIM HH:MM:SS` (contrato do
    `probe_background_job.py` do supervisor do Colab).

MÉTRICAS que o critério pede, além da perda: perda num conjunto HOLDOUT de prompts que o treino não
vê; fração de códigos ternários diferentes do código inicial; fração que trocou desde o último log
(estabilidade).

NÃO COBRE: nenhuma imagem e nenhum epsilon no protocolo do ComfyUI — isso é feito depois, fora, com
`aplica_mapa_diffusers_bfl.py` + `probe_epsilon_ckpt_ab.py` + `quality_ladder.py`. Nenhum tempo de
1,58 bit: o corpo roda desempacotado em bf16.

ORGANIZAÇÃO (2026-09-29). Este arquivo é só a entrada, com a MESMA CLI (as células do Colab chamam este
caminho); o código vive em `tools/qat_klein/` e a quantização em `tools/lowbit_canon.py`:
    config.py   CLI + `Config` imutável validada (recusa --so-escalas + --l1-corpo etc.) e impressão digital
    modulos.py  `LinearQuant` (sem os globais NIVEIS/STE_LIGADO), otimizador, L1 proximal
    professor.py shards por CHAVE DE CONTEÚDO (legados `p####_s#.pt` continuam lidos), cache no HF
    estado.py   `EstadoCorrida` dentro do checkpoint atômico; SIGTERM -> checkpoint ao fim do passo
    hf_sync.py  envio sem nunca recomeçar do zero por erro transitório nem subir passo menor que o remoto
    avalia.py   holdout, sens, `melhor` e parada;  exporta.py  desempacotado e/ou lowbit_affine
    status.py   `status.json` atômico para o probe;  treino.py  a orquestração
A retomada agora restaura também a permutação da época, os dois sorteios e o `melhor` do MESMO passo do
checkpoint (os JSONs viram relatório) e recusa checkpoint de outra configuração.
Para o Colab sobem juntos: este arquivo, `qat_klein/`, `lowbit_canon.py` e `ajusta_denso_diffusers.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowbit_canon import (  # noqa: F401 -- .scratch/exporta_ckpt_qat.py
    BLOCO,
    Quantizador,
    pilhas_reais,
)
from qat_klein.professor import (
    grava_professor,  # noqa: F401 -- usado pelo gera_dataset_klein
)
from qat_klein.treino import main as _main
from qat_klein.util import le_prompts, log  # noqa: F401 -- idem


def ternariza(w, grupo: int):
    """A receita ternária do QAT de sempre (zero +0.0), para quem importava daqui."""
    return Quantizador(1, grupo).quantiza(w)


def main(argv=None, carrega=None) -> int:
    return _main(argv, carrega=carrega, descricao=__doc__.splitlines()[0])


if __name__ == "__main__":
    raise SystemExit(main())
