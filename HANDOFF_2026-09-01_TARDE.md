# Handoff — 2026-09-01, tarde

Continuação do `HANDOFF_2026-09-01_NOITE.md`, que fechou a publicação. Esta sessão fechou o
**avaliador** (camadas 1 e 3) e a **migração dos sete conversores** para o núcleo.

Tudo abaixo foi **executado**, não lido. Onde algo veio de leitura, está dito na linha.

---

## Suíte completa, rodada no fim desta sessão

```
test_comfy_run_workflow      exit 0    24 asserções
test_conversion_core         exit 0    38 passaram, 0 falharam
test_gpu_lock                exit 0    31 passed, 0 failed
test_hf_parallel_get         exit 0     9/9
test_ltx_studio              exit 0    71 asserções
test_native_probe            exit 0    20/20
test_quant_mixed_provenance  exit 0     9/9
test_quant_mixed_sigma       exit 0    17 asserções, 0 falhas
test_svdq_verify             exit 0    14 asserções
test_svdq_write_contract     exit 0    24 passed, 0 failed
test_timing                  exit 0    41 asserções
test_verify_formats          exit 0    13/13
```

**12 suítes, 12 exit 0, zero falhas, ~311 asserções.**

---

## 1. O avaliador em lote existe, em duas camadas

| camada | ferramenta | GPU? | o que decide |
|---|---|---|---|
| 1 | `tools/avaliar.py` | **não** | cabeçalho, sidecar, análise |
| 2 | não existe | sim | contar forward quantizado contra `dequantize` |
| 3 | `tools/avaliar_referencia.py` | sim | a referência responde ao próprio condicionamento? |

**Camada 1: 163 checkpoints em 0,68 s.** Sem torch, sem carregar modelo. Rodada agora:
`21 OLHAR, 5 REPROVADO, 137 SEM VEREDITO`.

A mediana do erro efetivo sai **offline** — o sidecar diz que formato cada camada levou, a análise
diz o erro daquele formato naquela camada. Reproduz todos os números publicados.

**`APROVADO` não existe no conjunto de vereditos**, por decisão. Nenhum corte medido aqui separa
usável de inutilizável: 0,1837 correta contra 0,2147 destruída; 0,7173 boa contra 0,8255
destruída. Ela reprova, aponta e prevê. Não aprova.

### Camada 3 validada contra verdade conhecida

Critério e as três condições de refutação escritos em `bench/criterio_guarda_referencia.md`
**antes** de medir.

| braço | resposta | veredito | previsão | acertou |
|---|---|---|---|---|
| Wan VACE 1.0 — destruído | **0,2477** | REPROVADO | `< 0,5` | sim |
| Wan VACE 0.0 — bom | **0,7911** | OLHAR | `> 0,8` | **não**, por 1,1% |
| Z-Image v2 BF16 — bom | **1,3459** | SEM VEREDITO | `> 0,8` | sim |
| HunyuanVideo 1.5 — bom | **2,8603** | SEM VEREDITO | `> 0,8` | sim |

As três condições de refutação ficaram caladas. **3,19x** separam o quebrado do são mais próximo.
**Teria abortado o Wan na primeira imagem em vez da quarta.**

O limiar de 0,8 errou por 1,1% e **não foi mexido** — mudar limiar depois de ver o número que ele
deveria classificar é descrever, não calibrar.

---

## 2. Ticket 08: os sete conversores passam pelo núcleo

Auditado por contagem: todos importam `_conversion`, chamam `conv.commit`, `conv.guard` e
`refuse_unsafe`, e **nenhum** abre `.partial` próprio.

### Consertou três divergências reais

- **`quant_w4a4_smooth` não tinha guarda de RAM nem de disco** — o único dos sete sem nenhuma, e o
  que mais acumula. Ganhou as duas.
- **`svdq_to_bf16` não tinha recusa de saída existente.** Ganhou, com `allow_quantized_source=True`.
- **`quant_mixed` condicionava a recusa a `not --dry-run`** — o ensaio pulava a checagem da
  execução real. Agora incondicional.

### Provado sem GPU

```
to_native     reconverteu o arquivo inteiro       sha256 IDENTICO, 11,46 GiB
quant_w4a4    header reconstruído sem kernel      IDENTICO, 238.264 bytes, 432 camadas
```

`tools/verificar_migracao.py` re-roda a segunda linha a qualquer momento, sem placa. Funciona
porque `plan_lazy` carrega dtype, forma e nbytes explícitos — o header inteiro sai sem chamar
kernel.

### Duas coisas que só a migração expôs

- **O núcleo tinha um buraco**: `to_native` funde `to_{q,k,v}` num `qkv`, e o núcleo só copiava
  faixa única. Daí `plan_copy_many`.
- **O núcleo era mais FRACO** que o que substitui: abria `.partial` com `wb` (trunca) onde os seis
  usam `xb`. O `test_svdq_write_contract` pegou **durante** a migração.

E a convenção de header teve que ser decidida por **contagem**: cinco de cinco quantizadores
escrevem `__metadata__` primeiro com `ensure_ascii=False`, e o núcleo fazia o contrário nos dois.
Adotá-lo como estava teria mudado em silêncio o layout de todo checkpoint reconvertido.

---

## 3. O QUE FAZER COM A GPU — está tudo escrito

**`bench/janela_gpu_migracao.md`**, escrito **antes** de qualquer corrida. Traz os pares a
reconverter, onde ficam os parâmetros de cada um (no `.quant.json` ao lado de cada saída), e o
critério: **qualquer diferença reprova**, inclusive só de header.

O ticket 08 **não fecha** até isso rodar. Nenhum byte de dado quantizado foi conferido — nenhum
kernel rodou em verificação nenhuma. Um conversor que planeje o header certo e escreva peso errado
passa em tudo que já foi feito.

**Ordem sugerida:**

1. `tools/verificar_migracao.py` (sem placa, 2 s) — se o header já diverge, não gaste GPU.
2. `quant_w4a4` sobre o Hunyuan → comparar sha256 com `hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot`.
3. `quant_mixed` sobre o Wan (usar `promote_error` do sidecar) → comparar com `misto005`.
4. `quant_w4a8` sobre o Hunyuan → comparar com `hv15_w4a8`.
5. `quant_w4a4_smooth` — **não há saída no disco**, aceitação mais fraca e precisa ser dita como
   tal: carregar pelo loader normal + `verify_w4a4 --kernel-smoke`. É o mais importante dos dois
   sem par, porque ganhou guardas novas num caminho que nunca rodou com elas.

**Lock:** conversor **não** passa por `_timing.compare()`, então tome `Assert-GpuLock` à mão para
eles. Para benchmark, não tome — a ferramenta toma sozinha e recusa a própria run.

**Cuidado medido hoje:** o 3080 Ti tem trabalho de terceiro (3268 MiB) e a `BenchGuard` recusa com
teto de 2 GiB. Use `CUDA_VISIBLE_DEVICES=0` — tira a placa da run e da guarda ao mesmo tempo.

---

## 4. Correções publicadas hoje

**O card do Z-Image tinha o 0,1241 na linha errada.** É do W4A4 puro (170 convrot), não do misto,
que é 0,0774. Confirmado em **nove calibragens**. Corrigido no HF no mesmo dia. O `tolerado` da
banda não muda de valor, só ganha dono — e é o build *mais agressivo* medido.

**A semente da calibragem move a mediana 2–6%.** Nunca tinha sido medido. Menor que a própria
banda (45%), então a linha por modelo sobrevive — mas o espalhamento agora viaja ao lado do número.

**Duas afirmações desatualizadas no `CLAUDE.md`**, em direções opostas: mandava caçar seis
`normal_comfy_backend` que não existem mais (são zero, viraram um `native_backend_ready` com teste
que trava isso), e o `convert.py` dizia que o contrato de escrita "virou um só" quando ninguém
tinha adotado. As duas corrigidas.

---

## 5. Aberto

| item | estado |
|---|---|
| Camada 2 do avaliador (`--dispatch`) | precisa de GPU, não começada |
| Verificar migração dos 4 que acumulam | precisa de GPU — `bench/janela_gpu_migracao.md` |
| Teto do Z-Image | célula vazia; ninguém tentou acima de 0,1241 |
| `--sem-pensar` na bateria de texto | **mente** — modelos ainda emitem `<think>` |
| `tools/probe_piso_de_ruido_do_prompt.py` | escrito, **nunca rodado** |
| Perfis novos (Flux, Wan 2.2, LTX 2.5, Qwen-Image) | cada um é trabalho novo |
| Lado da reprova da camada 3 | apoia-se em **um** braço quebrado de verdade |

---

## Estado de fechamento

```
git raiz     limpo (só .scratch/avaliacao/ sem rastrear)
HEAD         f80f3de
GPU 0        42 MiB, 0%  — livre
GPU 1        3268 MiB, 13% — trabalho de terceiro
lock         LIVRE
disco F:     190,9 GiB livres
```

**A GPU está tua.**
