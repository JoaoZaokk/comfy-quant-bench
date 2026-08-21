# cortiq / LTX 2.5 — estado da sessão de 2026-08-19/20

Investigação do `infosave/LTX-2.5-cmf` (container `.cmf` de 20,6 GiB rodado pelo
`cortiq`, motor Rust) e comparação contra o ComfyUI desta bancada. Este documento é o
que resta quando o contexto da sessão some.

**O plano do que fazer a seguir NÃO está aqui.** Está em
[`.scratch/estado-entregavel/map.md`](.scratch/estado-entregavel/map.md), com 17
tickets. Aqui está só o que ficou *estabelecido*.

## Onde as coisas estão

| | |
|---|---|
| container | `F:\cortiq\ltx25-q4tp.cmf` — 22.071.398.554 B, sha256 `ce0733fb…92e8a1`, conferido contra o Hub. **NVMe local**, não o SMB |
| fonte do motor | `F:\cortiq-cmf` — clone de `github.com/infosave2007/cmf` |
| branch público | `pv-nt` → fork `JoaoZaokk/cmf`, 1 commit (patch + teste) |
| branch local | `pv-nt-audit-local` — 6 commits, instrumentação + 6 testes + 8 scripts + 1.393 linhas de documento |
| ledger completo | `F:\cortiq-cmf\AUDIT_LEDGER.md`, 938 linhas, rounds 0–9 com o método de cada medição |
| relatórios upstream | `F:\cortiq-cmf\UPSTREAM_REPORTS.md`, 5 achados |
| driver do ComfyUI | `tools/comfy_run_workflow.py` — converte workflow UI→API e roda por HTTP, sem MCP |
| página publicada | https://claude.ai/code/artifact/a210d79d-ccdc-4297-8b8f-9ffe07f92b4b |

Frames renderizados: `F:\cortiq\out_e2e_512_png\` (cortiq) e
`ComfyUI\output\ltx25_int8_convrot_*.png` (ComfyUI, 00001–00049 seed 1234, 00050–00098
seed 1235).

## O placar, medido nesta RTX 3090

Mesmo job dos dois lados: **512×512, 49 frames, 3 passos, cfg 1.0, euler, seed 1234,
25 fps**, mesmo prompt, decode de vídeo apenas. Frames **olhados**, não só contados.

| | cortiq q4tp (W4A16) | ComfyUI int8_convrot (W8A8) |
|---|---|---|
| processo novo, pesos do disco | **510,1 s** (mmap, NVMe) | **553,9 s** (SMB; ~544 s disso é carga) |
| pesos residentes | não há modo quente | **9,78 s** |
| prompt idêntico repetido | — | 0,01 s ← **cache, mediu nada** |

**Computo contra computo: ~52×.** Quebra do cortiq: encode 42,9 s + denoise 204,5 s +
decode 258,7 s.

**Ressalvas que viajam com o 52×:** precisão diferente, storage diferente na carga,
escada de sigmas diferente (o cortiq não aceita lista de sigmas), e o ComfyUI roda com
Sage e 60 pacotes de custom nodes. É teste de pipeline, não de kernel. Qualidade **não**
foi comparada e não dá para comparar deste par — mesma seed produziu cenas diferentes,
porque "seed 1234" não é o mesmo ruído em dois motores.

## Onde o tempo do cortiq está

**Denoise, 34–35% do passo são os GEMMs de attention.** 9216 chamadas por passo,
`1536 = 48 blocos × 32 cabeças` por forma:

```
values  n=1792 k=1792 m=128   1536 calls  8,65 / 6,61 s   146 / 191 GFLOP/s
values  n=1792 k=1024 m=128   1536 calls  7,64 / 7,53 s    94 /  96 GFLOP/s
scores  n=1792 k=128  m=1792  1536 calls  4,51 / 4,08 s   280 / 310 GFLOP/s
scores  n=1792 k=128  m=1024  1536 calls  2,74 / 2,49 s   263 / 290 GFLOP/s
```

**Decode do VAE, 46% do render inteiro** — 255,6 s medidos isolados via `ltx-decode`,
5,22 s/frame. Três blocos `res_x` carregam **81,6%** (97,5 / 64,7 / 57,8 s). Dentro do
`Conv3d::forward`, no host: **im2col 74,4 s (33%), gemm 139,3 s (61%), scatter 14,1 s
(6%)**.

**O que isso implica:** a convolução **já** é im2col + `gemm_nt`, e `gemm_nt` tem braço
de device. O alvo não é escrever kernel de convolução. Mas o im2col materializa 453 MB
de patches por chamada (209 MB de entrada viram 5,7 GB, porque cada voxel aparece em 27
patches), então mandar para a placa não é óbvio. Ticket `13` resolve.

## O patch PV-NT — correto, e mais preciso que o que ele substitui

`values` usa `gemm_dx` → `gemm_nn_coop`, que lê `v` **por colunas**. `scores` usa
`gemm_nt` → `gemm_nt_coop`, que lê **por linhas**. Os dois em tensor core; a diferença é
o padrão de acesso. Como `v` já é montado por cabeça num gather, montá-lo transposto é
de graça.

Contra referência **f64 calculada no teste**, via device, quatro formas reais:

| forma | NN atual | NT | fator |
|---|---|---|---|
| 1792×1792×128 self-attn | 2,836e-4 | **2,017e-7** | **1406×** |
| 1792×1024×128 cross-attn | 2,623e-4 | **1,530e-7** | **1714×** |
| 1792×49×128 vídeo↔áudio | 1,307e-7 | 7,149e-8 | 1,8× |
| 49×1792×128 áudio↔vídeo | 2,894e-4 | **2,050e-7** | **1412×** |

Velocidade: `values` 11,90–15,33 s → 5,10–5,38 s por passo, sem sobreposição em 4
corridas alternadas; `gather` +1,9 s. **Ganho no total do passo NÃO demonstrado** — o
braço OFF sozinho variou 56,1–66,5 s.

Fica `off by default` (`CMF_LTX_PV_NT=1`). O caso para ligar é precisão, não velocidade.

## Cinco afirmações que tiveram que ser desfeitas

Registradas porque a sequência é a lição, não nenhum passo dela.

1. **"O LTX não encosta na GPU no Windows."** Falso. Greppei `ltxdit.rs` e concluí
   ausência; as projeções passam por `Proj::matmat` (`dit.rs:126-133`). Execução:
   `coop=true` em toda dispatch, 14.757 MiB na placa.
2. **"O braço q4tp mede 414 GFLOP/s"** — número da docstring do autor. Mede
   **2842–2926**, e é *mais rápido* que o irmão f32 (1310–1957), porque lê 4 bits contra
   4 bytes num GEMM read-bound. Um plano de trabalho inteiro foi construído em cima e
   retirado.
3. **"O patch PV-NT está errado."** Métrica ruim: erro relativo elemento-a-elemento com
   denominador limitado em 1e-6, sobre latente centrado em zero. Esse critério **reprova
   o motor contra ele mesmo** — a via gpu diverge da via cpu, sem patch, mais do que o
   patch diverge (0,0856 contra 0,0700 em L2 relativo).
4. **"O VAE do LTX é transformer."** Li `vae3d.rs` porque o nome parecia certo e tinha
   profiler. É o VAE de outro modelo. Os dois comandos LTX constroem
   `ltxvae::ConvVaeDecoder`.
5. **"O teto de 32 threads deixa 16 dos 48 parados."** Esta máquina é um **Ryzen 5
   7600X, 6 núcleos / 12 threads**. O "48 threads" veio de um comentário sobre a máquina
   do autor.

Todas as cinco foram pegas porque o artefato carregava o **método** junto do número. A
única falha que eu **não** peguei sozinho foi a violação do lock da GPU — lá não havia
critério escrito, e quem pegou foi a sessão irmã.

## Upstream — enviado, e fora do escopo do plano

`PR #1` e `issues #2–#5` em `infosave2007/cmf`, todos com número medido e ressalva
colada. **Resposta de terceiro não é passo da nossa rota** — o mapa registra o que nós
fizemos, não o julgamento de quem recebe.

O `master` do clone continua intocado espelhando o upstream; o ledger e os relatórios
ficaram só no branch local, e o IP da NAS foi removido antes de qualquer push, com
**recommit** em vez de emenda (emenda deixaria o IP no histórico, que num repo público é
tão visível quanto o arquivo).

## Não feito, com o motivo

**Container q8.** O formato aceita (`--quant q8`). A passada do DiT sozinha consome
~41,5 GiB de RAM contra fonte de 39,13 GiB; o watchdog cortou em 5,93 GiB livres. Não é
limitação do cortiq nem do formato — é desta máquina. Sai liberando ~10 GiB ou noutro
host. Ver `run_pack_q8.ps1`, que tem watchdog e passe múltiplo.

## Ver também

- `.scratch/estado-entregavel/map.md` — o plano, 17 tickets
- `F:\cortiq-cmf\AUDIT_LEDGER.md` — o método de cada medição
- Memória: [[lock-gpu-compartilhado]], [[arquivo-plausivel-nao-e-o-caminho]],
  [[criterio-previo]], [[achados-viajam]]
