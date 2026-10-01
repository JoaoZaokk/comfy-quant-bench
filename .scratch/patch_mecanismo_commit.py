"""Corrige, nos quatro documentos, o mecanismo das mortes do servidor no braco BF16: NAO e o
redirecionador SMB (julgamento de 2026-09-13, escrito como tal), e COMMIT -- medido 2026-09-14
00:18-00:30 com scratchpad/probe_commit_mmap.py, probe_safeopen_trace.py e probe_double_map.py.
Anchors exatos; falha alto se algum nao existir."""
import io

def sub(path, old, new):
    s = io.open(path, encoding='utf-8').read()
    if s.count(old) != 1:
        raise SystemExit(f"anchor count {s.count(old)} in {path}: {old[:70]!r}")
    io.open(path, 'w', encoding='utf-8').write(s.replace(old, new))
    print(f"patched {path}")

MECANISMO_EN = """**And the redirector was the wrong suspect too — measured 2026-09-14.** The same two-view
mapping that `safe_open` performs, done by hand in a bare Python process without ComfyUI, dies the
same way on the local C: drive as on W:. What was measured on this 39.13 GiB file, with the system
commit counters read before and after each call (`GetPerformanceInfo`):

```
call                                                       commit charge
safetensors.safe_open(framework="pt")                      +40.8 GiB at open, before any tensor is read
   (two copy-on-write views of the same file: memmap2 for the header and
    torch.UntypedStorage.from_file(shared=False) for the data; +80.2 GiB while both are alive)
torch.empty of the model's parameters                      +40.7 GiB more
read-only mmap (what numpy and the GGUF loader use)          0
UntypedStorage.from_file(shared=True)                        0
```

Windows charges a copy-on-write view its whole size at mapping time, so ComfyUI's normal
safetensors path needs twice the file in commit just to open it and three times to build the
model. This machine's commit limit is 124.8 GiB (63.6 GiB of RAM plus a 61 GiB system-managed
pagefile) with about 70 GiB already committed by other processes. When the charge forces the
pagefile to grow, the new view sometimes comes back with the limit raised but the charge not
taken, and the first read through it is an access violation in `torch/storage.py __getitem__` —
the server's exact signature (`safe_open` + first `get_tensor` on W:, 3 of 3 runs; the hand-made
two-view mapping on C:, 1 of 1; the same calls survive on other runs, which is why one BF16 load
in seven succeeded). The two deaths inside `nn.Linear.__init__` — the model's `torch.empty` —
carry the same signature and were not reproduced in isolation. So the list above stays, with a
different reading: rows 3 to 5 died of commit, not of the network. The LTX 2.3 reference was
rendered without the safetensors reader at all: the same BF16 weights, bit for bit, in a GGUF
container read through a read-only memmap (`tools/safetensors_to_gguf_bf16.py`; bytes,
dequantized weight and Linear output verified identical on 12 sampled layers with
`tools/probe_gguf_bf16_equivalence.py`)."""

# 1. LTX 2.5 card
p = 'bench/hf/ltx25-22b-w4a8/README.md'
sub(p, """seconds. What is fragile is **a multi-tens-of-GiB safetensors memory-mapped over an SMB redirector
under load**: a paging read that fails surfaces as an in-page access violation and takes the whole
ComfyUI process with it. The only genuinely local NTFS disk on this machine with room is C:, and
that is where the 2.3 reference arm was moved. Judgement, not measurement: the redirector is the
suspect because it is the only thing the failing and surviving loads did not share evenly.""",
"""seconds. The redirector was the suspect for a few hours, as a judgement; the next paragraph is the
measurement that replaced it.

""" + MECANISMO_EN)

# 2. CLAUDE.md
p = 'CLAUDE.md'
sub(p, """Python process paged the same 39 GiB file through in 4 s. What is fragile is a tens-of-GiB mmap
over an SMB redirector under load: a failed paging read surfaces as an in-page access violation
and takes the process. The only local NTFS volumes are C: and F:; the 2.3 reference arm runs from
C:.""",
"""Python process paged the same 39 GiB file through in 4 s. The redirector was the suspect for a few
hours; the paragraph after this one is the measurement that replaced it. The only local NTFS
volumes are C: and F:.""")
sub(p, """Proofs on the Hub: `av/*.mp4`,
`av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.
""",
"""Proofs on the Hub: `av/*.mp4`,
`av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.

""" + MECANISMO_EN + """

**Rule that follows, for this machine:** never open a safetensors bigger than about half the free
commit through ComfyUI's normal reader (`safe_open` costs 2x the file; the model another 1x). For a
BF16 reference of a 20 B+ model, convert it losslessly with `tools/safetensors_to_gguf_bf16.py`
and load it with `UnetLoaderGGUF`; keep the text encoder out of the process with
`tools/ltx_video.py --encode-only` / `--cond-from` (saved conditioning costs nothing). The pagefile
is the owner's, not ours: `?:\\pagefile.sys`, system-managed, 61.2 GiB on C: at the time of writing,
and every death above needed it to grow.
""")

# 3. README_execucao (PT)
p = 'bench/ltx25/README_execucao.md'
sub(p, """de RAM livre — enquanto um processo Python nu percorreu o mesmo arquivo de 39 GiB em 4 s. O que é
frágil é **safetensors de dezenas de GiB mapeado em memória por um redirecionador SMB sob carga**:
leitura de paginação que falha vira in-page error, que vira access violation, que leva o processo
inteiro. Os únicos discos NTFS locais são C: e F:; o transformer BF16 do 2.3 foi para C:.""",
"""de RAM livre — enquanto um processo Python nu percorreu o mesmo arquivo de 39 GiB em 4 s. O
redirecionador ficou como suspeito por algumas horas, e como julgamento; o parágrafo seguinte é a
medição que o substituiu. Os únicos discos NTFS locais são C: e F:.""")
sub(p, """decide de qual VOLUME um mmap de 39 GiB sai — e isso não aparece em log nenhum.""",
"""decide de qual VOLUME um mmap de 39 GiB sai — e isso não aparece em log nenhum.

**CORREÇÃO DA CORREÇÃO (2026-09-14, 00:18–00:30, MEDIDO): não é o SMB, é COMMIT.** Três probes num
processo nu, sem ComfyUI, lendo os contadores de commit do sistema antes e depois de cada chamada
(`scratchpad/probe_commit_mmap.py`, `probe_safeopen_trace.py`, `probe_double_map.py`), no mesmo
arquivo de 39,13 GiB:

```
chamada                                                   cobrança de commit
safetensors.safe_open(framework="pt")                     +40,8 GiB ao abrir, antes de ler tensor algum
   (duas views copy-on-write do mesmo arquivo: memmap2 para o cabeçalho e
    torch.UntypedStorage.from_file(shared=False) para os dados; +80,2 GiB enquanto as duas vivem)
torch.empty dos parâmetros do modelo                      +40,7 GiB a mais
mmap somente-leitura (numpy, loader GGUF)                    0
UntypedStorage.from_file(shared=True)                        0
```

O Windows cobra uma view copy-on-write pelo tamanho inteiro no ato do mapeamento, então o caminho
normal do ComfyUI precisa de 2x o arquivo em commit só para abrir e 3x para construir o modelo. O
teto desta máquina é 124,8 GiB (63,6 GiB de RAM + pagefile de 61 GiB gerido pelo sistema) com ~70 GiB
já comprometidos por outros processos. Quando a cobrança força o pagefile a crescer, a view às vezes
volta com o limite aumentado e a cobrança NÃO feita, e a primeira leitura por ela dá access violation
em `torch/storage.py __getitem__` — a assinatura exata do servidor (`safe_open` + primeiro
`get_tensor` em W:, 3 de 3; o mapeamento duplo feito à mão em C:, 1 de 1; a mesma chamada sobrevive
em outras corridas, que é por que uma carga do BF16 em sete deu certo). As duas mortes dentro de
`nn.Linear.__init__` (o `torch.empty` do modelo) têm a mesma assinatura e não foram reproduzidas
isoladas. O volume nunca importou. O braço BF16 do 2.3 foi renderizado SEM o leitor de safetensors:
os mesmos bytes BF16 num GGUF (`tools/safetensors_to_gguf_bf16.py`), lido por memmap
somente-leitura pelo `UnetLoaderGGUF` — bytes, peso dequantizado e saída do Linear conferidos
idênticos em 12 camadas (`tools/probe_gguf_bf16_equivalence.py`). Regra que fica: nesta máquina, não
abrir por `safe_open` arquivo maior que metade do commit livre; encoder fora do processo
(`--encode-only` / `--cond-from`).""")

# 4. W4A4_PROGRESS parte 50
p = 'W4A4_PROGRESS.md'
sub(p, """Mecanismo: mmap de dezenas de
GiB por SMB sob carga -> in-page error -> access violation. O braco BF16 do 2.3 foi para C:, o
unico NTFS local com espaco.]""",
"""Mecanismo escrito aqui como "mmap por SMB
sob carga -> in-page error"; CORRIGIDO DE NOVO na parte 51: e COMMIT, nao SMB -- medido em processo
nu, morre igual em C:. O braco BF16 do 2.3 foi para C: e nao adiantou por isso.]""")
print('all patched')
