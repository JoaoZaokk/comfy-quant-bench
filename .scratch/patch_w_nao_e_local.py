"""Corrige a afirmacao errada 'W: e disco local' em quatro documentos e no yaml. Medido com
`net use` em 2026-09-13 22:08: W: = \\\\192.168.3.40\\zfe (SMB), como P: e D:. So C: e F: sao NTFS
locais (`Get-Volume`). Eu tinha deduzido 'local' de um grep de `net use` que so procurava D:."""
import io

def sub(path, old, new, must=True):
    s = io.open(path, encoding='utf-8').read()
    if old not in s:
        if must:
            raise SystemExit(f"anchor not found in {path}: {old[:60]!r}")
        return
    io.open(path, 'w', encoding='utf-8').write(s.replace(old, new))
    print(f"patched {path}")

# 1. LTX 2.5 card (English)
p = 'bench/hf/ltx25-22b-w4a8/README.md'
sub(p, """BF16, DisTorch2 40 GB on cpu,      249 frames, same file on a local NTFS disk, 40 GiB RAM free
                                              -> works, 769.6 s, and pixel-identical to the first run
```

The frame count alone is not the problem — 249 frames run fine on this W4A8 build with no
splitting at all. A 39 GiB model that has to be memory-mapped is what is fragile, and where the
bytes come from matters as much as how many there are.""",
"""BF16, DisTorch2 40 GB on cpu,      249 frames, same bytes under a second name on ANOTHER SMB
                                              share (W:), 40 GiB RAM free
                                              -> works, 769.6 s, and pixel-identical to the first run
```

**Correction, 2026-09-13 (later the same day).** The first version of this list called W: "a
local NTFS disk". It is not: `net use` lists it as `\\\\192.168.3.40\\zfe`, a network share like D:
and P:. So both BF16 loads were memory-maps over SMB — one died, one survived — and the LTX 2.3
work that followed added three more deaths with the same signature (`access violation` inside
`torch/storage.py __getitem__` while `load_torch_file` pages the mapped file), on 39–43 GiB files,
with 32–44 GiB of RAM free, while a bare Python process paged the same 39 GiB file through in four
seconds. What is fragile is **a multi-tens-of-GiB safetensors memory-mapped over an SMB redirector
under load**: a paging read that fails surfaces as an in-page access violation and takes the whole
ComfyUI process with it. The only genuinely local NTFS disk on this machine with room is C:, and
that is where the 2.3 reference arm was moved. Judgement, not measurement: the redirector is the
suspect because it is the only thing the failing and surviving loads did not share evenly.""")

# 2. execution notes (PT)
p = 'bench/ltx25/README_execucao.md'
sub(p, """dentro do `UNETLoaderDisTorch2MultiGPU` — o mmap do arquivo de 39 GiB, que o resolvedor de nomes
lia de **D: (SMB)**, com ~24 GiB de RAM livre (uma conversão de 43 GiB e um download de 23 GiB
corriam ao mesmo tempo). O mesmo arquivo existe em W: (disco local, byte a byte igual); um
**hardlink com outro nome** (`ltx-2.5-22b-distilled-transformer-bf16_W.safetensors`, mesmo inode,
nada copiado nem movido) faz o loader ler de W:. Com 40 GiB livres e sem conversão concorrente:
769,6 s, `cpu,40gb`, sucesso, e pixel-idêntico. Lição registrada: o resolvedor do ComfyUI devolve o
PRIMEIRO caminho do yaml que tem o nome, e a ordem do yaml decide de qual disco um mmap de 39 GiB
sai — o que não aparece em log nenhum.""",
"""dentro do `UNETLoaderDisTorch2MultiGPU` — o mmap do arquivo de 39 GiB, que o resolvedor de nomes
lia de **D: (SMB)**, com ~24 GiB de RAM livre (uma conversão de 43 GiB e um download de 23 GiB
corriam ao mesmo tempo). O mesmo arquivo existe em W: (byte a byte igual); um **hardlink com outro
nome** (`ltx-2.5-22b-distilled-transformer-bf16_W.safetensors`, mesmo inode, nada copiado nem
movido) faz o loader ler de W:. Com 40 GiB livres e sem conversão concorrente: 769,6 s, `cpu,40gb`,
sucesso, e pixel-idêntico.

**CORREÇÃO (22:08 do mesmo dia): W: NÃO é disco local.** `net use` lista `W: \\\\192.168.3.40\\zfe`
— compartilhamento SMB, como P: e D:. Eu tinha deduzido "local" de um grep de `net use` que só
procurava D:. Logo as duas cargas do BF16 foram mmap por SMB: uma morreu, uma sobreviveu. O 2.3
acrescentou TRÊS mortes com a mesma assinatura (`access violation` em `torch/storage.py
__getitem__`, dentro do `get_tensor` do `load_torch_file`), em arquivos de 39–43 GiB, com 32–44 GiB
de RAM livre — enquanto um processo Python nu percorreu o mesmo arquivo de 39 GiB em 4 s. O que é
frágil é **safetensors de dezenas de GiB mapeado em memória por um redirecionador SMB sob carga**:
leitura de paginação que falha vira in-page error, que vira access violation, que leva o processo
inteiro. Os únicos discos NTFS locais são C: e F:; o transformer BF16 do 2.3 foi para C:. Lição que
fica: o resolvedor do ComfyUI devolve o PRIMEIRO caminho do yaml que tem o nome, e a ordem do yaml
decide de qual VOLUME um mmap de 39 GiB sai — e isso não aparece em log nenhum.""")

# 3. CLAUDE.md audio section
p = 'CLAUDE.md'
sub(p, """**And the BF16 arm killed the server once.** `Windows fatal exception: access violation` in
`torch/storage.py __getitem__` under `comfy/utils.py:136 load_torch_file` — the memory-map of the
39 GiB file, which the name resolver was reading from **D: (SMB)** with ~24 GiB of RAM free while
a 43 GiB conversion and a 23 GiB download ran. The same bytes exist on W: (local disk); a hardlink
under another name (`..._bf16_W.safetensors`, same inode, nothing copied or moved) made the loader
read from W:, and with 40 GiB free it rendered in 769.6 s. `folder_paths` returns the **first**
yaml root that has the name, so yaml order decides which disk a 39 GiB mmap comes from, and no log
says which. Proofs on the Hub: `av/*.mp4`, `av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.""",
"""**And the BF16 arm killed the server once — then the 2.3 work killed it three more times, same
signature.** `Windows fatal exception: access violation` in `torch/storage.py __getitem__` under
`comfy/utils.py:136` (`f.get_tensor(k)` inside `load_torch_file`) — the page-in of a memory-mapped
39–43 GiB safetensors. First from D: with ~24 GiB RAM free; the retry from W: under a second name
(hardlink, same inode) rendered in 769.6 s with 40 GiB free. **This file first recorded W: as "a
local disk". It is not**: `net use` lists `W: \\\\192.168.3.40\\zfe`, an SMB share like D: and P:;
the claim came from a grep of `net use` that only looked for D:. So every BF16 load here was an
mmap over SMB — one survived, four died (32–44 GiB free, no correlation with RAM), while a bare
Python process paged the same 39 GiB file through in 4 s. What is fragile is a tens-of-GiB mmap
over an SMB redirector under load: a failed paging read surfaces as an in-page access violation
and takes the process. The only local NTFS volumes are C: and F:; the 2.3 reference arm runs from
C:. `folder_paths` returns the **first** yaml root that has the name, so yaml order decides which
*volume* a 39 GiB mmap comes from, and no log says which. Proofs on the Hub: `av/*.mp4`,
`av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.""")

# 4. progress log part 50
p = 'W4A4_PROGRESS.md'
sub(p, """2.5 (39 GiB) lido de D: (SMB) com ~24 GiB de RAM livre; (2) o checkpoint UNICO do 2.3 (43 GiB)
pelo caminho de checkpoint + DisTorch2, lido de W: (disco local) com ~40 GiB livres. O que
sobreviveu, duas vezes: o transformer sozinho pelo `UNETLoader`, de disco local, 39 GiB.""",
"""2.5 (39 GiB) lido de D: (SMB) com ~24 GiB de RAM livre; (2) o checkpoint UNICO do 2.3 (43 GiB)
pelo caminho de checkpoint + DisTorch2, lido de W: com ~40 GiB livres. [CORRIGIDO 22:08: W: NAO e
disco local -- `net use` da `\\\\192.168.3.40\\zfe`, SMB; eu deduzi "local" de um grep que so
procurava D:. Depois desta parte morreram mais dois, ambos o transformer extraido lido de W:, com
32-44 GiB livres, enquanto um processo nu percorre o arquivo em 4 s. Mecanismo: mmap de dezenas de
GiB por SMB sob carga -> in-page error -> access violation. O braco BF16 do 2.3 foi para C:, o
unico NTFS local com espaco.] O que sobreviveu, uma vez: o transformer sozinho pelo `UNETLoader`,
39 GiB, tambem por SMB.""")

# 5. yaml comments
p = 'ComfyUI/extra_model_paths.yaml'
sub(p, "# LTX 2.3 BF16 (43 GiB) copiado para W: (disco local) com OUTRO nome, porque o mmap do arquivo de",
       "# LTX 2.3 BF16 (43 GiB) copiado para W: (que NAO e local: SMB, \\\\192.168.3.40\\zfe) com OUTRO nome, porque o mmap do arquivo de")
print('all patched')
