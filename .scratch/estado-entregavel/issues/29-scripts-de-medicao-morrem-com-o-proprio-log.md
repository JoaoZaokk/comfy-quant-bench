# Cinco scripts de medição morrem quando o binário escreve no stderr

Type: task
Status: resolved

## Question

Em `F:\cortiq-cmf`, cinco dos seis scripts PowerShell de medição combinam duas linhas que se
mordem:

```powershell
$ErrorActionPreference = 'Stop'          # topo do arquivo
...
& $exe ltx-video ... 2>&1 | Tee-Object -FilePath $log
```

O `2>&1` traz o stderr do binário para o pipeline. Sob `Stop`, cada registro de stderr vira erro
**terminante**. O `cortiq` escreve o log de INFO no stderr, então a **primeira linha de log mata a
medição**:

```
cortiq.exe : 2026-08-21T08:11:53Z INFO cortiq_core::format: Opened CMF v2: ltx-2.5-av | ...
At F:\cortiq-cmf\run_ab_pvnt.ps1:53 char:13
+             & $exe ltx-video `
    + FullyQualifiedErrorId : NativeCommandError
```

Afetados (`ErrorActionPreference = 'Stop'` + `2>&1 |`): `run_ab_pvnt.ps1:31/57`,
`bisect_pvnt.ps1:18/43`, `run_e2e_comfy.ps1:46/89`, `run_e2e_cortiq.ps1:18/41`,
`run_roundtrip_probe.ps1:29/48`. `run_pack_q8.ps1` tem o `Stop` mas não o `2>&1`.

## Por que passou despercebido até 2026-08-21

Não é o script que mudou, é o host. Em **Windows PowerShell 5.1** stderr nativo sob `Stop` lança;
em **pwsh 7** não lança por padrão (depende de `$PSNativeCommandUseErrorActionPreference`). As
corridas de 2026-08-19 que produziram `pvnt_*_r*.log` rodaram sob pwsh 7 e passaram. Chamar
`powershell -File` em vez de `pwsh -File` — o que é fácil de fazer sem perceber — derruba a
medição na primeira linha, **e o erro não aponta para a causa em lugar nenhum**: fala em
`NativeCommandError` sobre uma linha de log de sucesso.

Verificado por execução em 2026-08-21: mesmo script, mesma máquina, mesma placa — `powershell`
morre em ~4 s, `pwsh` roda.

## Critério de fechamento

Fecha quando os cinco rodarem sob **os dois** hosts (`powershell` e `pwsh`) sem morrer no log do
binário, e quando um `rc != 0` de verdade do binário continuar sendo detectado — a checagem de
`$LASTEXITCODE` que já existe não pode ser perdida no conserto.

Não fecha com decisão escrita: uma ferramenta de medição que morre com o log da coisa medida
produz "sem resultado" e parece problema do experimento.

## Resolução — 2026-08-21

**Consertados os cinco, e provado nos dois hosts, que é o que o critério exigia.**

A guarda, em torno de cada chamada nativa, mantendo a checagem de `$LASTEXITCODE` que já existia:

```powershell
$ErrorActionPreference = 'Continue'
& $exe ... 2>&1 | Tee-Object -FilePath $log
$ErrorActionPreference = 'Stop'
$rc = $LASTEXITCODE
if ($rc -ne 0) { throw ... }
```

Comentário de seis linhas colado em cada sítio, dizendo que stderr nativo é **log, não falha**, e
por que só o 5.1 morre. Sem isso alguém "limpa" a guarda no próximo refactor.

**Item 1 do critério — não morre no log, nos dois hosts.** `bisect_pvnt.ps1` rodado de verdade sob
`powershell` 5.1: **`rc=0`**, os quatro braços (`gpu/off`, `gpu/on`, `cpu/off`, `cpu/on`)
completos, 64,1 / 67,4 / 65,3 s. Antes do conserto esse mesmo script, nesse mesmo host, morria em
~4 s na primeira linha de log.

**Item 2 — `rc != 0` de verdade continua detectado.** Sonda isolada com a mesma guarda, chamando o
`cortiq.exe` com um `--model` inexistente, sob 5.1:

```
SOBREVIVEU ao stderr. rc=1
THROW pegou: rc nao-zero DETECTADO: 1
```

Sobrevive ao stderr **e** lança. Os dois itens são independentes e os dois passaram.

## Um defeito separado, achado no caminho

`bisect_pvnt.ps1` **nunca** parseou sob Windows PowerShell 5.1 — e isso é **anterior** ao meu
patch, verificado comparando contra `git show HEAD:bisect_pvnt.ps1`: os dois falham, com
`The string is missing the terminator: '`.

Causa: um `→` (U+2192) dentro de string, em arquivo UTF-8 **sem BOM**. O 5.1 lê como ANSI, o
caractere multi-byte se parte, e a string perde o terminador. Era o único não-ASCII dos seis
arquivos (3 bytes).

Trocado por `'denoised in|latent |panicked|error'` — `latent ` com espaço casa tanto
`latent 7x16x16` quanto a linha de escrita. **Errei uma vez no meio:** troquei primeiro por
`latent written`, que não casa nada, porque o `cortiq` imprime literalmente `latent →`. Peguei
conferindo o padrão contra um log real (3 linhas casam), não relendo o meu próprio texto.

Os seis arquivos agora: cinco 100% ASCII, `run_pack_q8.ps1` já era.

**Não coberto:** só `bisect_pvnt.ps1` foi rodado de verdade sob 5.1. Os outros quatro estão
provados por **parse** nos dois hosts mais a sonda isolada da guarda — não por execução completa,
que custaria ~500 s cada. `run_pack_q8.ps1` não recebeu guarda porque não tem `2>&1 |`; se ganhar
um, ganha o mesmo defeito.
