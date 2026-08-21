# Cinco scripts de medição morrem quando o binário escreve no stderr

Type: task
Status: open

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
