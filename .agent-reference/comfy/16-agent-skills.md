# Rastreadores e habilidades

> Referência preservada do CLAUDE.md original, linhas 1424–1446, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Agent skills

Written by `/setup-matt-pocock-skills` on 2026-08-22. These three files are what the engineering
skills read as input; edit them directly rather than re-running the setup.

### Issue tracker

Local markdown under `.scratch/`, one file per ticket — this repo has **no git remote**, so there
are no GitHub issues and `gh` has nowhere to write. Every debt ticket carries its closing criterion,
written before anyone looked at the result. See [docs/agents/issue-tracker.md](../../docs/agents/issue-tracker.md).

### Triage labels

The five canonical roles, unchanged, recorded as a `Status:` line in each ticket file.
`ready-for-human` is load-bearing here: package installs, GPU windows and anything touching WSL are
the owner's call and cannot be delegated. See [docs/agents/triage-labels.md](../../docs/agents/triage-labels.md).

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the root, both created lazily. The provenance
rule extends to them: a glossary entry derived from reading rather than running says so in the
entry. See [docs/agents/domain.md](../../docs/agents/domain.md).

