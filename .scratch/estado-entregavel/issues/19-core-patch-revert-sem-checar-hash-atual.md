# `core_patch.py revert` sobrescreve o arquivo atual sem checar o hash dele antes

Type: task
Status: resolved

## Question

`tools/core_patch.py`'s docstring promete: "Every backup records the file's SHA-256 and a
timestamp, so a later ComfyUI update that rewrites the same file is detected instead of being
silently reverted over." Essa detecção existe em `command_backup` (compara o hash atual do
arquivo com o hash gravado antes de sobrescrever o backup). `command_revert` não tem a mesma
guarda:

```python
def command_revert(args) -> int:
    key, path = resolve(args.target)
    ledger = load_ledger()
    entry = ledger.get(key)
    if entry is None:
        raise SystemExit(f"No backup tracked for {key}")
    backup_path = PORTABLE_ROOT / entry["backup"]
    if not backup_path.is_file():
        raise SystemExit(f"Backup file is gone: {backup_path}")
    shutil.copy2(backup_path, path)                       # sobrescreve `path` sem checar o hash
    restored = sha256(path)                                # só confere DEPOIS
    if restored != entry["original_sha256"]:
        raise SystemExit(...)
```

A checagem pós-restauração é tautológica: confere se o que acabou de copiar bate com o próprio
backup — que sempre bate, porque acabou de ser copiado dele. Não confere se o arquivo que estava
**instalado antes do revert** já tinha sido reescrito por um update do ComfyUI. Um `revert` depois
de um `update_comfyui.bat` devolve um `comfy/ops.py` de versões atrás, e o comando imprime
"Reverted" como se nada tivesse dado errado.

`AUDITORIA_2026-08-18.md:36` (item 19) apontou isto. Não está na lista de correções aplicadas em
2026-08-18 (seção 6) — reconferido por leitura direta de `command_revert` (linhas 144-157) em
2026-08-21, o defeito segue lá.

Hoje `command_status` (`W4A4_PROGRESS.md:214`) reporta "No ComfyUI core file is tracked. Core is
untouched." — ninguém usou o `backup`/`revert` neste projeto ainda, então não morde hoje. É
prevenção, não incêndio.

## Critério de fechamento

Fecha quando `command_revert` calcular `sha256(path)` do arquivo **antes** de sobrescrevê-lo e
comparar contra `entry["original_sha256"]` — se divergir (arquivo foi tocado por outra coisa desde
o backup), recusar com uma mensagem que diga isso, em vez de sobrescrever silenciosamente. A
checagem pós-cópia pode continuar existindo (detecta corrupção do backup em si), mas não substitui
a checagem prévia.

Não fecha com decisão escrita: é uma condição de corrida entre o backup e o revert que só o código
fecha.

## Resolução

`command_revert` (`tools/core_patch.py`) agora calcula `sha256(path)` do arquivo instalado **antes**
de chamar `shutil.copy2`, e compara contra `entry["original_sha256"]` — o mesmo idioma que
`command_backup` já usava. Se divergir (ou se o arquivo instalado sumiu), recusa com
`SystemExit` citando os dois hashes (ou a ausência do arquivo) em vez de sobrescrever, salvo
`--force` (novo argumento em `revert`, adicionado ao `argparse`). A checagem pós-cópia existente
(linha ~177 depois da edição) foi mantida — ela continua tautológica sobre o próprio backup, mas
serve para detectar corrupção do backup, não substitui a checagem prévia.

LIDO: `tools/core_patch.py` inteiro (172 linhas antes da edição) antes de editar; só as funções
`parse_args` (subparser `revert`) e `command_revert` foram tocadas.

EXECUTADO (não só lido):
- `F:\COMFY_PORTABLE\python_embeded\python.exe -s -m py_compile tools\core_patch.py` -> exit 0.
- Teste próprio em
  `C:\Users\joaoz\AppData\Local\Temp\claude\F--COMFY-PORTABLE\f8b3a3e8-2bcc-442f-9487-5b4aebcfe1f2\scratchpad\test_core_patch_revert_hash_guard.py`,
  rodado com
  `F:\COMFY_PORTABLE\python_embeded\python.exe -s <caminho do teste>`. O teste monkeypatcha
  `PORTABLE_ROOT`/`COMFY_ROOT`/`BACKUP_ROOT`/`LEDGER` do módulo importado para uma árvore fake sob
  `tempfile.mkdtemp()` — nunca usa nem escreve em `F:\COMFY_PORTABLE\ComfyUI` nem no
  `F:\COMFY_PORTABLE\core_patches` real (confirmado depois, via timestamps: o `core_patches/` real
  é de 16/08, intocado por esta sessão). 12 asserções, todas PASS:
  1. backup + revert sem alteração -> sucede sem `--force`.
  2. arquivo alterado desde o backup (simula update do ComfyUI) -> `revert` sem `--force` recusa
     via `SystemExit`, mensagem cita os dois hashes, arquivo fica intocado.
  3. mesmo caso 2, com `--force` -> revert prossegue e restaura o conteúdo original.
  4. arquivo instalado ausente no momento do revert -> recusa sem `--force`; com `--force`,
     restaura.
  5. `--force` de fato chega em `parse_args()` no subcomando `revert` (nível argparse).

NÃO COBERTO por este teste ou por esta correção:
- `command_backup`, `command_diff`, `command_status` — não tocados, não reexercitados em detalhe
  (só `command_backup` foi chamado incidentalmente como setup do teste).
- Nenhum arquivo real do ComfyUI foi tocado nesta sessão — comando_revert/backup contra arquivo de
  verdade foi explicitamente proibido pelo ticket e não rodou.
- Não há teste de que a mensagem de recusa chega formatada do jeito exato no `stderr` do CLI real
  (subprocess); o teste chama `command_revert` in-process com um `argparse.Namespace` construído à
  mão, não via `sys.argv` completo (exceto o teste 5, que só cobre o parsing do `--force`, não a
  execução completa via CLI).
- Condição de corrida *durante* o `shutil.copy2` em si (TOCTOU entre o `sha256(path)` de checagem e
  a cópia) não é fechada — está fora do escopo que o ticket descreveu (a corrida é
  backup-vs-revert, não dentro do próprio revert).
