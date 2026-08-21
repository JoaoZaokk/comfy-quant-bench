# `core_patch.py revert` sobrescreve o arquivo atual sem checar o hash dele antes

Type: task
Status: open

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
