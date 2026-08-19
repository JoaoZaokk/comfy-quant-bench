"""Back up, inspect and revert edits to ComfyUI core files.

Every backup records the file's SHA-256 and a timestamp, so a later ComfyUI update that
rewrites the same file is detected instead of being silently reverted over.

    core_patch.py backup comfy/ops.py --note "convrot dtype dequant"
    core_patch.py status
    core_patch.py diff comfy/ops.py
    core_patch.py revert comfy/ops.py
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = PORTABLE_ROOT / "ComfyUI"
BACKUP_ROOT = PORTABLE_ROOT / "core_patches"
LEDGER = BACKUP_ROOT / "ledger.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup", help="snapshot a core file before editing it")
    backup.add_argument("target", help="path relative to ComfyUI/, e.g. comfy/ops.py")
    backup.add_argument("--note", default="", help="why this file is being changed")

    sub.add_parser("status", help="show every tracked file and whether it still matches its backup")

    show = sub.add_parser("diff", help="unified diff of backup vs current")
    show.add_argument("target")

    revert = sub.add_parser("revert", help="restore a tracked file from its backup")
    revert.add_argument("target")

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {}


def save_ledger(ledger: dict) -> None:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def resolve(target: str) -> tuple[str, Path]:
    key = target.replace("\\", "/").lstrip("./")
    path = (COMFY_ROOT / key).resolve()
    if COMFY_ROOT.resolve() not in path.parents:
        raise SystemExit(f"Refusing to touch a path outside ComfyUI/: {path}")
    return key, path


def command_backup(args) -> int:
    key, path = resolve(args.target)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    ledger = load_ledger()
    entry = ledger.get(key)
    current = sha256(path)
    if entry and entry["original_sha256"] == current:
        print(f"Already backed up and unmodified: {key}")
        return 0
    if entry:
        print(f"WARNING: {key} already has a backup from {entry['backed_up_at']} and has changed since.")
        print("Keeping the original backup; not overwriting it. Use revert first if that is what you want.")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_ROOT / "backups" / f"{key.replace('/', '__')}.{stamp}.bak"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)

    ledger[key] = {
        "backup": str(backup_path.relative_to(PORTABLE_ROOT)),
        "original_sha256": current,
        "backed_up_at": stamp,
        "note": args.note,
    }
    save_ledger(ledger)
    print(f"Backed up {key} -> {backup_path.relative_to(PORTABLE_ROOT)}")
    print(f"sha256 {current}")
    return 0


def command_status(args) -> int:
    ledger = load_ledger()
    if not ledger:
        print("No ComfyUI core file is tracked. Core is untouched.")
        return 0
    for key, entry in sorted(ledger.items()):
        _, path = resolve(key)
        if not path.is_file():
            state = "MISSING"
        elif sha256(path) == entry["original_sha256"]:
            state = "unmodified"
        else:
            state = "MODIFIED"
        print(f"{state:12} {key}  (backup {entry['backed_up_at']})")
        if entry["note"]:
            print(f"{'':12} note: {entry['note']}")
    return 0


def command_diff(args) -> int:
    key, path = resolve(args.target)
    ledger = load_ledger()
    entry = ledger.get(key)
    if entry is None:
        raise SystemExit(f"No backup tracked for {key}")
    backup_path = PORTABLE_ROOT / entry["backup"]
    original = backup_path.read_text(encoding="utf-8").splitlines(keepends=True)
    current = path.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = list(difflib.unified_diff(original, current, fromfile=f"{key} (backup)", tofile=f"{key} (current)"))
    if not diff:
        print(f"No difference: {key}")
        return 0
    sys.stdout.writelines(diff)
    return 0


def command_revert(args) -> int:
    key, path = resolve(args.target)
    ledger = load_ledger()
    entry = ledger.get(key)
    if entry is None:
        raise SystemExit(f"No backup tracked for {key}")
    backup_path = PORTABLE_ROOT / entry["backup"]
    if not backup_path.is_file():
        raise SystemExit(f"Backup file is gone: {backup_path}")
    shutil.copy2(backup_path, path)
    restored = sha256(path)
    if restored != entry["original_sha256"]:
        raise SystemExit(f"Restored file hash {restored} does not match recorded {entry['original_sha256']}")
    print(f"Reverted {key} from {entry['backup']}")
    return 0


def main() -> int:
    args = parse_args()
    return {
        "backup": command_backup,
        "status": command_status,
        "diff": command_diff,
        "revert": command_revert,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
