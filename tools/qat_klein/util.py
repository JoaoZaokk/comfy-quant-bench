"""Pequenas peças comuns: log com hora, JSON atômico, leitura de prompts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grava_json_atomico(arq: Path, obj) -> None:
    arq = Path(arq)
    tmp = arq.with_name(arq.name + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, arq)


def le_prompts(p: Path) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def sha_texto(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]
