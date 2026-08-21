"""Download one Hugging Face file over many parallel HTTP range requests, resumable per chunk.

`hf_hub_download` uses a single connection, which on a 300-500 Mbit link tops out well below the
line rate, and its resume is per-file: a dropped connection here left three separate 6.9 / 2.9 /
18.3 GiB `.incomplete` files for the same blob, none of which continued the others.

This splits the file into fixed-size chunks and records each completed chunk in a sidecar
`<dest>.parts.json`. A drop costs at most the chunks in flight, and rerunning the same command
resumes exactly where it stopped -- including across a reboot, since the state is on disk next to
the target rather than in the HF cache.

The CDN URL that huggingface.co redirects to is signed and expires, so it is resolved once, reused
while it works, and re-resolved on 401/403. The Authorization header is sent only to
huggingface.co, never to the CDN, because forwarding it there breaks the signature.

    python tools/hf_parallel_get.py --repo Lightricks/LTX-2.5 \\
        --file diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \\
        --dest D:/ComfyUI-Models --connections 8
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from huggingface_hub import get_token

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True, help="path inside the repo")
    parser.add_argument("--dest", required=True, type=Path,
                        help="directory; the repo path is appended, so folders match ComfyUI's")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--expected-size", type=int, help="fail loudly if the server disagrees")
    return parser.parse_args()


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError


class Source:
    """Resolves and re-resolves the signed CDN URL behind a hf.co resolve link."""

    def __init__(self, repo: str, filename: str, revision: str):
        self.origin = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        self.token = get_token()
        self._url: str | None = None
        self._lock = threading.Lock()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def resolve(self, force: bool = False) -> str:
        with self._lock:
            if self._url and not force:
                return self._url
            with httpx.Client(follow_redirects=False, timeout=30) as client:
                response = client.get(self.origin, headers={**self._headers(),
                                                            "Range": "bytes=0-0"})
                while response.status_code in (301, 302, 303, 307, 308):
                    target = response.headers["location"]
                    # Only hf.co gets the token; the CDN link is already signed.
                    headers = {"Range": "bytes=0-0"}
                    if target.startswith("https://huggingface.co"):
                        headers.update(self._headers())
                    response = client.get(target, headers=headers)
                    self._url = target
                response.raise_for_status()
            if self._url is None:
                self._url = self.origin
            return self._url

    def size(self) -> int:
        url = self.resolve()
        headers = self._headers() if url.startswith("https://huggingface.co") else {}
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url, headers={**headers, "Range": "bytes=0-0"})
            response.raise_for_status()
            return int(response.headers["content-range"].split("/")[-1])


def fetch_chunk(source: Source, dest: Path, index: int, start: int, end: int,
                retries: int, progress: dict, lock: threading.Lock) -> int:
    for attempt in range(retries):
        try:
            url = source.resolve()
            headers = {"Range": f"bytes={start}-{end}"}
            if url.startswith("https://huggingface.co") and source.token:
                headers["Authorization"] = f"Bearer {source.token}"
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=180)) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in (401, 403):
                        source.resolve(force=True)  # signature expired
                        raise httpx.HTTPStatusError("expired", request=response.request,
                                                    response=response)
                    if response.status_code in RETRY_STATUS:
                        raise httpx.HTTPStatusError("retryable", request=response.request,
                                                    response=response)
                    response.raise_for_status()
                    written = 0
                    with dest.open("r+b") as handle:
                        handle.seek(start)
                        for block in response.iter_bytes(1024 * 1024):
                            handle.write(block)
                            written += len(block)
                            with lock:
                                progress["done"] += len(block)
            expected = end - start + 1
            if written != expected:
                raise OSError(f"chunk {index}: got {written} bytes, expected {expected}")
            return index
        except Exception as error:
            with lock:
                progress["done"] -= progress.get(f"partial_{index}", 0)
                progress[f"partial_{index}"] = 0
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError


def download(repo: str, file: str, dest_dir: Path, *, revision: str = "main",
            connections: int = 8, chunk_mb: int = 256, retries: int = 6,
            expected_size: int | None = None) -> int:
    """Library entry point for the CLI above -- same body `main()` used to run inline against
    `args.*`, now against explicit parameters so a caller (e.g. `fetch_ltx25.py`) can import this
    instead of shelling out. Returns 0 on success, 130 if interrupted (matching the CLI's own exit
    code for Ctrl-C) -- the caller must check the return value, since interruption does NOT raise
    here (see the `except KeyboardInterrupt` below, unchanged from the original `main()`).
    Raises SystemExit on a server/expected-size mismatch or a short final file, exactly as the CLI
    did when run as a subprocess -- callers must catch `SystemExit`, not just `Exception`.
    """
    dest = (dest_dir / file).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    state_path = dest.with_suffix(dest.suffix + ".parts.json")

    source = Source(repo, file, revision)
    total = source.size()
    if expected_size and total != expected_size:
        raise SystemExit(f"server reports {total} bytes, expected {expected_size}")
    if dest.is_file() and dest.stat().st_size == total and not state_path.exists():
        print(f"already complete: {dest} ({human(total)})")
        return 0

    chunk = chunk_mb * 1024 * 1024
    ranges = [(i, s, min(s + chunk, total) - 1)
              for i, s in enumerate(range(0, total, chunk))]

    done: set[int] = set()
    if state_path.is_file():
        try:
            saved = json.loads(state_path.read_text())
            if saved.get("total") == total and saved.get("chunk") == chunk:
                done = set(saved.get("done", []))
        except (OSError, ValueError):
            done = set()

    # Do NOT preallocate with truncate(). On a local NTFS volume that is instant, but over SMB the
    # server physically writes the whole file as zeros first -- measured at ~20 MiB/s here, which
    # is 33 minutes of pure zero-writing on a 39 GiB target before a single byte is fetched.
    # Marking the file sparse makes the extend free where the filesystem supports it; where it does
    # not, the file simply grows as chunks land, and with ordered dispatch the only gaps are the
    # few chunks currently in flight.
    if not dest.is_file():
        dest.touch()
        if os.name == "nt":
            import subprocess

            subprocess.run(["fsutil", "sparse", "setflag", str(dest)],
                           capture_output=True, check=False)

    todo = [r for r in ranges if r[0] not in done]
    print(f"{file}")
    print(f"  {human(total)} in {len(ranges)} chunks of {chunk_mb} MiB, "
          f"{len(done)} already done, {connections} connections\n", flush=True)

    progress = {"done": 0}
    lock = threading.Lock()
    started = time.perf_counter()
    completed = len(done)

    def save() -> None:
        state_path.write_text(json.dumps({"total": total, "chunk": chunk,
                                          "done": sorted(done)}), encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = {pool.submit(fetch_chunk, source, dest, i, s, e, retries,
                                   progress, lock): i for i, s, e in todo}
            for future in as_completed(futures):
                index = future.result()
                done.add(index)
                completed += 1
                save()
                elapsed = time.perf_counter() - started
                rate = progress["done"] / elapsed / 1024**2 if elapsed else 0
                remaining = (len(ranges) - completed) * chunk
                eta = remaining / (rate * 1024**2) / 60 if rate else 0
                print(f"  {completed}/{len(ranges)} chunks  {rate:6.1f} MiB/s  "
                      f"eta {eta:5.1f} min", flush=True)
    except KeyboardInterrupt:
        save()
        print("\ninterrupted; rerun the same command to resume")
        return 130

    actual = dest.stat().st_size
    if actual != total:
        raise SystemExit(f"size mismatch after download: {actual} != {total}")
    state_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    print(f"\nOK  {dest}")
    print(f"    {human(total)} in {elapsed / 60:.1f} min "
          f"({progress['done'] / elapsed / 1024**2:.1f} MiB/s this run)")
    return 0


def main() -> int:
    args = parse_args()
    return download(args.repo, args.file, args.dest, revision=args.revision,
                    connections=args.connections, chunk_mb=args.chunk_mb,
                    retries=args.retries, expected_size=args.expected_size)


if __name__ == "__main__":
    raise SystemExit(main())
