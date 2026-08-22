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

What landed is checked against a content digest from Hugging Face -- the API's `lfs.sha256` when
the file is an LFS blob, otherwise the `ETag` -- and that check runs *before* the `.parts.json`
sidecar is removed, so a failure leaves the resume state on disk.

When Hugging Face offers neither, the file is kept and the run still returns 0. That is on
purpose, but it means the return code alone cannot tell a verified file from an unverified one:
pass a dict as `download(..., outcome={})` and it comes back carrying `verified` and a
`verification` phrase, so a caller's summary table can render the difference instead of printing
one `OK` for both.

    python tools/hf_parallel_get.py --repo Lightricks/LTX-2.5 \\
        --file diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \\
        --dest D:/ComfyUI-Models --connections 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from huggingface_hub import HfApi, get_token

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
REDIRECT_STATUS = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5

# The only host the bearer token is ever sent to. Exact hostnames, not prefixes: the previous
# `target.startswith("https://huggingface.co")` test is passed by `https://huggingface.co.evil.
# example/` and by `https://huggingface.co@evil.example/`, neither of which is huggingface.co.
# `urlsplit().hostname` lowercases, drops the port and drops any `user:pass@` userinfo, so both
# of those resolve to `evil.example` and get nothing. Keep this set to exactly the one host the
# prefix test used to admit -- `www.huggingface.co` did not match it and must not start matching.
TOKEN_HOSTS = frozenset({"huggingface.co"})

# Everything after the `?` of a URL. On the CDN that query string *is* the signature, so a
# traceback pasted into a chat used to carry a working pre-signed download link.
SIGNED_QUERY = re.compile(r"(https?://[^\s'\"?]+)\?[^\s'\"]*")

HEX = frozenset("0123456789abcdef")


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


def is_token_host(url: str) -> bool:
    """True only for an https URL whose host is exactly huggingface.co."""
    parts = urlsplit(url)
    # Scheme is part of the test because the prefix it replaces included `https://`; without it
    # a `http://huggingface.co/...` redirect would newly get the token in cleartext.
    return parts.scheme == "https" and (parts.hostname or "") in TOKEN_HOSTS


def redact(text: str) -> str:
    return SIGNED_QUERY.sub(r"\1?<redacted>", text)


def raise_for_status_redacted(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        # `from None` drops the chained original, whose own message carries the unredacted URL.
        raise httpx.HTTPStatusError(redact(str(error)), request=error.request,
                                    response=error.response) from None


def normalize_etag(value: str | None) -> str:
    tag = (value or "").strip()
    if tag.startswith(("W/", "w/")):
        tag = tag[2:].strip()
    return tag.strip('"').lower()


def file_digest(path: Path, algorithm: str) -> str:
    """sha256, or the git blob sha1 that a non-LFS file's ETag carries."""
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.sha1()
        # git hashes `blob <bytes>\0` before the content; verified here against `git hash-object`.
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class Source:
    """Resolves and re-resolves the signed CDN URL behind a hf.co resolve link."""

    def __init__(self, repo: str, filename: str, revision: str):
        self.repo = repo
        self.filename = filename
        self.revision = revision
        self.origin = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        self.token = get_token()
        self.etag: str | None = None
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
                # Take the ETag off huggingface.co's own response. Further down the chain it is
                # the CDN object's id, which is not the blob's content hash.
                #
                # TRACED, not confirmed against a live response: that `x-linked-etag` on
                # huggingface.co's own reply carries the blob's content hash is read off the
                # spec, not measured here. Nothing in the test suite makes a network request.
                if self.etag is None:
                    self.etag = (response.headers.get("x-linked-etag")
                                 or response.headers.get("etag"))
                redirects = 0
                while response.status_code in REDIRECT_STATUS:
                    redirects += 1
                    if redirects > MAX_REDIRECTS:
                        # A cycle used to spin here with no bound and no timeout.
                        raise RuntimeError(f"more than {MAX_REDIRECTS} redirects from "
                                           f"{self.origin}")
                    target = response.headers["location"]
                    # Only huggingface.co gets the token; the CDN link is already signed, and
                    # forwarding a bearer there breaks the signature.
                    headers = {"Range": "bytes=0-0"}
                    if is_token_host(target):
                        headers.update(self._headers())
                    response = client.get(target, headers=headers)
                    self._url = target
                raise_for_status_redacted(response)
            if self._url is None:
                self._url = self.origin
            return self._url

    def size(self) -> int:
        url = self.resolve()
        headers = self._headers() if is_token_host(url) else {}
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url, headers={**headers, "Range": "bytes=0-0"})
            raise_for_status_redacted(response)
            return int(response.headers["content-range"].split("/")[-1])

    def expected_digest(self) -> tuple[str, str, str] | None:
        """`(algorithm, hex, where it came from)`, or None if HF offered neither.

        Order is the ticket's: the API's `lfs.sha256` first, because that is the content hash
        git-lfs itself stores for the blob, then the `ETag`.

        **TRACED FROM THE git-lfs AND HUGGING FACE SPECS, NOT CONFIRMED AGAINST A LIVE RESPONSE.**
        The claim that an LFS blob's ETag is that same sha256, and that a plain git blob's is the
        sha1 of `blob <len>\\0<content>`, has never been run here -- no test in
        `tools/test_hf_parallel_get.py` touches the network. If the ETag turns out to be something
        else for some repo, the `algorithm` returned for that file is wrong and the verification
        below compares the right bytes against the wrong kind of digest. Settle it by pointing
        this at one small LFS file and one small non-LFS file and printing what came back.
        """
        try:
            entries = HfApi().get_paths_info(self.repo, self.filename, revision=self.revision,
                                             token=self.token)
            for entry in entries:
                lfs = getattr(entry, "lfs", None)
                if getattr(entry, "path", None) == self.filename and lfs is not None:
                    return ("sha256", lfs.sha256.lower(), "HF API lfs.sha256")
        except Exception as error:
            print(f"  (HF API file metadata unavailable: {type(error).__name__}: "
                  f"{redact(str(error))[:120]})", flush=True)

        tag = normalize_etag(self.etag)
        if set(tag) <= HEX and len(tag) == 64:
            return ("sha256", tag, "ETag")
        if set(tag) <= HEX and len(tag) == 40:
            return ("git-blob-sha1", tag, "ETag")
        return None


def fetch_chunk(source: Source, dest: Path, index: int, start: int, end: int,
                retries: int, progress: dict, lock: threading.Lock) -> int:
    for attempt in range(retries):
        # Clear this chunk's live counter before the attempt rather than at its first block: a
        # retry sleeps up to 30 s, and without this the abandoned attempt's bytes sit in the
        # readout for that whole sleep, claiming progress that is being refetched.
        with lock:
            progress["inflight"][index] = 0
        try:
            url = source.resolve()
            headers = {"Range": f"bytes={start}-{end}"}
            if is_token_host(url) and source.token:
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
                    raise_for_status_redacted(response)
                    written = 0
                    with dest.open("r+b") as handle:
                        handle.seek(start)
                        for block in response.iter_bytes(1024 * 1024):
                            handle.write(block)
                            written += len(block)
                            # `written` counts this attempt only, and the entry is *assigned*,
                            # never added to, so a retry of the same range replaces its own
                            # earlier figure instead of stacking on it. That stacking is what
                            # the deleted `partial_<index>` bookkeeping got wrong; deleting it
                            # fixed the double count and left the readout with nothing to say
                            # between chunk completions -- see the progress line in download().
                            with lock:
                                progress["inflight"][index] = written
            expected = end - start + 1
            if written != expected:
                raise OSError(f"chunk {index}: got {written} bytes, expected {expected}")
            # Whole chunks move from `inflight` to `done` in one acquisition, so a reader under
            # the same lock never sees the bytes twice and never sees them missing.
            with lock:
                progress["done"] += written
                progress["inflight"].pop(index, None)
            return index
        except Exception as error:
            if attempt == retries - 1:
                with lock:
                    progress["inflight"].pop(index, None)
                raise RuntimeError(redact(f"chunk {index}: {type(error).__name__}: "
                                          f"{error}")) from None
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError


def download(repo: str, file: str, dest_dir: Path, *, revision: str = "main",
            connections: int = 8, chunk_mb: int = 256, retries: int = 6,
            expected_size: int | None = None, outcome: dict | None = None) -> int:
    """Library entry point for the CLI above -- same body `main()` used to run inline against
    `args.*`, now against explicit parameters so a caller (e.g. `fetch_ltx25.py`) can import this
    instead of shelling out. Returns 0 on success, 130 if interrupted (matching the CLI's own exit
    code for Ctrl-C) -- the caller must check the return value, since interruption does NOT raise
    here (see the `except KeyboardInterrupt` below, unchanged from the original `main()`).
    Raises SystemExit on a server/expected-size mismatch, a short final file, or a digest that
    disagrees with Hugging Face's -- callers must catch `SystemExit`, not just `Exception`.

    Pass a dict as `outcome` to be told **whether the content was verified**, which the return
    code does not carry: rc 0 covers both "sha256 matched" and "Hugging Face offered no digest,
    so only the byte count was checked". Keys, always present once the call returns:
    `verified` (bool), `verification` (a phrase for a summary line), `bytes`, `path`.
    """
    # This record exists because a caller that prints OK on rc == 0 prints OK for a file nothing
    # verified: the WARNING below is loud in *this* function's output and invisible in the
    # caller's summary table. It is deliberately not a refusal and not a new return code --
    # whether an unverified download is acceptable is the owner's policy, and a new non-zero exit
    # would turn every `if rc:` in a wrapper into a failure for a file that is on disk and
    # probably fine. The fail-open stays open; it just stops being silent one level up.
    report = outcome if outcome is not None else {}
    report.update({"verified": False, "verification": "did not finish", "bytes": None,
                   "path": None})

    dest = (dest_dir / file).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    state_path = dest.with_suffix(dest.suffix + ".parts.json")

    source = Source(repo, file, revision)
    total = source.size()
    if expected_size and total != expected_size:
        raise SystemExit(f"server reports {total} bytes, expected {expected_size}")
    if dest.is_file() and dest.stat().st_size == total and not state_path.exists():
        # This path has never hashed anything -- it returns on size alone, and did so before the
        # digest check existed. Saying "verified" here would be the same lie one level down.
        report.update({"verification": "not rechecked (already present at the expected size)",
                       "bytes": total, "path": str(dest)})
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

    # `done` is whole chunks; `inflight` is {chunk index: bytes of the current attempt already
    # written to disk}. Both are needed for a rate that does not lie -- see the progress line.
    progress: dict = {"done": 0, "inflight": {}}
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
                # Counting only whole chunks made this number swing about 2x between prints:
                # with 8 connections and 256 MiB chunks, up to 2 GiB is on disk and uncounted at
                # any moment, so the rate sagged as `elapsed` grew and jumped back on each
                # completion. (0.53x-1.00x, reported by the round-1 review of ticket 11; not
                # re-measured here, since the shape needs a real multi-GiB fetch on the link.)
                # In-flight bytes are already written to disk, so adding them measures bytes
                # landed rather than bytes promised.
                #
                # Under the lock because `sum(...values())` iterates: a worker inserting a new
                # chunk key mid-iteration raises "dictionary changed size during iteration",
                # which would kill the download from inside its own progress line.
                with lock:
                    inflight = sum(progress["inflight"].values())
                    landed = progress["done"] + inflight
                rate = landed / elapsed / 1024**2 if elapsed else 0
                remaining = max((len(ranges) - completed) * chunk - inflight, 0)
                eta = remaining / (rate * 1024**2) / 60 if rate else 0
                # "avg", not an instantaneous rate: it is this run's bytes over this run's wall
                # clock, so it converges rather than tracking the link.
                print(f"  {completed}/{len(ranges)} chunks  {rate:6.1f} MiB/s avg  "
                      f"eta {eta:5.1f} min", flush=True)
    except KeyboardInterrupt:
        save()
        report["verification"] = "incomplete (interrupted)"
        print("\ninterrupted; rerun the same command to resume")
        return 130

    actual = dest.stat().st_size
    if actual != total:
        raise SystemExit(f"size mismatch after download: {actual} != {total}")

    # Byte count used to be the only completeness signal, and it is the one signal a
    # multi-connection resumable downloader cannot lean on: every chunk can be exactly the right
    # length and one of them still hold the wrong 256 MiB. Verify before `state_path.unlink()`,
    # so a mismatch leaves the sidecar on disk instead of discarding the resume state along with
    # the bad file.
    report.update({"bytes": total, "path": str(dest)})
    digest = source.expected_digest()
    if digest is None:
        report["verification"] = "NOT VERIFIED (no lfs.sha256 and no hex ETag from HF)"
        print("  WARNING: Hugging Face returned no lfs.sha256 and no hex ETag for this file; "
              "only the byte count was checked, the content was NOT verified", flush=True)
    else:
        algorithm, expected_hex, provenance = digest
        print(f"  verifying {algorithm} from {provenance} ...", flush=True)
        actual_hex = file_digest(dest, algorithm)
        if actual_hex != expected_hex:
            report["verification"] = f"{algorithm} MISMATCH against {provenance}"
            raise SystemExit(
                f"{algorithm} mismatch: got {actual_hex}, {provenance} says {expected_hex}. "
                f"The file is wrong; {state_path.name} was kept so a rerun resumes, but a digest "
                f"mismatch means at least one chunk recorded as done is bad -- delete that "
                f"sidecar to force a full refetch.")
        report.update({"verified": True, "verification": f"{algorithm} from {provenance}"})
        print(f"  {algorithm} OK  {actual_hex}", flush=True)

    state_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    # A bare "OK" for a file whose content nothing checked is the same sentence as an "OK" for
    # one whose sha256 matched, and a log is read long after the WARNING has scrolled away.
    print(f"\nOK  {dest}" if report["verified"]
          else f"\nOK (bytes only, {report['verification']})  {dest}")
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
