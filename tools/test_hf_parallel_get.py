"""Offline checks for hf_parallel_get's guards. No network, no GPU, no large file.

pytest is not installed in this embedded interpreter and installing it is the owner's call, so
this carries its own runner:

    .\\python_embeded\\python.exe -s .\\tools\\test_hf_parallel_get.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hf_parallel_get as hpg  # noqa: E402


def test_token_host_is_exact_not_a_prefix() -> None:
    assert hpg.is_token_host("https://huggingface.co/repo/resolve/main/f.safetensors")
    assert hpg.is_token_host("https://huggingface.co:443/x")
    # Each of these passes the old `startswith("https://huggingface.co")` test.
    assert not hpg.is_token_host("https://huggingface.co.evil.example/x")
    assert not hpg.is_token_host("https://huggingface.co@evil.example/x")
    assert not hpg.is_token_host("https://huggingface.corp.example/x")
    assert not hpg.is_token_host("https://huggingface.co-cdn.example/x")
    # And these must stay refused, or the header would travel further than it did before.
    assert not hpg.is_token_host("http://huggingface.co/x")
    assert not hpg.is_token_host("https://www.huggingface.co/x")
    assert not hpg.is_token_host("https://cdn-lfs-us-1.hf.co/repos/aa/bb?X-Amz-Signature=deadbeef")
    assert not hpg.is_token_host("/relative/redirect")


def test_redact_drops_the_signature_but_keeps_the_host() -> None:
    text = ("Client error '403 Forbidden' for url "
            "'https://cdn-lfs-us-1.hf.co/repos/aa/bb/cc?X-Amz-Signature=deadbeef&Expires=99'")
    out = hpg.redact(text)
    assert "X-Amz-Signature" not in out and "deadbeef" not in out
    assert "cdn-lfs-us-1.hf.co/repos/aa/bb/cc?<redacted>" in out
    # An unsigned URL keeps all of itself; there is nothing to hide and the host is diagnostic.
    assert hpg.redact("https://huggingface.co/a/b") == "https://huggingface.co/a/b"


class _StubResponse:
    def __init__(self, status_code: int, headers: dict):
        self.status_code = status_code
        self.headers = headers


class _RedirectClient:
    """Always answers 302 -> the same place, i.e. the cycle that used to hang the resolve."""

    seen: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        _RedirectClient.seen.append({"url": url, "headers": dict(headers or {})})
        return _StubResponse(302, {"location": "https://cdn.example/loop", "etag": '"x"'})


def test_redirects_are_bounded_and_the_token_stops_at_hf() -> None:
    _RedirectClient.seen = []
    real_client = hpg.httpx.Client
    hpg.httpx.Client = _RedirectClient
    try:
        source = hpg.Source("Some/Repo", "a/b.safetensors", "main")
        source.token = "hf_fake_token_for_the_test"
        try:
            source.resolve()
        except RuntimeError as error:
            assert "redirects" in str(error)
        else:
            raise AssertionError("unbounded redirect chain did not raise")
    finally:
        hpg.httpx.Client = real_client

    # MAX_REDIRECTS follows plus the initial GET, then the raise. Bounded, not a hang.
    assert len(_RedirectClient.seen) == hpg.MAX_REDIRECTS + 1
    first, rest = _RedirectClient.seen[0], _RedirectClient.seen[1:]
    assert first["url"].startswith("https://huggingface.co/")
    assert "Authorization" in first["headers"]
    for call in rest:
        assert call["url"] == "https://cdn.example/loop"
        assert "Authorization" not in call["headers"], "token leaked to the CDN"


def test_etag_fallback_classifies_by_length() -> None:
    source = hpg.Source("Some/Repo", "a/b", "main")
    source.token = None

    class _OfflineApi:
        def get_paths_info(self, *args, **kwargs):
            raise OSError("no network in this test")

    def digest_with(etag):
        source.etag = etag
        return source.expected_digest()

    real_api = hpg.HfApi
    hpg.HfApi = _OfflineApi
    try:
        sha256 = "b" * 64
        assert digest_with(f'"{sha256}"') == ("sha256", sha256, "ETag")
        sha1 = "a" * 40
        assert digest_with(f'W/"{sha1}"') == ("git-blob-sha1", sha1, "ETag")
        assert digest_with('"not-a-digest"') is None
        assert digest_with(None) is None
    finally:
        hpg.HfApi = real_api


def test_file_digest_matches_git_hash_object() -> None:
    # Both values checked against the tools that define them, on 2026-08-22:
    #   git hash-object <file>            -> ce013625030ba8dba906f756967f9e9ca394464a
    #   python -c "hashlib.sha256(...)"   -> 5891b5b5...
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hello.txt"
        path.write_bytes(b"hello\n")
        assert hpg.file_digest(path, "git-blob-sha1") == \
            "ce013625030ba8dba906f756967f9e9ca394464a"
        assert hpg.file_digest(path, "sha256") == \
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


class _StubSource:
    """Stands in for the network so download()'s ordering can be exercised offline."""

    def __init__(self, payload: bytes, digest):
        self.payload = payload
        self._digest = digest

    def size(self) -> int:
        return len(self.payload)

    def expected_digest(self):
        return self._digest


def _run_download(payload: bytes, digest, tmp: Path) -> tuple[object, Path, Path, dict]:
    real_source, real_fetch = hpg.Source, hpg.fetch_chunk
    hpg.Source = lambda repo, file, revision: _StubSource(payload, digest)

    def stub_fetch(source, dest, index, start, end, retries, progress, lock):
        with dest.open("r+b") as handle:
            handle.seek(start)
            handle.write(source.payload[start:end + 1])
        with lock:
            progress["done"] += end - start + 1
        return index

    hpg.fetch_chunk = stub_fetch
    dest = tmp / "sub" / "blob.bin"
    state = dest.with_suffix(dest.suffix + ".parts.json")
    outcome: dict = {}
    try:
        try:
            result = hpg.download("R/r", "sub/blob.bin", tmp, chunk_mb=1, connections=2,
                                  outcome=outcome)
        except SystemExit as error:
            result = error
    finally:
        hpg.Source, hpg.fetch_chunk = real_source, real_fetch
    return result, dest, state, outcome


def test_dest_naming_the_file_is_not_nested_under_itself() -> None:
    """`--dest` is a directory, but three callers on 2026-08-31 passed the full target path.

    Each one produced `<path>/<basename>` -- a directory named after the file, containing a
    file of the same name. The bytes were always correct, which is exactly why it survived
    three times: nothing failed, only the path was absurd. This pins both spellings.
    """
    payload = bytes(range(256)) * 8192
    good = hpg.hashlib.sha256(payload).hexdigest()
    real_source, real_fetch = hpg.Source, hpg.fetch_chunk
    hpg.Source = lambda repo, file, revision: _StubSource(payload, ("sha256", good, "stub"))

    def stub_fetch(source, dest, index, start, end, retries, progress, lock):
        with dest.open("r+b") as handle:
            handle.seek(start)
            handle.write(source.payload[start:end + 1])
        with lock:
            progress["done"] += end - start + 1
        return index

    hpg.fetch_chunk = stub_fetch
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # A caller passing the FULL path, which is the mistake being pinned.
            alvo = Path(tmp) / "text_encoders" / "blob.bin"
            hpg.download("R/r", "blob.bin", alvo, chunk_mb=1, connections=2)
            assert alvo.is_file(), f"{alvo} deveria ser o arquivo"
            assert alvo.read_bytes() == payload
            assert not (alvo / "blob.bin").exists(), "aninhou o arquivo dentro de si mesmo"

        with tempfile.TemporaryDirectory() as tmp:
            # The documented directory form has to keep working unchanged.
            hpg.download("R/r", "sub/blob.bin", Path(tmp), chunk_mb=1, connections=2)
            assert (Path(tmp) / "sub" / "blob.bin").read_bytes() == payload
    finally:
        hpg.Source, hpg.fetch_chunk = real_source, real_fetch


def test_verification_runs_before_the_sidecar_is_removed() -> None:
    payload = bytes(range(256)) * 8192  # 2 MiB, so it splits into two 1 MiB chunks
    good = hpg.hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        result, dest, state, _ = _run_download(payload, ("sha256", good, "stub"), Path(tmp))
        assert result == 0, result
        assert dest.read_bytes() == payload
        assert not state.exists(), "sidecar should be gone once the digest agrees"

    with tempfile.TemporaryDirectory() as tmp:
        wrong = ("sha256", "0" * 64, "stub")
        result, dest, state, _ = _run_download(payload, wrong, Path(tmp))
        assert isinstance(result, SystemExit), f"a bad digest must raise, got {result!r}"
        assert "mismatch" in str(result)
        # This is the criterion's item 3: the resume state survives a failed verification.
        assert state.exists(), "sidecar was removed before/despite the digest failing"

    with tempfile.TemporaryDirectory() as tmp:
        result, dest, state, _ = _run_download(payload, None, Path(tmp))
        assert result == 0
        assert not state.exists()


def test_outcome_separates_completed_from_verified() -> None:
    """rc 0 covers both "sha256 matched" and "nothing checked the content".

    `fetch_ltx25.py`'s end-of-run table prints `OK` from `path.stat().st_size == expected`, so
    a file HF gave no digest for reads exactly like a verified one there while `download()`'s
    own WARNING has long scrolled past. This is the fact that table has to be able to read; it
    is not a refusal, and the return code is unchanged in every case below.
    """
    payload = bytes(range(256)) * 8192
    good = hpg.hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        result, dest, _, outcome = _run_download(payload, ("sha256", good, "stub"), Path(tmp))
        assert result == 0 and outcome["verified"] is True, outcome
        assert "sha256" in outcome["verification"]
        assert outcome["bytes"] == len(payload) and outcome["path"] == str(dest)

        # Same directory, second call: the file is now complete with no sidecar, which is the
        # early return. It hashes nothing -- it never did -- so it must not claim it did.
        result, _, _, again = _run_download(payload, ("sha256", good, "stub"), Path(tmp))
        assert result == 0, result
        assert again["verified"] is False, again
        assert "not rechecked" in again["verification"], again

    with tempfile.TemporaryDirectory() as tmp:
        result, _, _, outcome = _run_download(payload, None, Path(tmp))
        assert result == 0, "the no-digest case must stay a fail-open, not become a refusal"
        assert outcome["verified"] is False, outcome
        assert "NOT VERIFIED" in outcome["verification"], outcome

    with tempfile.TemporaryDirectory() as tmp:
        result, _, _, outcome = _run_download(payload, ("sha256", "0" * 64, "stub"), Path(tmp))
        assert isinstance(result, SystemExit)
        assert outcome["verified"] is False and "MISMATCH" in outcome["verification"], outcome

    # A caller that passes nothing must still work: `outcome` is optional, and the two existing
    # call sites (fetch_ltx25.py, fetch_minimax_h3.py) do not pass it.
    with tempfile.TemporaryDirectory() as tmp:
        real_source, real_fetch = hpg.Source, hpg.fetch_chunk
        hpg.Source = lambda repo, file, revision: _StubSource(payload, None)

        def stub_fetch(source, dest, index, start, end, retries, progress, lock):
            with dest.open("r+b") as handle:
                handle.seek(start)
                handle.write(source.payload[start:end + 1])
            with lock:
                progress["done"] += end - start + 1
            return index

        hpg.fetch_chunk = stub_fetch
        try:
            assert hpg.download("R/r", "sub/blob.bin", Path(tmp), chunk_mb=1) == 0
        finally:
            hpg.Source, hpg.fetch_chunk = real_source, real_fetch


def test_throughput_counts_only_kept_bytes() -> None:
    """A chunk that fails once and succeeds on retry must contribute its length once.

    Both halves of the readout are here: the total credited when the chunk completes, and the
    in-flight figure the progress line adds so the rate stops sagging between completions.
    1.5 MiB crosses the wire below and 1 MiB is kept, so any sample above 1 MiB is the old
    double count coming back.
    """
    import threading

    payload = b"z" * (1024 * 1024)
    attempts = {"n": 0}
    progress: dict = {"done": 0, "inflight": {}}
    # What download()'s progress line would have printed, sampled where it cannot normally be
    # observed: at the start of each attempt, and after each block lands.
    at_attempt_start: list[int] = []
    after_block: list[int] = []

    def landed() -> int:
        return progress["done"] + sum(progress["inflight"].values())

    class _FlakyClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, headers=None):
            attempts["n"] += 1
            at_attempt_start.append(landed())
            return _FlakyStream(attempts["n"] == 1)

    class _FlakyStream:
        def __init__(self, half: bool):
            self.half = half
            self.status_code = 200
            self.request = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            body = payload[:len(payload) // 2] if self.half else payload
            for offset in range(0, len(body), size):
                yield body[offset:offset + size]
                after_block.append(landed())

    class _OkSource:
        token = None

        def resolve(self, force: bool = False) -> str:
            return "https://cdn.example/blob"

    real_client, real_sleep = hpg.httpx.Client, hpg.time.sleep
    hpg.httpx.Client = _FlakyClient
    hpg.time.sleep = lambda seconds: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "blob.bin"
            dest.write_bytes(b"\0" * len(payload))
            hpg.fetch_chunk(_OkSource(), dest, 0, 0, len(payload) - 1, 4,
                            progress, threading.Lock())
    finally:
        hpg.httpx.Client, hpg.time.sleep = real_client, real_sleep

    assert attempts["n"] == 2, attempts
    assert progress["done"] == len(payload), \
        f"retried bytes double-counted: {progress['done']} for a {len(payload)}-byte chunk"
    assert progress["inflight"] == {}, \
        f"a completed chunk left a live counter behind: {progress['inflight']}"
    # What each of the last two can see, checked on 2026-08-22 by re-execing the module with
    # the defect patched back in rather than by argument: deleting the per-attempt reset trips
    # `at_attempt_start` only; deleting the reset *and* accumulating per block instead of
    # assigning -- the historical shape -- trips both (`after_block` peaks at 1.5 MiB). Swapping
    # assign for accumulate on its own trips neither, because within one attempt the two are
    # arithmetically the same; the reset is the whole mechanism, and `at_attempt_start` is the
    # assertion that guards it.
    assert at_attempt_start == [0, 0], \
        f"the retry did not start from zero, so the abandoned bytes were still claimed during " \
        f"the backoff sleep: {at_attempt_start}"
    assert after_block and max(after_block) <= len(payload), \
        f"the live figure exceeded the chunk's own length: {after_block}"


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as error:
            failed += 1
            print(f"FAIL  {test.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("NOT covered here: any real HTTP, the HF API path of expected_digest(), and the "
          "sha256 of a real multi-GB blob. Those need the network and a download.")
    print("Also not covered: the printed MiB/s itself. The accounting behind it is tested "
          "above; whether the readout still swings on a real 8-connection multi-GiB fetch is "
          "not, and only such a fetch settles it.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
