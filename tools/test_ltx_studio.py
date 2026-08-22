"""Tests for the two things `ltx_studio` gets wrong when it trusts the request target.

    .\\python_embeded\\python.exe -s .\\tools\\test_ltx_studio.py

No GPU, no cortiq, no checkpoint -- that is the point. `JOB.start` is replaced by a recorder
for the HTTP half, so a passing run proves the malformed POST was refused *before* anything
could have spawned the child process, not merely that the child failed afterwards. The two
unit tests that do exercise the real `Job.start` reach it through a `_launch` that raises
instead of running anything.

pytest is not installed in this interpreter and installing it is the owner's call, so this
carries its own runner.

NOT covered: nothing here starts a render. That there is still a working render behind these
gates is not tested and cannot be from this file -- it needs the 3090 and
`F:\\cortiq-cmf\\target\\release\\cortiq.exe`. The closest thing to evidence for it is the
`POST` that reaches the recorder: the endpoint accepts a good body after a bad one.
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ltx_studio  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def raises(name: str, fn, exc_type) -> None:
    try:
        fn()
    except exc_type:
        print(f"  PASS  {name}")
        return
    except BaseException as exc:  # noqa: BLE001 -- the wrong exception is the finding
        check(name, False, f"raised {type(exc).__name__}: {exc}")
        return
    check(name, False, f"no {exc_type.__name__} raised")


# --------------------------------------------------------------------------- request parsing

GOOD_BODY = {
    "prompt": "a brass trumpet on a wooden table",
    "width": 512, "height": 512, "frames": 49, "fps": 25, "seed": 1234, "steps": 8,
    "audio": False, "two_stage": False, "coop": True, "split": True, "adapter": 1,
}


def test_parse() -> None:
    print("RenderRequest.parse")
    req = ltx_studio.RenderRequest.parse(dict(GOOD_BODY))
    check("the page's own body parses", req.fps == 25 and req.adapter == 1 and req.coop is True)

    raises("empty object", lambda: ltx_studio.RenderRequest.parse({}), ValueError)
    raises("not an object", lambda: ltx_studio.RenderRequest.parse([1, 2]), ValueError)
    raises("null body", lambda: ltx_studio.RenderRequest.parse(None), ValueError)

    for key in ("prompt", "width", "fps", "adapter"):
        body = dict(GOOD_BODY)
        del body[key]
        raises(f"missing {key}", lambda b=body: ltx_studio.RenderRequest.parse(b), ValueError)

    # bool is a subclass of int; without the explicit check this was accepted as width 1.
    raises("width true", lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "width": True}),
           ValueError)
    raises("fps null (what JSON.stringify makes of NaN)",
           lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "fps": None}), ValueError)
    raises("fps as string", lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "fps": "25"}),
           ValueError)
    raises("width out of range",
           lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "width": 999999}), ValueError)
    raises("blank prompt", lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "prompt": "   "}),
           ValueError)
    raises("control byte in prompt",
           lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "prompt": "a\x0ftrumpet"}),
           ValueError)
    raises("audio as 1 rather than true",
           lambda: ltx_studio.RenderRequest.parse({**GOOD_BODY, "audio": 1}), ValueError)


# ------------------------------------------------------------------------------- Job.start

def test_start_resets_running() -> None:
    print("Job.start clears `running` on any exception")
    job = ltx_studio.Job()
    req = ltx_studio.RenderRequest.parse(dict(GOOD_BODY))

    def boom(_req):
        raise KeyError("fps")  # the exact shape of the original wedge

    job._launch = boom  # type: ignore[method-assign]
    raises("a KeyError inside _launch propagates", lambda: job.start(req), KeyError)
    check("running is back to False", job.running is False, f"running={job.running}")
    check("phase says failed", job.phase == "failed", f"phase={job.phase!r}")
    check("the reason is on the snapshot", bool(job.snapshot()["error"]))

    # And the guard against a concurrent render still refuses without clearing anything.
    job.running = True
    raises("a second render is refused", lambda: job.start(req), RuntimeError)
    check("the running render was not cancelled by the refusal", job.running is True)


# -------------------------------------------------------------------------- resolve_output

def test_resolve_output(outdir: Path) -> None:
    print("resolve_output")
    check("a real frame resolves", ltx_studio.resolve_output("frame_0000.ppm", ".ppm") is not None)
    check("wrong suffix is refused",
          ltx_studio.resolve_output("frame_0000.ppm", ".mp4") is None)
    check("a name that does not exist is refused",
          ltx_studio.resolve_output("frame_9999.ppm", ".ppm") is None)

    # Every one of these is a name that `outdir / name` would have happily accepted.
    escapes = [
        "C:/Windows/win.ini",
        "C:\\Windows\\win.ini",
        r"..\..\..\Windows\win.ini",
        "../../../Windows/win.ini",
        "..",
        ".",
        "",
        "sub/frame_0000.ppm",
        "frame_0000.ppm:$DATA",
        "\\\\server\\share\\x.ppm",
        "frame\x000000.ppm",
    ]
    for name in escapes:
        check(f"refused {name!r}", ltx_studio.resolve_output(name, ".ppm") is None)

    # The drive-letter case is the one worth stating twice: it escapes with no `..` in it,
    # which is why a browser's URL normalizer leaves it alone.
    check("C:/Windows/win.ini really is a readable file on this host, so the 404 means the "
          "gate refused it rather than the file being absent",
          Path("C:/Windows/win.ini").is_file())


# ------------------------------------------------------------------------------ over HTTP

def raw(request: bytes, timeout: float = 15.0, port: int | None = None) -> tuple[int, bytes, bytes]:
    """Send bytes verbatim; no client-side URL normalisation between us and the server.

    This is the `curl.exe --path-as-is` of the closing criterion, minus curl: urllib and most
    HTTP clients collapse `..` and re-encode the target before it reaches the wire, which
    would test the client instead of the server.
    """
    with socket.create_connection(("127.0.0.1", port or ltx_studio.PORT), timeout=timeout) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            block = sock.recv(65536)
            if not block:
                break
            chunks.append(block)
    raw_response = b"".join(chunks)
    head, _, body = raw_response.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split(b" ")[1])
    return status, head, body


def get(target: str) -> tuple[int, bytes, bytes]:
    return raw(f"GET {target} HTTP/1.0\r\nHost: 127.0.0.1:{ltx_studio.PORT}\r\n\r\n"
               .encode("latin-1"))


def post(body: str, host: str | None = None, sec_fetch_site: str | None = None,
         target: str = "/render") -> tuple[int, bytes, bytes]:
    payload = body.encode("utf-8")
    lines = [f"POST {target} HTTP/1.0"]
    lines.append(f"Host: {host if host is not None else f'127.0.0.1:{ltx_studio.PORT}'}")
    if sec_fetch_site is not None:
        lines.append(f"Sec-Fetch-Site: {sec_fetch_site}")
    lines.append(f"Content-Length: {len(payload)}")
    return raw("\r\n".join(lines).encode("latin-1") + b"\r\n\r\n" + payload)


def test_http(outdir: Path, started: list) -> None:
    print("over HTTP")

    status, _, _ = get("/frame/frame_0000.ppm")
    check("a real frame is served", status == 200, f"status={status}")
    status, _, _ = get("/frame/frame_0000.ppm?v=2")
    check("the query string is stripped before the name is used", status == 200,
          f"status={status}")

    for target in ("/frame/C:/Windows/win.ini",
                   "/frame/C:\\Windows\\win.ini",
                   "/video/C:/Windows/win.ini",
                   "/frame/../../../Windows/win.ini",
                   "/video/..\\..\\..\\Windows\\win.ini",
                   "/frame/",
                   "/video/render.mp4/../../../x.mp4"):
        status, _, _ = get(target)
        check(f"404 for {target}", status == 404, f"status={status}")

    # (c): the wedge. A bad body must be refused, and must leave the tool able to render.
    before = len(started)
    status, _, body = post("{}")
    check("POST {} is 400", status == 400, f"status={status} body={body!r}")
    check("POST {} never reached Job.start", len(started) == before,
          f"start called {len(started) - before} time(s)")
    check("nothing is left running", ltx_studio.JOB.snapshot()["running"] is False)

    for bad in ("", "not json", "[]", '{"prompt": "x"}', '{"width": 512}'):
        status, _, _ = post(bad)
        check(f"400 for body {bad!r}", status == 400, f"status={status}")
    check("no malformed body reached Job.start", len(started) == before)

    status, _, body = post(json.dumps(GOOD_BODY))
    check("a good body is accepted after the bad ones -- the tool is not wedged",
          status == 200, f"status={status} body={body!r}")
    check("and it reached Job.start exactly once", len(started) == before + 1,
          f"start called {len(started) - before} time(s)")

    # (b): origin.
    for site in ("cross-site", "same-site", "nonsense"):
        status, _, _ = post(json.dumps(GOOD_BODY), sec_fetch_site=site)
        check(f"403 for Sec-Fetch-Site: {site}", status == 403, f"status={status}")
    for site in ("same-origin", "none"):
        status, _, _ = post(json.dumps(GOOD_BODY), sec_fetch_site=site)
        check(f"200 for Sec-Fetch-Site: {site}", status == 200, f"status={status}")
    for host in ("evil.example", "attacker.test:8123", "127.0.0.1", ""):
        status, _, _ = post(json.dumps(GOOD_BODY), host=host)
        check(f"403 for Host: {host!r}", status == 403, f"status={status}")
    status, _, _ = post(json.dumps(GOOD_BODY), host=f"localhost:{ltx_studio.PORT}")
    check("200 for Host: localhost", status == 200, f"status={status}")

    status, _, _ = post(json.dumps(GOOD_BODY), target="/anything-else")
    check("404 for an unknown POST route", status == 404, f"status={status}")


# ------------------------------------------------------------------------- stalled request

def test_stalled_request_is_abandoned() -> None:
    """A body that is promised and not sent must not hold a worker thread forever.

    Two checks, and they cover different halves. The first pins the value that ships -- nothing
    else here can see it, because the second runs against a subclass. The second proves that
    setting the attribute has the effect `ltx_studio`'s comment claims on *this* interpreter,
    at 1.5s instead of 15s so the suite stays quick; the mechanism itself lives in
    socketserver/http.server, not in our code, which is exactly why it is worth executing once
    rather than reading.
    """
    print("stalled request")
    shipped = ltx_studio.Handler.timeout
    check("the shipped Handler sets a finite timeout",
          isinstance(shipped, (int, float)) and not isinstance(shipped, bool) and shipped > 0,
          f"Handler.timeout={shipped!r}")

    class _Fast(ltx_studio.Handler):
        timeout = 1.5

    httpd = ltx_studio.Server(("127.0.0.1", 0), _Fast)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        baseline = threading.active_count()
        request = (f"POST /render HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n"
                   f"Content-Length: 5000\r\n\r\nX").encode("latin-1")
        started = time.perf_counter()
        # The client's own timeout is far longer than the server's, so "the client gave up
        # first" is distinguishable from "the server let go" rather than being scored as a pass.
        with socket.create_connection(("127.0.0.1", port), timeout=20) as sock:
            sock.sendall(request)
            try:
                while sock.recv(65536):
                    pass
                how = "server closed the connection"
            except TimeoutError:
                how = "the client gave up first"
            except OSError as exc:
                # Windows RSTs a socket closed while bytes are still owed on it, so a reset is
                # the expected shape of the drop here, not a failure.
                how = f"server dropped the connection ({type(exc).__name__})"
            elapsed = time.perf_counter() - started
            check("a half-sent body is abandoned instead of blocking forever",
                  how != "the client gave up first" and elapsed < _Fast.timeout * 4,
                  f"{how} after {elapsed:.1f}s")

            # The point of the timeout is the thread, not the socket, so this is asserted with
            # the client socket still OPEN. Run with `timeout = None` put back, the server
            # releases the worker too -- but only once the client itself hangs up, which is the
            # one thing a parked connection never does. Measured 2026-08-22 with a scratch copy
            # of this block against a `timeout = None` subclass: 'the client gave up first'
            # after 6.0s, and the thread count returned to baseline only after that close.
            deadline = time.perf_counter() + 5
            while threading.active_count() > baseline and time.perf_counter() < deadline:
                time.sleep(0.05)
            check("the worker thread was released while the client still held the socket",
                  threading.active_count() <= baseline,
                  f"{threading.active_count()} threads, baseline {baseline}")

        # And the server is still a server afterwards.
        status, _, _ = raw(f"GET / HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode("latin-1"),
                           port=port)
        check("the server still answers after a stalled connection", status == 200,
              f"status={status}")
    finally:
        httpd.shutdown()
        httpd.server_close()


# ----------------------------------------------------------------------------------- runner

def main() -> int:
    try:
        with socket.create_connection(("127.0.0.1", ltx_studio.PORT), timeout=1):
            pass
    except OSError:
        pass
    else:
        # SO_REUSEADDR on Windows lets a second bind steal a live port rather than failing,
        # so refusing here is the only way to avoid testing against, or hijacking, a real
        # studio the owner has open.
        print(f"something is already listening on 127.0.0.1:{ltx_studio.PORT}; "
              "stop it and rerun. The Host check pins the port, so the test cannot move.")
        return 2

    with tempfile.TemporaryDirectory(prefix="ltx_studio_test_") as tmp:
        root = Path(tmp)
        outdir = root / "run_000000"
        outdir.mkdir()
        # A 2x2 P6 PPM, written by hand so the test does not depend on Pillow to *create*
        # one. Pillow still has to read it, which is what /frame does.
        (outdir / "frame_0000.ppm").write_bytes(b"P6\n2 2\n255\n" + bytes(range(12)))

        ltx_studio.OUTROOT = root
        ltx_studio.JOB.outdir = outdir

        started: list = []
        # The real start would spawn cortiq on the locked 3090. Recording the call is what
        # the criterion actually asks about: whether a malformed body gets this far.
        ltx_studio.JOB.start = lambda req: started.append(req)  # type: ignore[method-assign]

        httpd = ltx_studio.Server(("127.0.0.1", ltx_studio.PORT), ltx_studio.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            test_parse()
            test_start_resets_running()
            test_resolve_output(outdir)
            test_http(outdir, started)
            test_stalled_request_is_abandoned()
        finally:
            httpd.shutdown()
            httpd.server_close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all passed")
    print("NOT covered: no render was started. Whether cortiq still runs behind these gates "
          "needs the GPU and cannot be shown from here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
