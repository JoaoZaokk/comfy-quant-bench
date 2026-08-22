# 03 - ltx_studio serves any file by URL, wedges on a bad POST, and takes no origin check

Type: task
Status: resolved
Blocked by: -
Severity: medium
Provenance: TRACED, with one item needing a run

## Problem

`tools/ltx_studio.py` opens a stdlib HTTP server on `127.0.0.1:8123` (`:462`). Three defects, one root
cause: the request target is trusted.

**a) Path join, `:404-417`.** `name = self.path[len("/video/"):]` then `path = JOB.outdir / name`.
`pathlib` join semantics mean a right-hand operand carrying a drive replaces the base entirely. The
only gates are `is_file()` and a suffix check. Grep for `resolve`, `relative_to`, `commonpath`,
`basename` over the file found nothing. So `GET /frame/C:/Users/joaoz/x.ppm` returns that file,
rendered to PNG. `BaseHTTPRequestHandler` does not percent-decode, so `%2e%2e` does not work -- but a
literal drive letter needs no `..` at all, which means a browser's URL normalizer passes it through.

**b) No origin check, `:398-434`.** No auth, no CSRF token, no `Origin` / `Referer` / `Sec-Fetch-Site`
/ `Host` inspection. The page's own JS at `:348` posts without setting `Content-Type`, so `fetch`
stamps it `text/plain` -- a CORS-simple request, no preflight. Any page open in the owner's browser can
start a render on the locked card. ComfyUI carries a mitigation for exactly this
(`ComfyUI/server.py:159-197`, `create_origin_only_middleware`); ltx_studio has none.

**c) Wedge on a bad POST, `:96-107` + `:429-434`.** `Job.start` sets `self.running = True` at `:101`
*before* reading `opts["fps"]` at `:106`. A missing key raises `KeyError`; `do_POST` catches only
`RuntimeError`, so nothing resets `running` and the worker thread was never started. Every later
render gets 409 and `/progress` reports `running: true, phase: "starting"` forever.

## Severity note

The refuting pass corrected this down from high. Loopback bind plus a single-user bench is not a
public service, and (b) requires the owner to have a hostile page open. It stays a real finding
because (c) needs no attacker at all -- a field-name change in the page wedges the tool -- and because
(a)+(b) compose: with `Host` unchecked, a rebinding page turns (a) from an existence oracle into an
arbitrary read.

## Closing criterion (written before the fix)

Closed when:

1. `GET /frame/<name>` and `/video/<name>` reject any `name` containing `/`, a backslash, `:`, or
   equal to `.`/`..`, **and** verify `(outdir / name).resolve().is_relative_to(OUTROOT.resolve())`
   before opening; a query string is stripped first;
2. `do_POST` validates the request body into a typed record before any state is touched, and
   `Job.start` resets `running` on any exception, not just `RuntimeError`;
3. `do_POST` rejects a request whose `Sec-Fetch-Site` is not `same-origin`/`none`, and whose `Host`
   is not `127.0.0.1:8123` or `localhost:8123`;
4. a `curl.exe --path-as-is` against a file outside the run directory returns 404, and a `POST {}`
   returns 400 and leaves the tool able to render afterwards.

Item 4 is the run that settles it, and it needs no GPU: the POST can be rejected before `cortiq` is
ever spawned.

## Related, same file, same fix window

`CACHE-07` / `EP-10`: the run directory is stamped `%H%M%S` with no date, so two renders 24 h apart
collide. `CACHE-08`: the `cortiq` child is never owned or terminated, so closing the server orphans a
GPU process -- and CLAUDE.md already records a 20,578 MiB orphan on this bench. `EP-15`: `JOB.outdir`
is read outside the lock in both file routes.

## Closed 2026-08-22, commit `13fbd8c`

All four criterion items met and independently re-run by the reviewer with the real
`C:/Windows/System32/curl.exe` against a live server.

- **Traversal:** eleven escape names refused, including `C:/Windows/win.ini`,
  `C:\Windows\win.ini`, `\\server\share\x.ppm`, `frame_0000.ppm:$DATA` and a NUL-bearing name;
  seven traversal targets 404 over HTTP while a legitimate frame returns 200; `?v=2` still 200,
  so the query strip works. The premise was confirmed separately by execution:
  `Path(r'F:/cortiq/studio/run_120000') / 'C:/Windows/win.ini'` evaluates to `C:\Windows\win.ini`
  with `.is_file() == True` -- on Windows a drive letter needs no `..` to escape, which is why the
  fix rejects separators and colons outright rather than reaching for `unquote()`.
- **The wedge:** `RenderRequest.parse` runs before any state is touched, and `Job.start` clears
  `running` on `BaseException`, not `RuntimeError` -- the wedge was a `KeyError` and the next one
  will be something else. Five malformed bodies returned 400 and the `cortiq` spawn recorder was
  called **zero** times, so the refusal happens before a child could exist.
- **Origin:** `cross-site` / `same-site` / nonsense -> 403; `Host: evil.example` and bare
  `127.0.0.1` -> 403.

One correction applied after review: `Host` was compared case-sensitively, so
`http://LOCALHOST:8123` returned 403 on render while the page itself loaded. A control that reads
as a broken button is one somebody switches off. Casefolded.

One item is a **decision, not a defect**, and it is the owner's: an *absent* `Sec-Fetch-Site` is
allowed. Rejecting it would make criterion 4's plain-`curl` `POST {}` return 403 instead of the 400
the criterion asks for, so items 3 and 4 only reconcile if absent means allowed. Carried to
ticket 16.
