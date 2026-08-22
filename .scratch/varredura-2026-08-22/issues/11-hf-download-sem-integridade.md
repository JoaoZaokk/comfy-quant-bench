# 11 - hf_parallel_get resumes across restarts and verifies nothing but byte count

Type: task
Status: resolved
Blocked by: -
Severity: medium
Provenance: TRACED

## Problem

`tools/hf_parallel_get.py` is genuinely good at the thing it was written for -- parallel range
requests with per-chunk resume in a `<dest>.parts.json` sidecar, surviving reboots, written after a
dropped `hf_hub_download` left three separate 6.9 / 2.9 / 18.3 GiB `.incomplete` files for one blob,
none continuing the others.

Four defects around that core:

- **No integrity check at all.** Grep for `sha256|hashlib|etag|checksum|verify` over the file: no hits.
  `:225-226` treats byte length as the only completeness signal. A resumable multi-connection
  downloader with no checksum is the exact shape that produces a silently truncated 40 GiB checkpoint
  -- and this bench's whole failure history is about numbers that are wrong rather than absent.
- **The redirect guard is a string prefix test** (`:79-83`). It decides whether to forward the HF
  bearer token by testing whether the redirect URL starts with a known host string. The file's own
  docstring gets the security property right -- the token goes to `huggingface.co` and never to the
  redirected CDN, because forwarding it breaks the signature -- but a prefix test is not a host test.
  `https://huggingface.co.evil.example/` passes it.
- **The redirect loop is unbounded** (`:79`). A redirect cycle hangs the download with no timeout.
- **Retried bytes are double-counted** (`:133`), so the MiB/s and ETA it prints are optimistic by
  however much was retried.

Also `EP-11`: `:118` lets the signed CDN URL into exception text, so a traceback pasted into a chat
carries a working pre-signed download link.

## Severity note

The refuting pass put all of these at low, and that is right: there is one user, the redirect target is
`huggingface.co` in practice, and a truncated file usually fails loudly at load. They are grouped into
one ticket because they are one afternoon's work in one file, and because the missing checksum is the
one that could waste a 40 GiB re-download.

## Closing criterion (written before the fix)

Closed when:

1. the redirect guard parses the URL and compares `urlsplit(u).hostname` against an exact host set,
   never a prefix;
2. redirects are bounded (5) and the loop raises on exceeding it;
3. the download verifies what landed -- the HF API's `sha256` from the file metadata when available,
   falling back to the `ETag`, and the check runs **before** the `.parts.json` sidecar is removed, so
   a failure leaves the resume state intact;
4. throughput counts transferred-and-kept bytes, not transferred bytes;
5. the signed URL is redacted from exception text.

Item 3 is the one that closes the ticket on its own if the others slip.

## Closed 2026-08-22, commit `13fbd8c`

The criterion items are met, including item 3 -- the one that closes this ticket on its own:
verification runs **before** the `.parts.json` sidecar is removed, so a failed check leaves the
resume state intact rather than forcing a fresh multi-GiB download.

The redirect guard is now `urlsplit().hostname` against an exact host set. The reviewer hunted
specifically for anything the new guard *admits* that the old prefix test refused, and found two:
`https://HUGGINGFACE.CO/x` and `https://user:pw@huggingface.co/x`. Both resolve to the genuine
`huggingface.co`, so the bearer still reaches only the host it was always meant for.

One correction applied after review, and it is a house-rule fix rather than a criterion one: two
comments asserted ETag semantics **as fact** -- that an LFS blob's ETag is the sha256, that a plain
git blob's is the sha1 of `blob <len>\0<content>`, that `x-linked-etag` off huggingface.co carries
the content hash. Nothing in the test suite makes a network request, so none of that has ever been
run. Both now carry "traced from the spec, not confirmed against a live response" inline, because
the report is gone next session and the file is what survives.

Two items carried to ticket 16: the throughput display oscillation, and the no-digest fail-open
where `fetch_ltx25`'s summary table prints `OK` for a file nothing verified.
