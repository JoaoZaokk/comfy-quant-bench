# Issue tracker: local markdown

This repo has **no git remote** (`git remote -v` is empty, checked 2026-08-22). There is no
GitHub issue list to write to, and `gh issue create` would have nowhere to land. Issues live as
markdown files under `.scratch/`, which is the convention this repo already grew on its own —
`.scratch/estado-entregavel/` was in use before this file existed, with `map.md`, a `triagem-*.md`
and one file per ticket under `issues/`.

`.scratch/` is **tracked**, not ignored. The `.gitignore` comment records why: it was ignored for
four days, during which 26 tickets and the map existed only on disk, one `git clean -xdf` from
gone.

## Conventions

- One effort per directory: `.scratch/<effort-slug>/`
- The spec, when there is one, is `.scratch/<effort-slug>/spec.md`
- Tickets are one file each at `.scratch/<effort-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
  Never a single combined tickets file — the existing effort has 29 separate files and that is the
  shape to keep.
- Triage state is a `Status:` line near the top of each ticket file. The role strings are in
  [triage-labels.md](./triage-labels.md).
- Conversation appends to the bottom under a `## Comments` heading.

### One local rule that is not in the skill template

**Every debt ticket carries its closing criterion, written before anyone looked at the result.**
This is the bench owner's standing rule, not a suggestion: a ticket may only be closed without a
code change when the criterion was written down *first*. Without a prior criterion, the decision to
close is the owner's, not an agent's. See the memory `criterio-previo`.

## When a skill says "publish to the issue tracker"

Create a file under `.scratch/<effort-slug>/` (creating the directory if needed). Do not open a
GitHub issue — there is no remote, and GitHub issues are explicitly out of scope for this bench
(memory `mapa-estado-entregavel`).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The owner normally passes the path or the number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` — the Destination / Notes / Decisions-so-far / Not-yet-specified body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question
  in the body. A `Type:` line records the ticket type (`research` / `prototype` / `grilling` /
  `task`); a `Status:` line records `claimed` / `resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it
  lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked and unclaimed;
  first by number wins.
- **Claim**: set `Status: claimed` and save *before* any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append
  a one-line gist + link to the map's Decisions-so-far in `map.md`.

## PRs as a request surface

Off. There is no remote, so there are no incoming PRs to triage.
