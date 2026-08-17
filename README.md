# python-docs-vi-translator

`pydocvi` translates the CPython documentation into Vietnamese gettext catalogs and audits every string it writes.

Upstream is [`tamnd/python-docs-vi`](https://github.com/tamnd/python-docs-vi), the Transifex mirror of the `python-doc/python-newest` project: 548 `.po` files, 87 008 entries, 1 711 382 English words, pinned to branch `3.15`. Output goes to [`tamnd/python-docs-vi-machine-translation`](https://github.com/tamnd/python-docs-vi-machine-translation). The mirror repo is never written to by this tool.

## What it does

The corpus is reStructuredText inside gettext, which means the strings are full of `:func:` roles, inline literals, link targets, substitutions and format specifiers. Sending that to a model and asking it to preserve the markup is how the previous pipeline produced broken cross-references at scale.

This one removes the problem before the call. Every markup span is replaced by an opaque `⟦n⟧` placeholder, the model sees prose and nothing else, the spans are restored byte-for-byte afterwards, and nine deterministic invariants reject anything that did not come back intact. Failure is per entry, not per batch, so one bad string in a batch of forty costs one string.

```
.po ──► protect ──► batch ──► model ──► parse ──► invariants ──► restore ──► memory ──► apply ──► .po
                                                      │
                                                      └── refused entries retried, then recorded and left empty
```

Everything the tool writes is marked `fuzzy`. Sphinx renders the English source for a fuzzy string, so the worst outcome a reader gets is the English they would have seen anyway. Unfuzzying is a human act on one entry at a time and there is deliberately no command for doing it in bulk.

## Status

Early. Milestones M0 through M10 are tracked as issues, each with a checklist and an exit criterion that has to be met by a real run rather than by code existing. See the [milestone issues](https://github.com/tamnd/python-docs-vi-translator/issues?q=is%3Aissue+label%3Amilestone).

## Install

```sh
uv tool install python-docs-vi-translator
pydocvi version
```

From a checkout:

```sh
make sync
uv run pydocvi version
```

## Usage

```
pydocvi
  sync        pull the upstream pin, load human translations into the memory
  classify    no-op, doctest, literal block, version marker
  glossary    mine | curate | check | diff | show | bump
  batch       build batches under the three caps
  translate   run a tier or a file through the fleet
  apply       write the memory into the content repo catalogs
  review      roundtrip | judge | calibrate | sample
  audit       run the check catalogue over a corpus
  report      coverage | quality | review | sync | usage
  tm          stats | rebuild | show | export
  stale       find what a glossary, prompt or upstream change invalidated
  queue       stats | reap | retry | drain | dead
  fleet       up | down | status | probe | bench | doctor | trace
  prompt      list | show | hash
  version
```

Every command that writes takes `--dry-run` and prints what would change rather than a count. Every command that spends model calls prints the batch count and the estimated wall clock first, and `--yes` skips the confirmation.

Exit codes are stable: `0` success, `1` a check failed, `2` usage error, `3` the fleet is unreachable.

## Development

```sh
make sync     # uv sync --locked --all-groups
make fmt      # ruff format
make lint     # ruff check
make type     # mypy --strict src/
make test     # pytest
make cover    # pytest with the coverage floor
make secrets  # refuse to ship anything key-shaped
make build    # uv build
```

Python 3.14 is the floor. There is no `from __future__ import annotations` in this codebase and there should not be one: PEP 649 made annotations lazy in 3.14, and the import is dead weight.

`invariants.py`, `segment.py`, `classify.py` and `parse.py` are pure functions over strings with no I/O, they are covered at 100 %, and they carry property tests. Those four decide what 87 008 entries look like.

Tests marked `corpus` need a local checkout of the upstream repo and skip cleanly without one. Tests marked `fleet` need a live endpoint and never run in CI.

Nothing in the test suite opens a socket, starts a tunnel or sleeps. Time is a protocol with a fake that advances only when something waits on it, subprocesses go through a runner protocol with canned answers, and the HTTP client takes a transport. That is what makes it possible to test a five-minute cooldown doubling to an hour, and a lease expiring thirty minutes after a worker was killed, in a suite that finishes in seconds.

## On keys

No key is ever written in a file. Routes name an environment variable and the value is read at call time, so a key never sits in a long-lived object that something might print. Anything key-shaped is stripped from every trace and every error before it reaches a terminal, because a trace is the thing a person pastes into an issue. `make secrets` runs in CI and refuses any tracked file containing a key-shaped string, with no allowlist: the test fixtures assemble their fake key at import rather than spelling it out.

The route file lives in the user's config directory, never in a checkout, and holds no key of its own.

The consequence is that a fresh shell has no key in it, and a health probe answers 200 without one because health needs no auth. Run `pydocvi doctor` before anything that spends calls. It names the variable that is not set and exits 3, and `fleet bench` now makes the same check itself rather than spending a route's worth of calls to discover it.

## On PEP 545

Machine output does not satisfy PEP 545 and this project does not claim it does. What the tool produces is a fuzzy reference corpus. The path to docs.python.org runs through a person reading a string, agreeing with it, and submitting it through Transifex under the Documentation Contribution Agreement, and the tool's job is to save that person typing.

The language-switcher condition is `bugs.po`, the whole of `tutorial/` and `library/functions.po` at 100 %. That set is roughly 1 700 entries, it is the first thing translated, and it is the only part of the corpus with a defined human end state.

## License

MIT. The catalogs themselves are the CPython documentation and carry their own terms.
