"""The gate for everything downstream.

A writer that reformats produces 548 files of churn on the first run and buries
every real change afterwards. These tests run against a real checkout of the
upstream mirror and skip cleanly without one.

The counts are asserted as exact numbers rather than as ranges. When upstream
moves they will fail, and the right response is to look at what moved and update
the numbers here, because a count nobody re-measured is the thing that turns a
fact about one branch into a wrong assumption about the next.
"""

from pathlib import Path

import pytest

from pydocvi import catalog, sync

pytestmark = pytest.mark.corpus

#: Measured against tamnd/python-docs-vi branch 3.15 at da475ff.
EXPECTED_COMMIT = "da475ff9dbbd2eaaefcd7338ca2648b6b7f0c61b"
EXPECTED_FILES = 548
EXPECTED_ENTRIES = 87_008
EXPECTED_WORDS = 1_711_382
EXPECTED_CHARACTERS = 12_526_506
EXPECTED_HUMAN = 1_435


@pytest.fixture(scope="module")
def catalogs(upstream: Path) -> list[catalog.Catalog]:
    return sync.read_corpus(upstream)


def test_the_checkout_is_the_pinned_commit(upstream: Path) -> None:
    if sync.head_commit(upstream) != EXPECTED_COMMIT:
        pytest.skip("checkout is not at the pinned commit, counts would be about something else")


def test_every_catalog_round_trips_byte_for_byte(upstream: Path) -> None:
    differing = []
    for path in catalog.walk(upstream):
        original = path.read_text(encoding="utf-8")
        if catalog.render(catalog.parse(original, path=path)) != original:
            differing.append(path.relative_to(upstream).as_posix())
    assert differing == []


def test_the_corpus_measures_what_the_pin_says(catalogs: list[catalog.Catalog]) -> None:
    files, entries, words, characters, translated = sync.measure(catalogs)
    assert (files, entries, words, characters, translated) == (
        EXPECTED_FILES,
        EXPECTED_ENTRIES,
        EXPECTED_WORDS,
        EXPECTED_CHARACTERS,
        EXPECTED_HUMAN,
    )


def test_segment_ids_are_unique_within_a_file(catalogs: list[catalog.Catalog]) -> None:
    """Two entries in one file with the same id would mean losing one of them."""
    clashing = []
    for cat in catalogs:
        seen: set[str] = set()
        for entry in cat:
            if entry.id in seen:
                clashing.append(f"{cat.path.name}: {entry.msgid[:60]!r}")
            seen.add(entry.id)
    assert clashing == []


def test_human_translations_are_loaded_as_human(catalogs: list[catalog.Catalog]) -> None:
    segments = sync.human_segments(catalogs)
    assert len(segments) == EXPECTED_HUMAN
    assert {s.source for s in segments} == {"human"}
    assert all(s.msgstr for s in segments)
