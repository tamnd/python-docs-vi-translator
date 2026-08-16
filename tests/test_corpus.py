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

from pydocvi import batch, catalog, classify, segment, sync

pytestmark = pytest.mark.corpus

#: Measured against tamnd/python-docs-vi branch 3.15 at da475ff.
EXPECTED_COMMIT = "da475ff9dbbd2eaaefcd7338ca2648b6b7f0c61b"
EXPECTED_FILES = 548
EXPECTED_ENTRIES = 87_008
EXPECTED_WORDS = 1_711_382
EXPECTED_CHARACTERS = 12_526_506
EXPECTED_HUMAN = 1_435

EXPECTED_KINDS = {
    "prose": 73_413,
    "noop": 5_712,
    "doctest": 2_120,
    "literal_block": 1_916,
    "version_marker": 3_847,
}
EXPECTED_BATCHES = 2_776
EXPECTED_CHANGELOG_BATCHES = 577
EXPECTED_OVERSIZED = 6
EXPECTED_LARGEST_ENTRY = 12_707


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


def test_markup_protection_round_trips_on_every_msgid(catalogs: list[catalog.Catalog]) -> None:
    """The gate M2 exists for. A placeholder that does not come back out is a
    corrupted entry, and one corrupted entry is worse than a thousand untranslated
    ones because nothing downstream flags it."""
    failures = []
    for cat in catalogs:
        for entry in cat:
            protected = segment.protect(entry.msgid)
            if segment.restore(protected.text, protected.spans) != entry.msgid:
                failures.append(f"{cat.path.name}: {entry.msgid[:60]!r}")
    assert failures == []


def test_the_corpus_uses_no_placeholder_characters_of_its_own(
    catalogs: list[catalog.Catalog],
) -> None:
    """If U+27E6 already appeared in a msgid the markers would be ambiguous."""
    markers = {segment.OPEN, segment.CLOSE}
    assert not any(markers & set(entry.msgid) for cat in catalogs for entry in cat)


def test_the_kinds_add_up_to_the_corpus(catalogs: list[catalog.Catalog]) -> None:
    counts = classify.counts([entry.msgid for cat in catalogs for entry in cat])
    measured = {
        "prose": counts.prose,
        "noop": counts.noop,
        "doctest": counts.doctest,
        "literal_block": counts.literal_block,
        "version_marker": counts.version_marker,
    }
    assert measured == EXPECTED_KINDS
    assert sum(measured.values()) == EXPECTED_ENTRIES


def test_the_batching_numbers(upstream: Path, catalogs: list[catalog.Catalog]) -> None:
    batches = batch.build(catalogs, root=upstream)
    per_file = batch.by_file(batches)
    assert len(batches) == EXPECTED_BATCHES
    assert per_file["whatsnew/changelog.po"] == EXPECTED_CHANGELOG_BATCHES
    assert not any(b.oversized for b in batches)


def test_only_a_handful_of_entries_break_a_cap_on_their_own(
    catalogs: list[catalog.Catalog],
) -> None:
    """An entry over a cap is the shape that would have forced splitting a
    ``msgid``, and splitting a ``msgid`` is not recoverable. Every one of the six
    is a code block, so the classifier keeps all of them away from a model and no
    batch in the corpus is oversized."""
    over = [
        (item.characters, classify.classify(entry.msgid))
        for cat in catalogs
        for entry, item in zip(cat, batch.items(cat), strict=True)
        if item.characters > batch.CHARACTER_CAP or item.spans > batch.SPAN_CAP
    ]
    assert len(over) == EXPECTED_OVERSIZED
    assert max(characters for characters, _ in over) == EXPECTED_LARGEST_ENTRY
    assert not any(kind.translatable for _, kind in over)
