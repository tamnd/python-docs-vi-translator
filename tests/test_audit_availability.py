"""``A01`` to ``A05``: what is missing, and whether anybody wrote down why.

The group about absence. Every other check reads an entry and asks whether it is
right; these ask about the entries that are not there, so every test below has
to build the absence rather than a wrong string.
"""

from pathlib import Path

from conftest import catalog_of, corpus_of, entry, findings, machine_segment
from pydocvi.audit import availability
from pydocvi.catalog import Entry
from pydocvi.memory import Memory, Segment
from pydocvi.queue import Job, Stage, State

COVERAGE = '# Coverage\n\n<!-- counts: {{"1": {done}}} -->\n'


def died(where: Path, **overrides: object) -> Path:
    """One job in the translate stage's dead letter queue.

    Written as a file rather than through :class:`~pydocvi.queue.Queue`, because
    what the check reads is the directory and driving a job into ``dead`` takes
    three failed claims and a lease clock.
    """
    values: dict[str, object] = {
        "id": "a1b2c3d4",
        "stage": Stage.TRANSLATE,
        "payload": {"file": "library/one.po", "segments": ["s1", "s2"]},
        "attempts": 3,
        "error": "1 entries refused",
    }
    values.update(overrides)
    job = Job(**values)  # type: ignore[arg-type]
    target = where / str(Stage.TRANSLATE) / str(State.DEAD)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{job.id}.json").write_text(job.as_json(), encoding="utf-8")
    return where


class TestA01:
    """Coverage is the number this project is judged by, and it is quoted in the
    README and in every milestone comment."""

    def test_a_recount_that_disagrees_with_the_report_is_found(self) -> None:
        one = catalog_of(entry("One.", "Một."), entry("Two.", ""), name="tutorial/one.po")
        corpus = corpus_of(one, coverage=COVERAGE.format(done=90))
        found = findings(availability.a01_coverage_is_current, corpus)
        assert len(found) == 1
        assert "a recount finds 1" in found[0].detail

    def test_a_report_that_matches_is_clean(self) -> None:
        one = catalog_of(entry("One.", "Một."), entry("Two.", ""), name="tutorial/one.po")
        corpus = corpus_of(one, coverage=COVERAGE.format(done=1))
        assert findings(availability.a01_coverage_is_current, corpus) == []

    def test_a_report_with_no_machine_readable_counts_is_found(self) -> None:
        corpus = corpus_of(catalog_of(entry("One.", "Một.")), coverage="# Coverage\n")
        assert len(findings(availability.a01_coverage_is_current, corpus)) == 1

    def test_a_tier_the_report_leaves_out_is_found(self) -> None:
        one = catalog_of(entry("One.", "Một."), name="library/one.po")
        corpus = corpus_of(one, coverage=COVERAGE.format(done=1))
        found = findings(availability.a01_coverage_is_current, corpus)
        assert any("is not reported" in item.detail for item in found)

    def test_a_run_with_no_coverage_report_says_nothing(self) -> None:
        assert findings(availability.a01_coverage_is_current, corpus_of()) == []


class TestA02:
    """Dying is allowed. Dying without a line in a report is not."""

    def test_a_dead_job_nobody_reported_is_found(self, tmp_path: Path) -> None:
        corpus = corpus_of(queue=died(tmp_path), quality="# Quality\n")
        found = findings(availability.a02_dead_jobs_are_reported, corpus)
        assert len(found) == 1
        assert "a1b2c3d4" in found[0].detail

    def test_a_dead_job_the_report_names_is_accounted_for(self, tmp_path: Path) -> None:
        corpus = corpus_of(queue=died(tmp_path), quality="# Quality\n\njob a1b2c3d4 died on ...\n")
        assert findings(availability.a02_dead_jobs_are_reported, corpus) == []

    def test_a_job_that_died_with_no_reason_still_says_so(self, tmp_path: Path) -> None:
        corpus = corpus_of(queue=died(tmp_path, error=None))
        assert findings(availability.a02_dead_jobs_are_reported, corpus)[0].got == (
            "no reason recorded"
        )

    def test_a_run_with_no_queue_says_nothing(self) -> None:
        assert findings(availability.a02_dead_jobs_are_reported, corpus_of()) == []


class TestA03:
    """An empty msgstr with a machine record in the memory is the shape of a
    lost write: the translation exists, it was paid for, and the catalog it
    belongs in never got it."""

    def test_a_translation_that_never_reached_the_catalog_is_found(self) -> None:
        one = catalog_of(entry("Return a list."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.")])
        found = findings(
            availability.a03_no_untranslated_with_a_record, corpus_of(one, memory=memory)
        )
        assert len(found) == 1
        assert found[0].got == "Trả về một danh sách."

    def test_an_entry_nobody_has_translated_yet_is_clean(self) -> None:
        """Most of the corpus is this, and a check that fired on it would fire
        eighty thousand times on every run."""
        corpus = corpus_of(catalog_of(entry("Return a list.")), memory=Memory([]))
        assert findings(availability.a03_no_untranslated_with_a_record, corpus) == []

    def test_a_persons_translation_is_not_a_lost_write(self) -> None:
        """A person can retract a translation and the memory keeps the record.
        Reapplying it would be overruling them."""
        one = catalog_of(entry("Return a list."))
        theirs = Entry(msgid="Return a list.", msgstr="Trả về một danh sách.")
        memory = Memory([Segment.from_entry(theirs, source="human")])
        corpus = corpus_of(one, memory=memory)
        assert findings(availability.a03_no_untranslated_with_a_record, corpus) == []

    def test_a_translated_entry_is_not_this_check_s_business(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.")])
        corpus = corpus_of(one, memory=memory)
        assert findings(availability.a03_no_untranslated_with_a_record, corpus) == []


class TestA04:
    """Scattered deaths are individual hard strings. A hundred in one file is
    something about that file."""

    def test_a_file_over_the_ceiling_is_found(self, tmp_path: Path) -> None:
        one = catalog_of(*(entry(f"{n}.") for n in range(50)))
        corpus = corpus_of(one, queue=died(tmp_path))
        found = findings(availability.a04_dead_per_file, corpus)
        assert len(found) == 1
        assert "2 of 50 entries dead" in found[0].detail

    def test_a_file_under_the_ceiling_is_clean(self, tmp_path: Path) -> None:
        one = catalog_of(*(entry(f"{n}.") for n in range(500)))
        corpus = corpus_of(one, queue=died(tmp_path))
        assert findings(availability.a04_dead_per_file, corpus) == []

    def test_a_file_the_corpus_does_not_have_is_not_divided_by_zero(self, tmp_path: Path) -> None:
        queue = died(tmp_path, payload={"file": "gone.po", "segments": ["s1"]})
        assert findings(availability.a04_dead_per_file, corpus_of(queue=queue)) == []


class TestA05:
    """A refusal rate is the pipeline criticising itself, and a high one is
    information about the prompt rather than about the entries."""

    def test_a_heavily_refused_batch_is_named(self, tmp_path: Path) -> None:
        corpus = corpus_of(queue=died(tmp_path), quality="# Quality\n")
        found = findings(availability.a05_refusal_rates_are_reported, corpus)
        assert len(found) == 1
        assert "refused 1 of 2 entries (50%)" in found[0].detail

    def test_a_batch_the_report_names_is_accounted_for(self, tmp_path: Path) -> None:
        corpus = corpus_of(queue=died(tmp_path), quality="job a1b2c3d4 refused most of a batch\n")
        assert findings(availability.a05_refusal_rates_are_reported, corpus) == []

    def test_a_death_that_was_not_a_refusal_is_a02_s_business(self, tmp_path: Path) -> None:
        queue = died(tmp_path, error="connection reset")
        assert findings(availability.a05_refusal_rates_are_reported, corpus_of(queue=queue)) == []

    def test_a_light_refusal_rate_is_not_worth_naming(self, tmp_path: Path) -> None:
        payload = {"file": "library/one.po", "segments": [f"s{n}" for n in range(40)]}
        queue = died(tmp_path, payload=payload, error="3 entries refused")
        assert findings(availability.a05_refusal_rates_are_reported, corpus_of(queue=queue)) == []
