"""The two generated reports.

Most of these are about one number being kept apart from another. The whole
reason this module exists is that "done" and "translated" are different
quantities, and every test below that looks pedantic is guarding that
difference.
"""

from pathlib import Path

import pytest

from conftest import catalog_of, corpus_of, entry
from pydocvi import apply, report
from pydocvi.audit.model import counts
from pydocvi.catalog import Entry
from pydocvi.glossary import Glossary, Term
from pydocvi.queue import Job, Stage, State
from pydocvi.translate import Tally

README = "# Vietnamese\n\n<!-- generated: coverage -->\n<!-- /generated: coverage -->\n\nProse.\n"


def machine(msgid: str, msgstr: str) -> Entry:
    """An entry as ``apply`` writes one: fuzzy, with a provenance comment."""
    return entry(msgid, msgstr, comments=(f"{apply.MARKER} model=gpt-5 run=2026-08-15T04:00Z",))


def copied(msgid: str) -> Entry:
    """A passthrough, which carries the kind rather than a model."""
    return entry(msgid, msgid, comments=(f"{apply.MARKER} {apply.PASSTHROUGH_FIELD}noop",))


def reviewed(msgid: str, msgstr: str) -> Entry:
    """Somebody's work. Not fuzzy, which is the only mark a reviewer leaves."""
    return Entry(msgid=msgid, msgstr=msgstr)


class TestWrittenAs:
    """The five columns, and what puts an entry in each of them."""

    def test_an_empty_msgstr_is_untranslated(self) -> None:
        assert report.written_as(entry("Return a list.")) is report.Written.UNTRANSLATED

    def test_a_provenance_comment_makes_it_machine(self) -> None:
        assert report.written_as(machine("One.", "Một.")) is report.Written.MACHINE

    def test_a_passthrough_field_makes_it_passthrough(self) -> None:
        """Read off the same comment, so the order of the two rules matters."""
        assert report.written_as(copied(">>> x = 1")) is report.Written.PASSTHROUGH

    def test_an_unfuzzy_entry_with_no_marker_is_a_persons_work(self) -> None:
        assert report.written_as(reviewed("One.", "Một.")) is report.Written.HUMAN

    def test_a_fuzzy_entry_with_no_marker_came_from_before_this_tool(self) -> None:
        """The 2026 Google output. Not ours, and not anybody's judgement."""
        assert report.written_as(entry("One.", "Một.")) is report.Written.LEGACY


class TestCount:
    def test_done_and_translated_are_different_numbers(self) -> None:
        """The distinction the whole module exists for."""
        count = report.Count()
        for one in (machine("One.", "Một."), copied(">>> x"), reviewed("Two.", "Hai.")):
            count.add(one)
        assert count.done == 3
        assert count.translated == 1

    def test_words_are_counted_on_the_english_in_every_column(self) -> None:
        """So that words done and words left add up to one corpus."""
        count = report.Count()
        count.add(entry("one two three"))
        assert count.words[report.Written.UNTRANSLATED] == 3

    def test_an_empty_section_divides_by_nothing(self) -> None:
        assert report.Count().share(report.Written.HUMAN) == 0.0


class TestByTier:
    def test_a_file_lands_in_its_tier(self) -> None:
        corpus = corpus_of(catalog_of(machine("One.", "Một."), name="tutorial/one.po"))
        assert report.by_tier(corpus)[1].entries[report.Written.MACHINE] == 1

    def test_the_header_is_not_an_entry(self) -> None:
        """87 008 entries is the count everything else is checked against, and a
        header counted as an entry would put it 548 out."""
        corpus = corpus_of(catalog_of(machine("One.", "Một.")))
        assert report.by_tier(corpus)[5].total == 1

    def test_sections_are_what_a_reader_recognises(self) -> None:
        corpus = corpus_of(
            catalog_of(entry("One."), name="library/one.po"),
            catalog_of(entry("Two."), name="library/two.po"),
            catalog_of(entry("Three."), name="bugs.po"),
        )
        assert report.by_section(corpus).keys() == {"library", "bugs.po"}
        assert report.by_section(corpus)["library"].total == 2


class TestCoverage:
    def test_only_the_human_column_is_called_translated(self) -> None:
        """The sentence this file exists to make it impossible to get wrong."""
        corpus = corpus_of(
            catalog_of(machine("One.", "Một."), reviewed("Two.", "Hai."), name="tutorial/one.po")
        )
        written = report.coverage(corpus)
        assert "**1 entries are translated**" in written
        assert "machine entries are a corpus to review" in written

    def test_the_counts_marker_is_what_the_audit_reads(self) -> None:
        """A01 recounts entries done, so the marker holds done and not
        translated. The two are different and reading it back is the check."""
        corpus = corpus_of(
            catalog_of(machine("One.", "Một."), entry("Two."), name="tutorial/one.po")
        )
        assert counts(report.coverage(corpus)) == {"1": 1}

    def test_every_tier_present_in_the_corpus_is_reported(self) -> None:
        corpus = corpus_of(
            catalog_of(entry("One."), name="tutorial/one.po"),
            catalog_of(entry("Two."), name="c-api/two.po"),
        )
        assert set(counts(report.coverage(corpus)) or {}) == {"1", "4"}

    def test_an_empty_corpus_produces_a_report_rather_than_a_traceback(self) -> None:
        """The state a new checkout is in, and the state CI runs first."""
        assert report.coverage(corpus_of()).startswith("# Coverage")


class TestRender:
    def test_the_table_lands_between_the_markers(self) -> None:
        corpus = corpus_of(catalog_of(machine("One.", "Một."), name="tutorial/one.po"))
        written = report.render(README, corpus)
        assert written.index(report.TABLE_OPEN) < written.index("| Tier |")
        assert written.index("| Tier |") < written.index(report.TABLE_CLOSE)

    def test_the_prose_around_it_is_untouched(self) -> None:
        """The half of the README no machine can write."""
        assert report.render(README, corpus_of()).endswith("Prose.\n")

    def test_regenerating_twice_changes_nothing(self) -> None:
        """H06 compares two generated files, so a generator that is not stable
        would fail the check on a corpus nobody touched."""
        corpus = corpus_of(catalog_of(machine("One.", "Một."), name="tutorial/one.po"))
        once = report.render(README, corpus)
        assert report.render(once, corpus) == once

    def test_the_readme_and_the_report_agree_by_construction(self) -> None:
        """Which is what makes H06 a check on staleness rather than on arithmetic."""
        corpus = corpus_of(catalog_of(machine("One.", "Một."), name="tutorial/one.po"))
        assert counts(report.render(README, corpus)) == counts(report.coverage(corpus))

    def test_a_readme_with_no_fence_says_which_two_lines_to_add(self) -> None:
        with pytest.raises(report.ReportError, match="generated: coverage"):
            report.render("# Vietnamese\n", corpus_of())


class TestQualityInvariants:
    def test_a_broken_machine_entry_is_counted_against_its_rule(self) -> None:
        """P04, a leading space the model added, which is the one the corpus
        actually has 46 of."""
        corpus = corpus_of(catalog_of(machine("One.", " Một.")))
        assert "| `P04` | 1 |" in report.quality(corpus)

    def test_a_persons_sentence_is_not_the_pipelines_pass_rate(self) -> None:
        """A human entry that breaks a rule is their decision about their own
        sentence, and counting it would dilute the number that says whether the
        pipeline holds."""
        corpus = corpus_of(catalog_of(reviewed("One.", " Một.")))
        assert "No machine-written entry" in report.quality(corpus)

    def test_a_clean_corpus_says_so_rather_than_printing_an_empty_table(self) -> None:
        corpus = corpus_of(catalog_of(machine("One.", "Một.")))
        assert "Every rule passes on every entry." in report.quality(corpus)


class TestQualityRefusals:
    def test_no_run_on_record_is_reported_as_absent_and_not_as_zero(self) -> None:
        """A refusal rate of 0.0 % reads as a perfect run. An absent one reads
        as absent, which is the true statement."""
        written = report.quality(corpus_of())
        assert "No translation run on record" in written
        assert "0.0%" not in written

    def test_the_rules_come_from_the_run_because_the_corpus_cannot_have_them(self) -> None:
        """An entry refused on rung 1 and accepted on rung 2 is in the corpus
        once, with nothing to say it cost three calls."""
        tally = Tally(run="r1", accepted=90, refused=10, by_rule={"P01": 8, "P08": 2})
        written = report.quality(corpus_of(), tallies=[tally])
        assert "| `P01` | 8 |" in written
        assert "Acceptance rate 90.00%" in written

    def test_runs_are_added_up_rather_than_reported_one_by_one(self) -> None:
        """A tier is translated over several sittings and the question is what
        the tier cost."""
        runs = [
            Tally(run="r1", accepted=10, by_rule={"P01": 1}, by_attempt={1: 1}),
            Tally(run="r2", accepted=10, by_rule={"P01": 2}, by_attempt={2: 2}),
        ]
        written = report.quality(corpus_of(), tallies=runs)
        assert "| `P01` | 3 |" in written
        assert "2 runs" in written

    def test_the_rungs_are_reported_in_order(self) -> None:
        tally = Tally(run="r1", accepted=1, by_attempt={2: 5, 1: 9})
        rungs = report.quality(corpus_of(), tallies=[tally])
        assert rungs.index("| 1 | 9 |") < rungs.index("| 2 | 5 |")


class TestQualityAdherence:
    def test_the_worst_terms_come_first(self) -> None:
        glossary = Glossary(version=1, terms=(Term(en="list", vi="danh sách"),))
        one = catalog_of(
            machine("Return a list.", "Trả về một mảng."),
            machine("Another list.", "Một mảng khác."),
        )
        assert "| list | 2 |" in report.quality(corpus_of(one, glossary=glossary))

    def test_a_checkout_with_no_glossary_says_so(self) -> None:
        assert "No glossary on this checkout." in report.quality(corpus_of())

    def test_a_corpus_that_kept_to_it_says_so(self) -> None:
        glossary = Glossary(version=1, terms=(Term(en="list", vi="danh sách"),))
        one = catalog_of(machine("Return a list.", "Trả về một danh sách."))
        assert "Every term that appeared was rendered." in report.quality(
            corpus_of(one, glossary=glossary)
        )


class TestQualityDead:
    def test_every_dead_job_is_named_by_the_id_a02_looks_for(self, tmp_path: Path) -> None:
        """A02 fails unless the id is in this file, so the check and the report
        are two halves of one thing."""
        job = Job(
            id="a1b2c3d4",
            stage=Stage.TRANSLATE,
            payload={"file": "library/one.po", "segments": ["s1", "s2"]},
            attempts=3,
            error="2 entries refused",
        )
        target = tmp_path / str(Stage.TRANSLATE) / str(State.DEAD)
        target.mkdir(parents=True)
        (target / f"{job.id}.json").write_text(job.as_json(), encoding="utf-8")
        written = report.quality(corpus_of(queue=tmp_path))
        assert "a1b2c3d4" in written
        assert "2 entries refused" in written

    def test_a_job_that_died_with_no_reason_still_says_so(self, tmp_path: Path) -> None:
        job = Job(id="deadbeef", stage=Stage.TRANSLATE, payload={"file": "one.po"}, attempts=3)
        target = tmp_path / str(Stage.TRANSLATE) / str(State.DEAD)
        target.mkdir(parents=True)
        (target / f"{job.id}.json").write_text(job.as_json(), encoding="utf-8")
        assert "no reason recorded" in report.quality(corpus_of(queue=tmp_path))

    def test_an_empty_dead_letter_queue_is_worth_saying(self, tmp_path: Path) -> None:
        assert "Nothing in the dead letter queue." in report.quality(corpus_of(queue=tmp_path))


class TestQualityRoutes:
    def test_calls_are_reported_by_route(self) -> None:
        tally = Tally(run="r1", calls=200, seconds=3600.0, by_route={"server1": 120, "server2": 80})
        written = report.quality(corpus_of(), tallies=[tally])
        assert "| server1 | 120 | 60.0% |" in written
        assert "200 calls over 1.0 hours" in written

    def test_a_run_that_made_no_calls_is_not_a_table_of_zeros(self) -> None:
        assert "made no calls" in report.quality(corpus_of(), tallies=[Tally(run="r1")])


class TestTallyPersistence:
    """The one number in the quality report that the corpus cannot produce, so
    the one that has to survive the run that measured it."""

    def test_a_tally_reads_back_as_what_was_written(self) -> None:
        tally = Tally(run="r1", accepted=9, by_rule={"P01": 2}, by_attempt={1: 2})
        assert Tally.from_json(tally.as_json()) == tally

    def test_the_rungs_come_back_as_numbers_and_not_as_strings(self) -> None:
        """JSON has no integer keys, so a naive read sorts 10 before 2."""
        assert Tally.from_json(Tally(run="r1", by_attempt={1: 1}).as_json()).by_attempt == {1: 1}

    def test_a_field_a_later_version_added_does_not_make_the_file_unreadable(self) -> None:
        """These accumulate one per run over months."""
        assert Tally.from_json('{"run": "r1", "invented_later": 3}').run == "r1"

    def test_the_run_names_the_file_with_the_colons_taken_out(self, tmp_path: Path) -> None:
        written = Tally(run="2026-08-15T04:11Z").save(tmp_path)
        assert written.name == "2026-08-15T04-11Z.json"

    def test_every_run_on_record_comes_back_oldest_first(self, tmp_path: Path) -> None:
        Tally(run="2026-08-15T04:00Z", accepted=1).save(tmp_path)
        Tally(run="2026-08-14T04:00Z", accepted=2).save(tmp_path)
        assert [one.accepted for one in Tally.read_all(tmp_path)] == [2, 1]

    def test_a_half_written_file_costs_one_run_and_not_the_report(self, tmp_path: Path) -> None:
        """A machine that lost power mid-write should not take the report with it."""
        Tally(run="r1", accepted=1).save(tmp_path)
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert len(Tally.read_all(tmp_path)) == 1

    def test_no_directory_is_no_runs_rather_than_an_error(self, tmp_path: Path) -> None:
        assert Tally.read_all(tmp_path / "nothing") == []
