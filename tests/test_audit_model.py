"""The pieces every check is built out of, tested once here rather than in six places."""

from pathlib import Path

import pytest

from conftest import catalog_of, corpus_of, entry
from pydocvi import audit
from pydocvi.audit.model import (
    Check,
    Corpus,
    Finding,
    Group,
    Registry,
    Report,
    Result,
    UnknownCheckError,
    counts,
    plural,
)


def made(identifier: str, group: Group, *, hard: bool = True) -> Check:
    return Check(id=identifier, group=group, hard=hard, title="a check", body=lambda _: iter(()))


def failing(identifier: str, group: Group, *, hard: bool = True, count: int = 1) -> Result:
    one = made(identifier, group, hard=hard)
    return Result(
        check=one,
        findings=tuple(
            Finding(check=identifier, path="library/one.po", detail="wrong") for _ in range(count)
        ),
    )


class TestFinding:
    def test_it_names_a_file_and_a_line(self) -> None:
        """A check that can only say something is wrong somewhere in library/ is
        not a check, it is a mood."""
        one = Finding(check="P03", path="library/one.po", line=42, detail="target changed")
        assert str(one).startswith("library/one.po:42")

    def test_a_finding_with_no_line_still_names_the_file(self) -> None:
        one = Finding(check="H01", path="library/one.mo", detail="compiled")
        assert str(one) == "library/one.mo  compiled"

    def test_the_dictionary_carries_the_english_and_what_came_back(self) -> None:
        one = Finding(check="P03", path="a.po", detail="d", english="Return a list.", got="Trả về.")
        assert one.as_dict()["english"] == "Return a list."
        assert one.as_dict()["got"] == "Trả về."


class TestReport:
    def test_a_clean_run_is_ok(self) -> None:
        report = Report(results=(Result(check=made("S01", Group.STRUCTURE)),))
        assert report.ok

    def test_a_hard_failure_is_not_ok(self) -> None:
        assert not Report(results=(failing("S02", Group.STRUCTURE),)).ok

    def test_a_soft_failure_alone_is_still_ok(self) -> None:
        """L05, L06 and L08 each have real exceptions, and a hard rule that a
        correct translation breaks is not a rule."""
        assert Report(results=(failing("L05", Group.LANGUAGE, hard=False),)).ok

    def test_fail_soft_makes_a_soft_failure_count(self) -> None:
        report = Report(results=(failing("L05", Group.LANGUAGE, hard=False),), fail_soft=True)
        assert not report.ok

    def test_failing_can_be_asked_for_one_hardness(self) -> None:
        report = Report(
            results=(
                failing("S02", Group.STRUCTURE),
                failing("L05", Group.LANGUAGE, hard=False),
            )
        )
        assert [one.check.id for one in report.failing(hard=True)] == ["S02"]
        assert [one.check.id for one in report.failing(hard=False)] == ["L05"]

    def test_findings_are_flattened_across_checks(self) -> None:
        report = Report(
            results=(
                failing("S02", Group.STRUCTURE, count=2),
                failing("P01", Group.PLACEHOLDERS, count=3),
            )
        )
        assert len(report.findings) == 5

    def test_the_json_names_every_failing_check(self) -> None:
        report = Report(results=(failing("S02", Group.STRUCTURE),))
        assert '"S02"' in report.as_json()
        assert report.as_json().endswith("\n")


class TestRegistry:
    def test_a_check_may_not_be_registered_twice(self) -> None:
        registry = Registry()
        registry.add(made("S01", Group.STRUCTURE))
        with pytest.raises(ValueError, match="twice"):
            registry.add(made("S01", Group.STRUCTURE))

    def test_a_check_must_be_numbered_in_its_own_group(self) -> None:
        """P05 and S05 differ by one keystroke and totally in what they mean."""
        registry = Registry()
        with pytest.raises(ValueError, match="does not belong"):
            registry.check("P05", Group.STRUCTURE, hard=True, title="wrong group")(
                lambda _: iter(())
            )

    def test_declaration_order_is_report_order(self) -> None:
        registry = Registry()
        for identifier in ("S03", "S01", "S02"):
            registry.add(made(identifier, Group.STRUCTURE))
        assert [one.id for one in registry.selected()] == ["S03", "S01", "S02"]

    def test_only_takes_an_id(self) -> None:
        assert [one.id for one in audit.REGISTRY.selected(only=["P03"])] == ["P03"]

    def test_only_takes_a_group_name(self) -> None:
        selected = audit.REGISTRY.selected(only=["hygiene"])
        assert {one.group for one in selected} == {Group.HYGIENE}

    def test_only_takes_a_one_letter_prefix(self) -> None:
        """The runbook says --only P,G and somebody will type it."""
        selected = audit.REGISTRY.selected(only=["P", "G"])
        assert {one.group for one in selected} == {Group.PLACEHOLDERS, Group.GLOSSARY}

    def test_skip_removes_a_group(self) -> None:
        selected = audit.REGISTRY.selected(skip=["glossary"])
        assert Group.GLOSSARY not in {one.group for one in selected}

    def test_skip_wins_over_only(self) -> None:
        selected = audit.REGISTRY.selected(only=["P"], skip=["P03"])
        assert "P03" not in {one.id for one in selected}
        assert "P01" in {one.id for one in selected}

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(UnknownCheckError):
            audit.REGISTRY.selected(only=["P99"])

    def test_the_refusal_offers_the_near_misses(self) -> None:
        """P03 and S03 are one keystroke apart and mean entirely different things."""
        with pytest.raises(UnknownCheckError, match="P03"):
            audit.REGISTRY.selected(only=["X03"])


class TestCorpus:
    def test_translated_skips_the_entries_nobody_has_got_to(self) -> None:
        """A corpus that is 4 % done reads as clean if untranslated entries pass
        every rule, which they do, because there is nothing in them to break."""
        one = catalog_of(entry("Done.", "Xong."), entry("Not yet."))
        assert [e.msgid for _, e in corpus_of(one).translated()] == ["Done."]

    def test_prose_skips_the_entries_no_model_was_ever_asked_about(self) -> None:
        """A copied doctest has a ``msgstr`` because copying it is what it is
        for. Judging it as a translation is how ``G02`` went from 321 findings
        to 2 836 the day ``apply`` started minting the copies, without a single
        translation changing."""
        code = ">>> sorted(d.keys())"
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."), entry(code, code))
        assert [e.msgid for _, e in corpus_of(one).prose()] == ["Return a list."]

    def test_prose_asks_the_classifier_and_not_the_marker(self) -> None:
        """``L04``'s whole job is to find where the two have drifted apart, so a
        check that trusted the marker here would go blind on exactly the entries
        that are mislabelled."""
        code = ">>> len([1, 2, 3])"
        lying = entry(code, code).with_comments(["# pydocvi: passthrough=prose"])
        assert list(corpus_of(catalog_of(lying)).prose()) == []

    def test_relative_is_the_path_a_reviewer_would_type(self) -> None:
        root = Path("/corpus")
        one = catalog_of(entry("a", "b"), name="library/os.po", root=root)
        assert corpus_of(one, root=root).relative(one.path) == "library/os.po"

    def test_a_path_outside_the_root_is_left_whole(self) -> None:
        corpus = Corpus(root=Path("/corpus"))
        assert corpus.relative(Path("/elsewhere/a.po")) == "/elsewhere/a.po"

    def test_paired_hands_back_none_when_upstream_has_no_such_entry(self) -> None:
        one = catalog_of(entry("Edited here.", "Đã sửa."))
        pairs = list(corpus_of(one).paired())
        assert pairs[0][2] is None


class TestCountsMarker:
    def test_it_reads_the_generated_marker(self) -> None:
        assert counts('# Coverage\n\n<!-- counts: {"1": 1626, "2": 0} -->\n') == {
            "1": 1626,
            "2": 0,
        }

    def test_a_file_with_no_marker_reads_as_none(self) -> None:
        """None rather than empty, because "no marker" and "nothing translated"
        are different situations and only one of them is a check that cannot run."""
        assert counts("# Coverage\n") is None

    def test_a_broken_marker_reads_as_none_rather_than_raising(self) -> None:
        assert counts("<!-- counts: {not json} -->") is None

    def test_non_integer_values_are_dropped(self) -> None:
        assert counts('<!-- counts: {"1": 5, "2": "many"} -->') == {"1": 5}


class TestPlural:
    def test_one_gets_no_s(self) -> None:
        assert plural(1, "check") == "1 check"

    def test_everything_else_does(self) -> None:
        assert plural(0, "check") == "0 checks"
        assert plural(2, "finding") == "2 findings"

    def test_large_counts_are_grouped(self) -> None:
        assert plural(87008, "entry") == "87,008 entrys"


class TestTheAssembledRegistry:
    def test_every_check_in_the_spec_is_registered(self) -> None:
        assert len(audit.REGISTRY) == 41

    def test_the_groups_run_in_report_order(self) -> None:
        """A structural failure explains most of what follows it, so printing an
        L02 above the S02 that caused it wastes the reader's first minute."""
        order = [one.group for one in audit.REGISTRY.selected()]
        assert order == sorted(order, key=list(Group).index)

    def test_every_id_is_numbered_in_its_own_group(self) -> None:
        assert all(one.id.startswith(one.group.prefix) for one in audit.REGISTRY.selected())

    def test_every_check_has_a_title_that_reads_as_a_claim(self) -> None:
        """A title says what is true when the check passes, so it goes in a table
        row and not at the end of a sentence."""
        titles = [one.title for one in audit.REGISTRY.selected()]
        assert all(titles)
        assert not any(title.endswith(".") for title in titles)


class TestMarkdown:
    def test_a_clean_report_says_so_and_prints_no_findings(self) -> None:
        text = audit.markdown(Report(results=(Result(check=made("S01", Group.STRUCTURE)),)))
        assert text.startswith("# Audit")
        assert "## Findings" not in text

    def test_a_failing_report_prints_the_findings(self) -> None:
        text = audit.markdown(Report(results=(failing("S02", Group.STRUCTURE),)))
        assert "### `S02`" in text
        assert "library/one.po" in text

    def test_truncation_is_announced_rather_than_silent(self) -> None:
        """A report that truncated without saying so would read as though the
        problem were twenty entries wide."""
        text = audit.markdown(Report(results=(failing("S02", Group.STRUCTURE, count=25),)), limit=5)
        assert "and 20 more, not printed" in text
