"""The four sources, and the order they are trusted in.

Mining decides nothing, so the tests here are about what gets *proposed* and in
what order. A source that proposes nothing is a silent failure, which is why
every source has a test that it proposes something as well as tests that it
leaves things alone.
"""

from pathlib import Path

import pytest

from pydocvi import mine
from pydocvi.catalog import Catalog, Entry, segment_id
from pydocvi.memory import Memory, Segment
from pydocvi.mine import Candidate, Source


def catalog(*pairs: tuple[str, str], path: str = "glossary.po") -> Catalog:
    return Catalog(
        path=Path(path),
        header=Entry(msgid=""),
        entries=tuple(Entry(msgid=msgid, msgstr=msgstr) for msgid, msgstr in pairs),
    )


def memory(*rows: tuple[str, str, str]) -> Memory:
    return Memory(
        Segment(id=segment_id(msgid), msgid=msgid, msgstr=msgstr, source=source)  # type: ignore[arg-type]
        for msgid, msgstr, source in rows
    )


class TestTermLike:
    def test_a_short_phrase_is_a_term(self):
        assert mine.term_like("context manager")

    def test_a_sentence_is_not(self):
        assert not mine.term_like("A value passed to a function.")

    def test_a_clause_ending_in_a_colon_is_not(self):
        assert not mine.term_like("There are two kinds:")

    def test_four_words_is_not_a_term(self):
        assert not mine.term_like("the abstract base class")

    def test_a_long_phrase_is_not_a_term_even_at_three_words(self):
        assert not mine.term_like("supercalifragilistic expialidocious pseudopseudonym")

    def test_an_empty_string_is_not_a_term(self):
        assert not mine.term_like("")

    def test_a_string_that_is_only_markup_is_not_a_term(self):
        assert not mine.term_like("``>>>``")

    def test_a_cross_reference_is_markup_rather_than_a_term(self):
        """The whole entry is a role, so there is no prose left to be a term."""
        assert not mine.term_like(":term:`iterable`")

    def test_a_term_carrying_markup_is_judged_on_the_prose_around_it(self):
        assert mine.term_like("the :term:`iterable` protocol")

    def test_a_number_on_its_own_is_not_a_term(self):
        assert not mine.term_like("3.15")


class TestHumanSource:
    def test_a_term_a_person_translated_is_proposed(self):
        found = mine.from_human(memory(("context manager", "trình quản lý ngữ cảnh", "human")))
        assert [candidate.en for candidate in found] == ["context manager"]

    def test_the_rendering_a_person_chose_comes_with_it(self):
        found = mine.from_human(memory(("iterable", "khả lặp", "human")))
        assert found[0].seen == ("khả lặp",)

    def test_machine_work_is_not_a_human_source(self):
        assert mine.from_human(memory(("iterable", "khả lặp", "machine"))) == []

    def test_a_sentence_is_left_alone_rather_than_aligned(self):
        rows = memory(("A value passed to a function.", "Một giá trị.", "human"))
        assert mine.from_human(rows) == []

    def test_an_untranslated_entry_proposes_nothing(self):
        assert mine.from_human(memory(("iterable", "   ", "human"))) == []

    def test_the_same_english_rendered_two_ways_under_two_contexts_is_contested(self):
        """The memory holds one rendering per msgid and msgctxt, so this is the
        only shape a human disagreement can have, and it is worth surfacing."""
        rows = Memory(
            [
                Segment(
                    id=segment_id("iterable", "protocol"),
                    msgid="iterable",
                    msgstr="khả lặp",
                    msgctxt="protocol",
                    source="human",
                ),
                Segment(
                    id=segment_id("iterable", "type"),
                    msgid="iterable",
                    msgstr="lặp được",
                    msgctxt="type",
                    source="human",
                ),
            ]
        )
        found = mine.from_human(rows)
        assert found[0].contested and found[0].count == 2

    def test_candidates_are_attributed_to_the_most_trusted_source(self):
        assert mine.from_human(memory(("iterable", "khả lặp", "human")))[0].source is Source.HUMAN


class TestTermPageSource:
    def test_a_term_followed_by_a_definition_is_proposed(self):
        page = catalog(
            ("iterable", ""),
            ("An object capable of returning its members one at a time.", ""),
        )
        assert [candidate.en for candidate in mine.from_term_page(page)] == ["iterable"]

    def test_the_definition_sentence_comes_with_it(self):
        page = catalog(
            ("iterable", ""),
            ("An object capable of returning its members. More text follows.", ""),
        )
        assert (
            mine.from_term_page(page)[0].definition == "An object capable of returning its members."
        )

    def test_a_term_with_no_definition_after_it_is_not_proposed(self):
        assert mine.from_term_page(catalog(("iterable", ""))) == []

    def test_two_terms_in_a_row_means_neither_defines_the_other(self):
        page = catalog(("iterable", ""), ("iterator", ""), ("An object. Yes.", ""))
        assert [candidate.en for candidate in mine.from_term_page(page)] == ["iterator"]

    def test_a_glossary_key_that_is_only_markup_is_not_a_term(self):
        page = catalog(("``>>>``", ""), ("The default Python prompt. It is shown.", ""))
        assert mine.from_term_page(page) == []

    def test_the_definition_keeps_one_sentence_rather_than_the_paragraph(self):
        page = catalog(("annotation", ""), ("A label. Another sentence. A third.", ""))
        assert mine.from_term_page(page)[0].definition == "A label."

    def test_markup_is_stripped_out_of_the_definition(self):
        page = catalog(("annotation", ""), ("A label used by :term:`type hint` here.", ""))
        assert ":term:" not in mine.from_term_page(page)[0].definition

    def test_an_empty_entry_between_a_term_and_its_definition_is_skipped(self):
        page = catalog(("iterable", ""), ("", ""), ("An object capable of it.", ""))
        assert len(mine.from_term_page(page)) == 1

    def test_the_last_entry_in_the_file_has_nothing_defining_it(self):
        page = catalog(("A sentence about things.", ""), ("iterable", ""))
        assert mine.from_term_page(page) == []


class TestFrequencySource:
    def test_a_repeated_phrase_is_proposed(self):
        found = mine.from_frequency(["a context manager here"] * 10, minimum=8)
        assert "context manager" in {candidate.en for candidate in found}

    def test_a_phrase_below_the_floor_is_not_proposed(self):
        assert mine.from_frequency(["a context manager here"] * 3, minimum=8) == []

    def test_a_phrase_starting_with_a_stopword_is_not_proposed(self):
        found = {candidate.en for candidate in mine.from_frequency(["the module"] * 10, minimum=8)}
        assert "the module" not in found and "module" in found

    def test_a_phrase_ending_with_a_stopword_is_not_proposed(self):
        found = {candidate.en for candidate in mine.from_frequency(["module of"] * 10, minimum=8)}
        assert "module of" not in found

    def test_a_phrase_never_crosses_a_comma(self):
        found = {
            candidate.en for candidate in mine.from_frequency(["module, function"] * 10, minimum=8)
        }
        assert "module function" not in found

    def test_a_phrase_never_crosses_a_full_stop(self):
        found = {
            candidate.en
            for candidate in mine.from_frequency(["one module. Another thing"] * 10, minimum=8)
        }
        assert "module another" not in found

    def test_a_phrase_inside_a_protected_span_is_code_and_is_not_counted(self):
        found = mine.from_frequency(["see ``context manager`` here"] * 10, minimum=8)
        assert "context manager" not in {candidate.en for candidate in found}

    def test_a_phrase_is_never_longer_than_three_words(self):
        found = mine.from_frequency(["abstract base class metaclass"] * 10, minimum=8)
        assert all(candidate.words <= 3 for candidate in found)

    def test_counting_is_case_insensitive(self):
        found = mine.from_frequency(["Context manager"] * 5 + ["context manager"] * 5, minimum=8)
        assert "context manager" in {candidate.en for candidate in found}

    def test_the_limit_caps_the_list(self):
        corpus = [" ".join(f"word{'x' * n}" for n in range(50))] * 10
        assert len(mine.from_frequency(corpus, limit=5, minimum=8)) == 5

    def test_the_most_frequent_phrase_comes_first(self):
        corpus = ["common word"] * 20 + ["rare word"] * 8
        found = mine.from_frequency(corpus, minimum=8)
        assert found[0].en == "word"

    def test_an_identifier_is_not_counted_as_a_word(self):
        found = {
            candidate.en for candidate in mine.from_frequency(["call foo_bar"] * 10, minimum=8)
        }
        assert "foo_bar" not in found

    def test_a_hyphenated_word_is_one_word(self):
        found = {
            candidate.en for candidate in mine.from_frequency(["a built-in thing"] * 10, minimum=8)
        }
        assert "built-in" in found

    def test_an_empty_corpus_proposes_nothing(self):
        assert mine.from_frequency([]) == []


class TestMachineSource:
    def test_a_phrase_rendered_two_ways_is_proposed(self):
        one = catalog(("iterable", "khả lặp"), path="a.po")
        two = catalog(("iterable", "lặp được"), path="b.po")
        assert [candidate.en for candidate in mine.from_machine([one, two])] == ["iterable"]

    def test_the_disagreeing_renderings_come_with_it(self):
        one = catalog(("iterable", "khả lặp"), path="a.po")
        two = catalog(("iterable", "lặp được"), path="b.po")
        assert mine.from_machine([one, two])[0].seen == ("khả lặp", "lặp được")

    def test_agreement_is_not_a_signal_and_proposes_nothing(self):
        one = catalog(("iterable", "khả lặp"), path="a.po")
        two = catalog(("iterable", "khả lặp"), path="b.po")
        assert mine.from_machine([one, two]) == []

    def test_an_untranslated_entry_is_not_a_disagreement(self):
        one = catalog(("iterable", "khả lặp"), path="a.po")
        two = catalog(("iterable", ""), path="b.po")
        assert mine.from_machine([one, two]) == []

    def test_a_sentence_rendered_two_ways_is_not_a_term(self):
        one = catalog(("A value passed to a function.", "Một giá trị."), path="a.po")
        two = catalog(("A value passed to a function.", "Giá trị."), path="b.po")
        assert mine.from_machine([one, two]) == []

    def test_the_source_says_the_machine_contradicted_itself(self):
        one = catalog(("iterable", "khả lặp"), path="a.po")
        two = catalog(("iterable", "lặp được"), path="b.po")
        assert mine.from_machine([one, two])[0].source is Source.MACHINE


class TestTrust:
    def test_the_sources_are_ordered_by_how_much_they_are_worth(self):
        assert Source.HUMAN < Source.TERM_PAGE < Source.FREQUENCY < Source.MACHINE

    def test_every_source_can_say_what_it_is_in_a_report(self):
        assert Source.HUMAN.label == "human translation"
        assert Source.MACHINE.label == "machine disagreement"


class TestMerging:
    def test_a_phrase_from_two_sources_keeps_the_more_trusted_one(self):
        merged = mine.merge(
            [Candidate(en="iterable", source=Source.FREQUENCY, count=40)],
            [Candidate(en="iterable", source=Source.HUMAN, count=2)],
        )
        assert [(c.en, c.source) for c in merged] == [("iterable", Source.HUMAN)]

    def test_the_counts_are_summed_across_sources(self):
        merged = mine.merge(
            [Candidate(en="iterable", source=Source.FREQUENCY, count=40)],
            [Candidate(en="iterable", source=Source.HUMAN, count=2)],
        )
        assert merged[0].count == 42

    def test_the_renderings_are_unioned_across_sources(self):
        merged = mine.merge(
            [Candidate(en="iterable", source=Source.MACHINE, seen=("lặp được",))],
            [Candidate(en="iterable", source=Source.HUMAN, seen=("khả lặp",))],
        )
        assert merged[0].seen == ("khả lặp", "lặp được")

    def test_a_definition_survives_from_whichever_source_had_one(self):
        merged = mine.merge(
            [Candidate(en="iterable", source=Source.TERM_PAGE, definition="An object.")],
            [Candidate(en="iterable", source=Source.HUMAN)],
        )
        assert merged[0].definition == "An object."

    def test_the_list_is_sorted_by_trust_and_never_by_count(self):
        merged = mine.merge(
            [Candidate(en="the following example", source=Source.FREQUENCY, count=900)],
            [Candidate(en="context manager", source=Source.HUMAN, count=2)],
        )
        assert [c.en for c in merged] == ["context manager", "the following example"]

    def test_candidates_within_a_source_are_alphabetical(self):
        merged = mine.merge(
            [
                Candidate(en="zebra", source=Source.HUMAN),
                Candidate(en="alpha", source=Source.HUMAN),
            ]
        )
        assert [c.en for c in merged] == ["alpha", "zebra"]

    def test_merging_nothing_produces_nothing(self):
        assert mine.merge() == []

    def test_a_phrase_from_one_source_survives_unchanged(self):
        one = Candidate(en="iterable", source=Source.HUMAN, count=3, seen=("khả lặp",))
        assert mine.merge([one]) == [one]


class TestWriting:
    def test_a_candidate_writes_its_phrase_and_its_source(self):
        written = mine.dumps([Candidate(en="iterable", source=Source.HUMAN, count=3)])
        assert '  - en: "iterable"\n    source: human\n    count: 3\n' in written

    def test_a_definition_is_written_when_there_is_one(self):
        written = mine.dumps(
            [Candidate(en="iterable", source=Source.TERM_PAGE, definition="An object.")]
        )
        assert '    definition: "An object."' in written

    def test_the_observed_renderings_are_written_as_a_list(self):
        written = mine.dumps(
            [Candidate(en="iterable", source=Source.HUMAN, seen=("khả lặp", "lặp được"))]
        )
        assert '    seen:\n      - "khả lặp"\n      - "lặp được"\n' in written

    def test_the_header_says_how_many_there_are(self):
        assert "# 1 candidate(s)" in mine.dumps([Candidate(en="x", source=Source.HUMAN)])

    def test_an_empty_list_still_writes_a_readable_file(self):
        assert mine.dumps([]).endswith("candidates:\n")


class TestStats:
    def test_the_candidates_are_counted_by_source(self):
        found = mine.stats(
            [
                Candidate(en="a", source=Source.HUMAN),
                Candidate(en="b", source=Source.HUMAN),
                Candidate(en="c", source=Source.FREQUENCY),
            ]
        )
        assert found.by_source == {"human translation": 2, "corpus frequency": 1}

    def test_the_ones_with_a_definition_are_counted(self):
        found = mine.stats(
            [
                Candidate(en="a", source=Source.TERM_PAGE, definition="An object."),
                Candidate(en="b", source=Source.HUMAN),
            ]
        )
        assert (found.total, found.defined) == (2, 1)

    def test_the_contested_ones_are_counted(self):
        found = mine.stats([Candidate(en="a", source=Source.MACHINE, seen=("x", "y"))])
        assert found.contested == 1


@pytest.mark.corpus
class TestAgainstTheRealCorpus:
    """The sources against the files they were written for.

    Marked ``corpus`` so it skips without a checkout. It exists because every
    heuristic in this module was tuned against real files, and a change that
    quietly stops proposing anything would otherwise pass the whole suite.
    """

    def test_the_cpython_glossary_page_yields_terms_with_definitions(self, upstream):
        from pydocvi import catalog as catalogs

        page = catalogs.parse((upstream / "glossary.po").read_text(encoding="utf-8"))
        found = mine.from_term_page(page)
        assert len(found) > 100
        assert all(candidate.definition for candidate in found)

    def test_the_terms_this_project_talks_about_are_among_them(self, upstream):
        from pydocvi import catalog as catalogs

        page = catalogs.parse((upstream / "glossary.po").read_text(encoding="utf-8"))
        found = {candidate.en for candidate in mine.from_term_page(page)}
        assert {"iterable", "decorator", "context manager", "argument"} <= found


class TestReading:
    def test_what_dumps_wrote_reads_back_identically(self):
        """The file is edited by hand between mining and curating, so the write
        and the read have to agree about every field or the human pass is lost."""
        before = [
            Candidate(en="iterable", source=Source.TERM_PAGE, count=12, definition="An object."),
            Candidate(en="decorator", source=Source.HUMAN, seen=("decorator", "trình trang trí")),
        ]
        assert mine.loads(mine.dumps(before)) == before

    def test_an_empty_file_reads_as_no_candidates(self):
        assert mine.loads("candidates:\n") == []

    def test_a_file_with_no_candidates_key_reads_as_none(self):
        assert mine.loads("other: 1\n") == []

    def test_a_source_the_enum_does_not_have_is_refused(self):
        text = 'candidates:\n  - en: "x"\n    source: guesswork\n'
        with pytest.raises(mine.MineError, match="guesswork"):
            mine.loads(text)

    def test_an_unknown_field_is_refused_rather_than_dropped(self):
        text = 'candidates:\n  - en: "x"\n    source: human\n    vi: "y"\n'
        with pytest.raises(mine.MineError, match="vi"):
            mine.loads(text)

    def test_a_top_level_list_is_refused(self):
        with pytest.raises(mine.MineError, match="mapping"):
            mine.loads("- one\n- two\n")

    def test_text_that_is_not_yaml_is_refused(self):
        with pytest.raises(mine.MineError):
            mine.loads("candidates: [unclosed\n")

    def test_a_candidate_that_is_not_a_mapping_names_its_position(self):
        with pytest.raises(mine.MineError, match="candidate 2"):
            mine.loads('candidates:\n  - en: "x"\n    source: human\n  - just a string\n')


class TestSubstance:
    """A phrase needs a word in it. Found by running the mine over the corpus."""

    def test_an_alphabet_heading_is_not_a_term(self):
        """``**A**`` through ``**Z**`` are the index headings on CPython's
        glossary page. They are short, carry no sentence punctuation, and were
        19 of the 102 highest-trust candidates the first real run produced."""
        assert not mine.term_like("**A**")

    def test_a_single_letter_is_not_a_term(self):
        assert not mine.term_like("x")

    def test_a_two_letter_word_is_a_term(self):
        assert mine.term_like("os")

    def test_a_glossary_entry_that_names_a_character_survives(self):
        """The rule counts letters in a word rather than the ratio of letters to
        punctuation, because this is a real entry and a ratio would refuse it."""
        assert mine.term_like("# (hash)")

    def test_the_frequency_source_drops_single_letters_too(self):
        found = {one.en for one in mine.from_frequency(["call f and call g"] * 10, minimum=8)}
        assert "f" not in found and "g" not in found

    def test_the_frequency_source_keeps_real_words(self):
        found = {one.en for one in mine.from_frequency(["the context manager"] * 10, minimum=8)}
        assert "context manager" in found

    def test_a_letter_inside_a_phrase_does_not_condemn_the_phrase(self):
        assert mine.term_like("a keyword")


class TestContractions:
    """Apostrophes, found by reading what the first real run declined."""

    def test_a_contraction_is_one_word_rather_than_two(self):
        """``doesn`` and ``doesn t`` both came back at count 550 from the real
        corpus, which is one contraction counted twice as two non-words."""
        found = {one.en for one in mine.from_frequency(["it doesn't work"] * 10, minimum=8)}
        assert "doesn" not in found and "doesn t" not in found

    def test_a_possessive_is_not_proposed(self):
        found = {one.en for one in mine.from_frequency(["python's module"] * 10, minimum=8)}
        assert "python s" not in found

    def test_the_typographic_apostrophe_is_handled_too(self):
        found = {one.en for one in mine.from_frequency(["it doesn\u2019t work"] * 10, minimum=8)}
        assert not any("doesn" in one for one in found)

    def test_the_words_around_a_contraction_still_count(self):
        found = {
            one.en for one in mine.from_frequency(["it doesn't break parsing"] * 10, minimum=8)
        }
        assert "parsing" in found

    def test_a_plain_phrase_is_untouched(self):
        found = {one.en for one in mine.from_frequency(["the context manager"] * 10, minimum=8)}
        assert "context manager" in found
