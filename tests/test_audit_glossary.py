"""``G01`` to ``G06``: the terminology, and whether the corpus kept to it.

All six are soft until M7, so what these tests establish is mostly that each one
says something specific enough to act on. A soft check whose findings a reviewer
cannot tell apart from noise is a soft check nobody reads.
"""

from conftest import catalog_of, corpus_of, entry, findings, machine_segment
from pydocvi.audit import glossary
from pydocvi.catalog import Entry
from pydocvi.glossary import Glossary, Term, render
from pydocvi.memory import Memory, Segment

MARKDOWN = "# Glossary\n\n<!-- generated: terms -->\n<!-- /generated: terms -->\n"


def terms(*rows: Term, version: int = 1) -> Glossary:
    return Glossary(version=version, terms=rows)


LIST = Term(en="list", vi="danh sách")
TUPLE = Term(en="tuple", vi="tuple", keep_en=True)


class TestG01:
    def test_two_terms_sharing_a_rendering_are_found(self) -> None:
        """A collision is how a reader searching for one thing finds another,
        and a glossary can acquire one with nobody editing it, by a merge that
        takes both sides of a conflict."""
        both = terms(LIST, Term(en="array", vi="danh sách"))
        assert len(findings(glossary.g01_glossary_is_consistent, corpus_of(glossary=both))) >= 1

    def test_a_clean_glossary_is_clean(self) -> None:
        assert findings(glossary.g01_glossary_is_consistent, corpus_of(glossary=terms(LIST))) == []

    def test_a_run_with_no_glossary_says_nothing(self) -> None:
        assert findings(glossary.g01_glossary_is_consistent, corpus_of()) == []


class TestG02:
    def test_a_rendering_that_never_arrived_is_found(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một mảng."))
        found = findings(glossary.g02_renderings_are_used, corpus_of(one, glossary=terms(LIST)))
        assert len(found) == 1
        assert "danh sách" in found[0].detail

    def test_a_rendering_that_arrived_is_clean(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        corpus = corpus_of(one, glossary=terms(LIST))
        assert findings(glossary.g02_renderings_are_used, corpus) == []

    def test_a_kept_term_is_g03_s_business_and_not_this_one(self) -> None:
        one = catalog_of(entry("Return a tuple.", "Trả về một bộ."))
        corpus = corpus_of(one, glossary=terms(TUPLE))
        assert findings(glossary.g02_renderings_are_used, corpus) == []

    def test_a_term_inside_a_role_is_not_the_word(self) -> None:
        """``:class:`list``` is a link target, not a noun in a sentence."""
        one = catalog_of(entry("See :class:`list`.", "Xem :class:`list`."))
        corpus = corpus_of(one, glossary=terms(LIST))
        assert findings(glossary.g02_renderings_are_used, corpus) == []


class TestG03:
    def test_a_kept_term_somebody_translated_is_found(self) -> None:
        """A translated keep_en term is a translation that invented a Vietnamese
        word for something Vietnamese programmers say in English."""
        one = catalog_of(entry("Return a tuple.", "Trả về một bộ."))
        found = findings(
            glossary.g03_kept_terms_stay_english, corpus_of(one, glossary=terms(TUPLE))
        )
        assert len(found) == 1
        assert "kept in English" in found[0].detail

    def test_a_kept_term_that_survived_is_clean(self) -> None:
        one = catalog_of(entry("Return a tuple.", "Trả về một tuple."))
        corpus = corpus_of(one, glossary=terms(TUPLE))
        assert findings(glossary.g03_kept_terms_stay_english, corpus) == []

    def test_a_translated_row_is_g02_s_business_and_not_this_one(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một mảng."))
        corpus = corpus_of(one, glossary=terms(LIST))
        assert findings(glossary.g03_kept_terms_stay_english, corpus) == []


class TestG04:
    def test_an_english_term_left_standing_is_found(self) -> None:
        """The sentence was translated around a word that was left alone, which
        G02 cannot see because the rendering is absent in both readings."""
        one = catalog_of(entry("Return a list.", "Trả về một list."))
        found = findings(
            glossary.g04_no_english_term_survives, corpus_of(one, glossary=terms(LIST))
        )
        assert len(found) == 1
        assert "still in English" in found[0].detail

    def test_a_kept_term_is_meant_to_survive(self) -> None:
        one = catalog_of(entry("Return a tuple.", "Trả về một tuple."))
        corpus = corpus_of(one, glossary=terms(TUPLE))
        assert findings(glossary.g04_no_english_term_survives, corpus) == []

    def test_a_translated_entry_is_clean(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        corpus = corpus_of(one, glossary=terms(LIST))
        assert findings(glossary.g04_no_english_term_survives, corpus) == []


class TestG05:
    def test_a_markdown_table_nobody_regenerated_is_found(self) -> None:
        corpus = corpus_of(glossary=terms(LIST), markdown=MARKDOWN)
        assert len(findings(glossary.g05_markdown_agrees, corpus)) == 1

    def test_the_generated_file_agrees_with_itself(self) -> None:
        one = terms(LIST)
        corpus = corpus_of(glossary=one, markdown=render(MARKDOWN, one))
        assert findings(glossary.g05_markdown_agrees, corpus) == []

    def test_a_missing_markdown_file_is_not_a_glossary_failure(self) -> None:
        assert findings(glossary.g05_markdown_agrees, corpus_of(glossary=terms(LIST))) == []


class TestG06:
    def test_an_entry_translated_against_an_older_version_is_counted(self) -> None:
        """Counted rather than fixed, so that the number is visible rather than
        discovered later by somebody wondering why a term reads two ways."""
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.", glossary=1)])
        corpus = corpus_of(one, glossary=terms(LIST, version=2), memory=memory)
        found = findings(glossary.g06_glossary_version_is_current, corpus)
        assert len(found) == 1
        assert "current is 2" in found[0].detail

    def test_an_entry_on_the_current_version_is_clean(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.", glossary=2)])
        corpus = corpus_of(one, glossary=terms(LIST, version=2), memory=memory)
        assert findings(glossary.g06_glossary_version_is_current, corpus) == []

    def test_a_person_s_translation_is_not_stale(self) -> None:
        """A person's translation was not made against a glossary version, and
        marking it stale would claim the glossary outranks them."""
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        theirs = Entry(msgid="Return a list.", msgstr="Trả về một danh sách.")
        memory = Memory([Segment.from_entry(theirs, source="human")])
        corpus = corpus_of(one, glossary=terms(LIST, version=2), memory=memory)
        assert findings(glossary.g06_glossary_version_is_current, corpus) == []
