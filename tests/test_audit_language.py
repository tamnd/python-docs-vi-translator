"""``L01`` to ``L08``: the Vietnamese itself.

Four of these are hard and four are soft, and the split is the interesting part.
A soft rule here is one a correct translation can break, and the tests below say
which correct translations those are.
"""

from dataclasses import replace

from conftest import catalog_of, corpus_of, entry, findings, machine_segment
from pydocvi.audit import language
from pydocvi.audit.model import Corpus
from pydocvi.glossary import Glossary, Term
from pydocvi.memory import Memory


def over(msgid: str, msgstr: str, **overrides: object) -> object:
    return corpus_of(catalog_of(entry(msgid, msgstr, **overrides)))


def with_kept(corpus: object, *english: str) -> Corpus:
    """The same corpus with a glossary that keeps these terms in English."""
    terms = tuple(Term(en=one, vi=one, keep_en=True) for one in english)
    return replace(corpus, glossary=Glossary(version=1, terms=terms))  # type: ignore[type-var]


def with_standalone(corpus: object, en: str, vi: str) -> Corpus:
    """The same corpus with one row translated in prose and kept on its own."""
    terms = (Term(en=en, vi=vi, identifier=True),)
    return replace(corpus, glossary=Glossary(version=1, terms=terms))  # type: ignore[type-var]


class TestL01:
    LONG = "A sentence long enough that a reader would expect a diacritic in it."

    def test_an_english_answer_that_was_committed_is_found(self) -> None:
        assert len(findings(language.l01_is_vietnamese, over(self.LONG, self.LONG))) == 1

    def test_a_translation_with_diacritics_is_clean(self) -> None:
        translated = "Một câu đủ dài để người đọc mong thấy dấu thanh trong đó."
        assert findings(language.l01_is_vietnamese, over(self.LONG, translated)) == []

    def test_a_short_undiacriticked_string_is_ordinary(self) -> None:
        """ "API" and "Unicode" are what a Vietnamese reader expects to see, so
        only a long diacritic-free string is suspicious."""
        assert findings(language.l01_is_vietnamese, over("Unicode", "Unicode")) == []

    def test_a_doctest_is_not_this_check_s_business(self) -> None:
        code = ">>> print(sorted(d.keys()))\n['a', 'b', 'c', 'd', 'e']"
        assert findings(language.l01_is_vietnamese, over(code, code)) == []


class TestL02:
    def test_the_english_handed_back_is_found(self) -> None:
        assert (
            len(findings(language.l02_not_the_source, over("Return a list.", "Return a list.")))
            == 1
        )

    def test_a_passthrough_entry_is_meant_to_be_identical(self) -> None:
        """Being identical to the source is what a version marker is for."""
        assert findings(language.l02_not_the_source, over("3.14", "3.14")) == []

    def test_a_real_translation_is_clean(self) -> None:
        assert (
            findings(language.l02_not_the_source, over("Return a list.", "Trả về một danh sách."))
            == []
        )

    def test_an_entry_that_is_a_kept_term_is_the_glossary_s_business(self) -> None:
        """``sys`` is an index entry naming a module and a reviewer left it in
        English because that is what a Vietnamese programmer calls it. Nothing
        in the string separates it from ``Notes``, so the decision is made once
        in the glossary and ``G03`` checks it in both directions."""
        corpus = over("sys", "sys", flags=())
        assert len(findings(language.l02_not_the_source, corpus)) == 1
        assert findings(language.l02_not_the_source, with_kept(corpus, "sys")) == []

    def test_a_kept_term_inside_a_sentence_exempts_nothing(self) -> None:
        """Matched on the whole ``msgid``. A kept term in a sentence says
        nothing about whether the sentence was translated."""
        english = "Import the sys module first."
        corpus = with_kept(over(english, english), "sys")
        assert len(findings(language.l02_not_the_source, corpus)) == 1

    def test_a_word_the_glossary_does_not_keep_is_still_reported(self) -> None:
        """``object`` is the other side of the same 144 findings, an ordinary
        English word used as an index category, and it wants translating. It is
        one of 36 such entries and the 124 sentences around them say ``đối
        tượng`` 93 times."""
        corpus = with_kept(over("object", "object", flags=()), "sys")
        assert len(findings(language.l02_not_the_source, corpus)) == 1

    def test_a_row_translated_in_prose_can_still_stand_alone(self) -> None:
        """``float`` is ``số thực`` in a sentence and the name of a C type in
        the ``struct`` format table. While this check read ``keep_en`` the row
        could say one or the other, and saying "keep the English" to get the
        table cell right made ``G03`` report both correct translations of it."""
        corpus = with_standalone(over("float", "float", flags=()), "float", "số thực")
        assert findings(language.l02_not_the_source, corpus) == []

    def test_standing_alone_does_not_excuse_a_sentence(self) -> None:
        english = "Divide and get a float."
        corpus = with_standalone(over(english, english), "float", "số thực")
        assert len(findings(language.l02_not_the_source, corpus)) == 1

    def test_no_glossary_means_no_exemption_rather_than_a_crash(self) -> None:
        assert len(findings(language.l02_not_the_source, over("sys", "sys", flags=()))) == 1


class TestL03:
    def test_a_model_talking_about_the_work_is_found(self) -> None:
        corpus = over("Return a list.", "Bản dịch: Trả về một danh sách.")
        assert len(findings(language.l03_no_narration, corpus)) == 1

    def test_a_faithful_translation_of_a_note_is_not_narration(self) -> None:
        """The English is what decides whether "Lưu ý:" is narration or a
        translation, and reading the Vietnamese alone cannot tell them apart."""
        corpus = over("Note: this is slow.", "Lưu ý: điều này chậm.")
        assert findings(language.l03_no_narration, corpus) == []


class TestL04:
    ENGLISH = "in the :mod:`os` module"

    def test_a_recorded_passthrough_that_reads_as_prose_now_is_found(self) -> None:
        """A false positive here silently leaves an English sentence in a
        Vietnamese page, and nothing else will ever look at it again."""
        corpus = over(self.ENGLISH, self.ENGLISH, comments=["# pydocvi: passthrough=noop"])
        found = findings(language.l04_no_prose_in_passthrough, corpus)
        assert len(found) == 1
        assert "noop" in found[0].detail

    def test_a_recorded_passthrough_that_is_still_a_passthrough_is_clean(self) -> None:
        corpus = over("os.path", "os.path", comments=["# pydocvi: passthrough=version_marker"])
        assert findings(language.l04_no_prose_in_passthrough, corpus) == []

    def test_a_reclassification_within_passthrough_is_not_reported(self) -> None:
        """A string that was a no-op and is now a literal block is still not
        being translated and still reads correctly."""
        code = "x = f()\n    return x"
        corpus = over(code, code, comments=["# pydocvi: passthrough=noop"])
        assert findings(language.l04_no_prose_in_passthrough, corpus) == []

    def test_an_entry_with_no_provenance_is_not_judged(self) -> None:
        """This runs over a corpus most of which upstream wrote, and an entry
        that records nothing is claiming nothing to be wrong about."""
        corpus = over(self.ENGLISH, self.ENGLISH)
        assert findings(language.l04_no_prose_in_passthrough, corpus) == []

    def test_a_machine_translation_is_not_this_check_s_business(self) -> None:
        corpus = over(
            "Return a list of the items.",
            "Trả về một danh sách các mục.",
            comments=["# pydocvi: model=gpt-5 run=2026-08-15T09:00Z"],
        )
        assert findings(language.l04_no_prose_in_passthrough, corpus) == []


class TestL05:
    def test_a_forbidden_pronoun_is_found(self) -> None:
        """Vietnamese pronouns encode age and relationship, so choosing one is a
        claim about who is reading."""
        corpus = over("You can call it.", "Anh có thể gọi nó.")
        assert len(findings(language.l05_pronouns, corpus)) == 1

    def test_ban_where_the_english_says_you_is_allowed(self) -> None:
        corpus = over("You can call it.", "Bạn có thể gọi nó.")
        assert findings(language.l05_pronouns, corpus) == []

    def test_ban_where_the_english_is_impersonal_is_found(self) -> None:
        corpus = over("The list methods make this easy.", "Các phương thức giúp bạn dễ dàng.")
        assert len(findings(language.l05_pronouns, corpus)) == 1

    def test_ban_under_an_english_imperative_is_allowed(self) -> None:
        """ "See the ssl module for a list" is addressed to the reader as squarely
        as "you can see" and carries no pronoun at all."""
        corpus = over("See the :mod:`ssl` module.", "Hãy xem mô-đun :mod:`ssl` của bạn.")
        assert findings(language.l05_pronouns, corpus) == []

    def test_ban_is_not_confused_with_ban_meaning_a_copy(self) -> None:
        """Strip the tone marks and "bản dịch", a translation, becomes the
        pronoun. Of ninety-three findings that version produced, eighty-six were
        words with nothing to do with the reader."""
        corpus = over("The translation of this page.", "Bản dịch của trang này.")
        assert findings(language.l05_pronouns, corpus) == []

    def test_anh_is_not_confused_with_anh_meaning_an_image(self) -> None:
        corpus = over("The image is large.", "Ảnh này lớn.")
        assert findings(language.l05_pronouns, corpus) == []

    def test_sibling_class_is_not_two_pronouns(self) -> None:
        """ "anh em" is "siblings", and the super() documentation uses it."""
        corpus = over("A sibling class of type.", "Một lớp anh em của type.")
        assert findings(language.l05_pronouns, corpus) == []


class TestL06:
    def test_a_title_cased_heading_is_found(self) -> None:
        corpus = over("List Comprehensions", "Danh Sách Rút Gọn")
        assert len(findings(language.l06_headings, corpus)) == 1

    def test_a_hortative_heading_is_found(self) -> None:
        """A heading is a noun phrase naming a section, not an invitation."""
        corpus = over("Installing packages", "Hãy cài đặt gói")
        found = findings(language.l06_headings, corpus)
        assert any("instruction" in one.detail for one in found)

    def test_hay_meaning_or_does_not_read_as_the_hortative(self) -> None:
        """Without its tone mark "hãy" is "hay", which means "or"."""
        corpus = over("Numbers and strings", "Hay số hay chuỗi")
        assert findings(language.l06_headings, corpus) == []

    def test_a_sentence_case_heading_is_clean(self) -> None:
        corpus = over("Internet access", "Truy cập internet")
        assert findings(language.l06_headings, corpus) == []

    def test_a_sentence_is_not_a_heading(self) -> None:
        corpus = over("Return a list of items.", "Trả Về Một Danh Sách.")
        assert findings(language.l06_headings, corpus) == []


class TestL07:
    def test_an_entry_from_a_cut_down_model_is_found(self) -> None:
        """A route can be reconfigured and a proxy can fall back under load.
        Without this the only evidence would be a reviewer noticing that a few
        hundred entries read worse than the rest."""
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        memory = Memory(
            [machine_segment("Return a list.", "Trả về một danh sách.", model="gpt-5-mini")]
        )
        assert len(findings(language.l07_no_cut_down_model, corpus_of(one, memory=memory))) == 1

    def test_the_model_the_run_asked_for_is_clean(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.", model="gpt-5")])
        assert findings(language.l07_no_cut_down_model, corpus_of(one, memory=memory)) == []

    def test_a_human_translation_is_not_judged_on_a_model_name(self) -> None:
        one = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        assert findings(language.l07_no_cut_down_model, corpus_of(one)) == []


class TestL08:
    def test_a_lost_half_of_a_paragraph_is_found(self) -> None:
        """An answer that loses the second half passes every placeholder rule,
        reads fluently, and is missing what the reader needed."""
        english = "One. Two. Three. Four."
        assert len(findings(language.l08_sentence_parity, over(english, "Một."))) == 1

    def test_one_sentence_of_slack_is_allowed(self) -> None:
        """Vietnamese genuinely splits and joins sentences."""
        assert findings(language.l08_sentence_parity, over("One. Two.", "Một, hai.")) == []
