import pytest

from pydocvi import textguard


class TestEnglishNarration:
    @pytest.mark.parametrize(
        "text",
        [
            "Here is the translation: Trả về danh sách.",
            "I have translated the strings below.",
            "I cannot translate this string.",
            "Sorry, this one is unclear.",
            "As an AI, I should note this.",
            "Let me know if you need changes.",
            "Hope this helps!",
        ],
    )
    def test_a_model_talking_instead_of_translating(self, text: str) -> None:
        assert not textguard.clean(text)


class TestVietnameseNarration:
    @pytest.mark.parametrize(
        "text",
        [
            "Đây là bản dịch của chuỗi trên.",
            "Dưới đây là các câu đã dịch.",
            "Tôi không thể dịch chuỗi này.",
            "Xin lỗi, chuỗi này khó hiểu.",
            "Bản dịch: Trả về danh sách.",
        ],
    )
    def test_the_half_a_guard_written_in_english_would_miss(self, text: str) -> None:
        assert not textguard.clean(text)


class TestFencesAndAsides:
    def test_a_fenced_answer_is_narration_of_a_kind(self) -> None:
        assert not textguard.clean("```\nTrả về danh sách.\n```")

    def test_a_horizontal_rule_counts_too(self) -> None:
        assert not textguard.clean("---\nTrả về danh sách.")

    def test_a_translator_note_in_brackets(self) -> None:
        assert not textguard.clean("Trả về danh sách. (translator's note: unclear)")

    def test_a_vietnamese_translator_note(self) -> None:
        assert not textguard.clean("Trả về danh sách. [ghi chú của người dịch]")


class TestCleanTranslations:
    @pytest.mark.parametrize(
        "text",
        [
            "Trả về một danh sách các phần tử đã sắp xếp.",
            "Kể từ phiên bản 3.14, hàm này không còn được dùng.",
            "Sử dụng ⟦1⟧ để mở tệp.",
            "Đối tượng này không thể thay đổi.",
            "",
        ],
    )
    def test_ordinary_vietnamese_passes(self, text: str) -> None:
        assert textguard.clean(text)

    def test_the_word_note_inside_a_sentence_is_not_narration(self) -> None:
        """A long phrase list becomes a filter that occasionally deletes a real
        translation, and a wrongly refused entry is worse than a narrated one."""
        assert textguard.clean("Ghi chú này áp dụng cho mọi phiên bản.")


class TestReporting:
    def test_only_the_first_phrase_is_reported(self) -> None:
        found = textguard.find("Sorry. Here is the translation. Hope this helps.")
        assert found is not None
        assert found.phrase.lower() == "sorry"

    def test_a_finding_says_where_it_was(self) -> None:
        found = textguard.find("Trả về danh sách. Hope this helps.")
        assert found is not None
        assert found.where > 0
        assert "offset" in str(found)


class TestWhatTheSourceLicenses:
    """The half of this module that was missing until a real run found it.

    Every case here is a faithful translation that the guard used to refuse. The
    first one is the string itself: it opened a batch of 28 in
    ``library/functions.po``, and ``P06`` rejects the whole batch, so one wrong
    answer here threw away 27 correct translations beside it.
    """

    def test_a_note_in_the_source_licenses_a_note_in_the_translation(self) -> None:
        msgid = "Note: Unlike :func:`iter`, :func:`aiter` has no 2-argument variant."
        msgstr = "Lưu ý: Không giống như ⟦1⟧, ⟦2⟧ không có biến thể 2 đối số."
        assert not textguard.clean(msgstr)
        assert textguard.clean(msgstr, msgid)

    def test_the_following_licenses_duoi_day_la(self) -> None:
        assert textguard.clean("Dưới đây là các tùy chọn.", "The following are the options.")

    def test_hope_licenses_hy_vong(self) -> None:
        assert textguard.clean("Chúng tôi hy vọng điều này.", "We hope this.")

    def test_sorry_licenses_xin_loi(self) -> None:
        assert textguard.clean("Xin lỗi, không tìm thấy.", "Sorry, not found.")

    def test_a_source_that_says_nothing_of_the_kind_licenses_nothing(self) -> None:
        """The licence is the source raising the subject, not the source
        existing. A string about sorting does not excuse an apology."""
        assert not textguard.clean("Xin lỗi, chuỗi này khó hiểu.", "Return a sorted list.")

    def test_a_phrase_with_no_licence_fires_however_the_source_reads(self) -> None:
        """Some phrases are never a translation of anything. The English
        documentation does not say "as an AI", so nothing can excuse it."""
        assert not textguard.clean("As an AI, I note this.", "As an AI system would note that.")

    def test_a_fence_the_source_did_not_open_is_the_model_presenting(self) -> None:
        assert not textguard.clean("```\nTrả về danh sách.\n```", "Return a list.")

    def test_a_fence_the_source_opened_is_licensed_like_any_other_phrase(self) -> None:
        """This asserted the opposite until August 2026, on the grounds that a
        literal block reaches this module as a placeholder and never as three
        backticks. True of the translate path and not of the audit, which hands
        over the raw pair: ``L03`` reported the ``---`` in ``c-api/call.po`` and
        the class diagram in ``howto/mro.po`` as a model drawing a rule, when
        both are the English copied through exactly."""
        assert textguard.clean("---", "---")
        assert textguard.clean(" ----\n| O  |", " ----\n| O  |")

    def test_a_fence_with_no_source_is_still_narration(self) -> None:
        """Passing no source licenses nothing, which is the right answer for a
        caller holding a string and no idea what it was made from."""
        assert not textguard.clean("---\nTrả về danh sách.")

    def test_the_leftmost_phrase_is_the_one_reported_not_the_first_listed(self) -> None:
        found = textguard.find("Trả về danh sách. Xin lỗi. Here is the translation.")
        assert found is not None
        assert found.phrase.lower() == "xin lỗi"

    def test_the_contraction_licenses_it_too(self) -> None:
        """From ``library/functions.po``. "Here's an example" is how the
        documentation opens a code block, and it was the last human translation
        in the corpus this guard still refused."""
        assert textguard.clean(
            "Dưới đây là ví dụ tính nghịch đảo của ``38`` theo modulo ``97``::",
            "Here's an example of computing an inverse for ``38`` modulo ``97``::",
        )
