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
