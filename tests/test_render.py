import pytest

from pydocvi import batch, render
from pydocvi.catalog import Entry
from pydocvi.glossary import Glossary, Term


def make(path: str, *msgids: str) -> batch.Batch:
    entries = [Entry(msgid=msgid) for msgid in msgids]
    return next(iter(batch.pack(path, batch.items(entries))))


def glossary(*terms: Term) -> Glossary:
    return Glossary(version=1, terms=terms)


ITERABLE = Term(en="iterable", vi="iterable", keep_en=True)
FUNCTION = Term(en="function", vi="hàm")


class TestTheShippedFile:
    def test_the_prompt_is_a_file_in_the_package(self) -> None:
        """Not a string literal, which is what lets a prompt change be reviewed
        as a diff and printed by ``prompt show`` without a fleet."""
        assert "⟦1⟧" in render.template()

    def test_the_hash_is_of_the_file_and_does_not_move(self) -> None:
        assert render.fingerprint() == render.fingerprint()
        assert len(render.fingerprint()) == 64

    def test_a_prompt_with_nowhere_to_put_the_terminology_is_refused(self) -> None:
        with pytest.raises(render.RenderError):
            render.fill("A prompt that forgot its terminology section.", (FUNCTION,))

    def test_the_shipped_prompt_has_somewhere_to_put_it(self) -> None:
        assert render.SLOT in render.template()

    def test_the_slot_is_gone_once_the_terminology_is_in(self) -> None:
        assert render.SLOT not in render.system((FUNCTION,))


class TestTerminology:
    def test_a_translated_row_is_written_as_a_rendering(self) -> None:
        assert render.terminology([FUNCTION]) == "- function -> hàm"

    def test_a_keep_en_row_says_what_to_do_rather_than_showing_a_no_op(self) -> None:
        """``iterable -> iterable`` reads as a mistake. The glossary's decision
        is that the English word is the Vietnamese term, so the prompt has to
        say so."""
        assert render.terminology([ITERABLE]) == f"- iterable -> {render.KEEP}"

    def test_a_note_travels_with_the_row(self) -> None:
        term = Term(en="type", vi="type", keep_en=True, note="not kiểu in this sense")
        assert "not kiểu in this sense" in render.terminology([term])

    def test_no_matching_row_says_so_rather_than_leaving_a_blank_heading(self) -> None:
        assert render.terminology([]) == render.NO_TERMS
        assert render.NO_TERMS in render.system(())


class TestSelection:
    def test_only_the_rows_the_batch_needs_are_sent(self) -> None:
        one = make("f.po", "Call the function.")
        assert render.terms_for(one, glossary(FUNCTION, ITERABLE)) == (FUNCTION,)

    def test_a_row_is_sent_once_however_many_entries_use_it(self) -> None:
        one = make("f.po", "The function is here.", "That function is there.")
        assert render.terms_for(one, glossary(FUNCTION)) == (FUNCTION,)

    def test_a_term_only_inside_markup_puts_no_row_in_the_prompt(self) -> None:
        """The model never sees inside a marker, so a row it cannot act on
        would only dilute the rows it can."""
        one = make("f.po", "See :func:`function` for this.")
        assert render.terms_for(one, glossary(FUNCTION)) == ()


class TestTheUserMessage:
    def test_entries_are_numbered_from_one(self) -> None:
        one = make("f.po", "First.", "Second.")
        assert render.user(one).splitlines()[-2:] == ["1 First.", "2 Second."]

    def test_the_markup_is_gone_and_the_markers_are_there(self) -> None:
        one = make("f.po", "Return :func:`len` of it.")
        assert ":func:" not in render.user(one)
        assert "1 Return ⟦1⟧ of it." in render.user(one)

    def test_the_first_line_names_the_file_and_the_kind_of_writing(self) -> None:
        assert render.user(make("tutorial/x.po", "Hi.")).startswith(
            "These strings are from tutorial/x.po, part of the tutorial"
        )

    def test_a_file_at_the_top_of_the_tree_still_gets_a_context_line(self) -> None:
        assert render.area("bugs.po") == render.ROOT
        assert "bugs.po" in render.context_line("bugs.po")

    def test_a_directory_nobody_listed_falls_back_rather_than_failing(self) -> None:
        assert render.area("sphinx/internals.po") == render.ROOT

    def test_the_segment_id_is_not_in_the_prompt(self) -> None:
        """40 characters of hex an entry the model has no use for and can get
        wrong."""
        one = make("f.po", "Hello.")
        assert one.items[0].segment not in render.user(one)


class TestRendering:
    def test_a_batch_renders_to_both_messages_and_the_hash(self) -> None:
        prompt = render.render(make("tutorial/x.po", "Call the function."), glossary(FUNCTION))
        assert "hàm" in prompt.system
        assert prompt.user.endswith("1 Call the function.")
        assert prompt.terms == (FUNCTION,)
        assert prompt.fingerprint == render.fingerprint()

    def test_the_same_batch_renders_to_the_same_bytes(self) -> None:
        """A prompt change should be visible as a hash change and never as
        anything else."""
        one = make("f.po", "Call the function.")
        assert render.render(one, glossary(FUNCTION)) == render.render(one, glossary(FUNCTION))

    def test_the_characters_counted_are_both_messages(self) -> None:
        prompt = render.render(make("f.po", "Hello."), glossary())
        assert prompt.characters == len(prompt.system) + len(prompt.user)
