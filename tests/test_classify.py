import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import classify
from pydocvi.classify import Kind


class TestNoop:
    @pytest.mark.parametrize(
        "msgid",
        [
            ":mod:`os.path`",
            "``None``",
            ":func:`f` :func:`g`",
            "3.14",
            "%s",
            "https://example.com/",
        ],
    )
    def test_entries_with_no_prose_left(self, msgid: str) -> None:
        assert classify.is_noop(msgid)

    @pytest.mark.parametrize(
        "msgid",
        ["the :mod:`os.path` module", "Pass ``None`` to skip.", "Deprecated since version 3.14."],
    )
    def test_entries_with_prose_around_the_markup(self, msgid: str) -> None:
        assert not classify.is_noop(msgid)

    def test_a_single_letter_is_not_a_word(self) -> None:
        """One letter beside markup is a label, not a sentence."""
        assert classify.is_noop(":class:`x` a")

    def test_the_rule_is_biased_towards_translating(self) -> None:
        """A false negative costs one wasted call. A false positive leaves an
        English sentence in the corpus wearing a translation's clothes."""
        assert not classify.is_noop("or")


class TestDoctest:
    def test_an_interactive_example(self) -> None:
        assert classify.classify(">>> sorted([3, 1, 2])\n[1, 2, 3]") is Kind.DOCTEST

    def test_a_leading_blank_line_does_not_hide_it(self) -> None:
        assert classify.is_doctest("\n>>> f()")

    def test_prose_mentioning_the_prompt_is_not_a_doctest(self) -> None:
        assert not classify.is_doctest("Type >>> to get a prompt.")

    def test_an_empty_string_is_not_a_doctest(self) -> None:
        assert not classify.is_doctest("")


class TestLiteralBlock:
    def test_a_c_function_starting_at_column_zero(self) -> None:
        """The design notes wanted every line indented, which selects nothing:
        a code block here opens flush left and indents from line two."""
        msgid = "static PyObject *\nf(PyObject *self)\n{\n    return NULL;\n}"
        assert classify.classify(msgid) is Kind.LITERAL_BLOCK

    def test_a_single_line_is_never_a_block(self) -> None:
        assert not classify.is_literal_block("    indented but alone")

    def test_a_wrapped_paragraph_is_not_a_block(self) -> None:
        assert not classify.is_literal_block("A sentence\nthat wrapped across lines.")


class TestVersionMarker:
    @pytest.mark.parametrize("msgid", ["3.14", "3", "3.14.0", "asyncio", "os.path", "_thread"])
    def test_bare_versions_and_identifiers(self, msgid: str) -> None:
        assert classify.is_version_marker(msgid)

    @pytest.mark.parametrize("msgid", ["version 3.14", "3.14 and later", "", "a b"])
    def test_anything_with_a_second_token(self, msgid: str) -> None:
        assert not classify.is_version_marker(msgid)


class TestClassify:
    def test_prose_is_the_only_translatable_kind(self) -> None:
        assert Kind.PROSE.translatable
        assert not any(kind.translatable for kind in Kind if kind is not Kind.PROSE)

    def test_ordinary_prose(self) -> None:
        assert classify.classify("Return the sorted list.") is Kind.PROSE

    def test_a_doctest_wins_over_the_block_rule(self) -> None:
        """Most doctests also have an indented continuation line, and the more
        specific answer is the useful one to report."""
        assert classify.classify(">>> f(\n...     x)\n") is Kind.DOCTEST

    def test_counts_add_up(self) -> None:
        counts = classify.counts(["Return a value.", ":mod:`os`", ">>> f()", "3.14"])
        assert counts.total == 4
        assert counts.prose == 1
        assert counts.passthrough == 3

    def test_counts_of_nothing(self) -> None:
        assert classify.counts([]).total == 0


class TestProperties:
    @given(st.text())
    @settings(max_examples=500)
    def test_every_string_gets_exactly_one_kind(self, value: str) -> None:
        """The classifier runs over 87 008 entries before anything else does, so
        an input it refuses to answer for is an outage rather than a bug."""
        assert classify.classify(value) in set(Kind)

    @given(st.lists(st.text(), max_size=20))
    @settings(max_examples=200)
    def test_the_counts_partition_the_input(self, values: list[str]) -> None:
        counts = classify.counts(values)
        assert counts.total == len(values)
        assert counts.prose + counts.passthrough == counts.total
