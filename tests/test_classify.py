from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import classify, sync
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

    def test_an_example_quoted_from_the_middle(self) -> None:
        """No prompt in it, only continuations. tutorial/errors.po has one and
        it cost three model calls in the first tier 1 run before dying."""
        assert classify.is_doctest("... except RuntimeError, TypeError:\n...     pass")

    def test_one_line_of_ellipsis_is_not_an_example(self) -> None:
        """25 entries open with an ellipsis and most pick up a sentence from
        the heading above them, so a single line of it proves nothing."""
        assert not classify.is_doctest("... install packages just for the current user?")


class TestLiteralBlock:
    def test_a_c_function_starting_at_column_zero(self) -> None:
        """The design notes wanted every line indented, which selects nothing:
        a code block here opens flush left and indents from line two."""
        msgid = "static PyObject *\nf(PyObject *self)\n{\n    return NULL;\n}"
        assert classify.classify(msgid) is Kind.LITERAL_BLOCK

    def test_one_indented_line_alone_is_not_a_block(self) -> None:
        assert not classify.is_literal_block("    indented but alone")

    def test_a_wrapped_paragraph_is_not_a_block(self) -> None:
        assert not classify.is_literal_block("A sentence\nthat wrapped across lines.")


class TestCodeWithNoIndent:
    """The 26 tier 1 entries the indent rule called prose.

    Every one went to a model, came back unchanged because unchanged was
    correct, and was then refused by ``P05`` for being identical to the source
    and by ``P08`` for having no Vietnamese in it. Those two rules were 161 of
    the 206 refusals in that run.
    """

    @pytest.mark.parametrize(
        "msgid",
        [
            "#!/usr/bin/env python3",
            "x = MyClass()",
            "squares = [x**2 for x in range(10)]",
            '__all__ = ["echo", "surround", "reverse"]',
            "import sound.effects.echo",
            "from sound.effects.echo import echofilter",
            "from . import echo",
            "echofilter(input, output, delay=0.7, atten=4)",
            "json.dump(x, f)",
            "['demo.py', 'one', 'two', 'three']",
            "python -m venv tutorial-env",
            "source tutorial-env/bin/activate",
            "tutorial-env\\Scripts\\activate",
            "$ chmod +x myscript.py",
        ],
    )
    def test_a_line_of_code_alone_is_a_block(self, msgid: str) -> None:
        assert classify.classify(msgid) is Kind.LITERAL_BLOCK

    def test_several_flush_left_lines_of_code(self) -> None:
        msgid = (
            "import sound.effects.echo\nimport sound.effects.surround\nfrom sound.effects import *"
        )
        assert classify.is_literal_block(msgid)

    @pytest.mark.parametrize(
        "msgid",
        [
            "(Contributed by Eddie Elizondo in :issue:`35810`.)",
            "`2002::/16` is considered private.",
            "I/O control",
            "popen() (in module os)",
            "Return a new dictionary.",
            "> (greater)",
        ],
    )
    def test_prose_that_an_earlier_draft_swallowed(self, msgid: str) -> None:
        """Each of these was a false positive of a looser rule. A missed block
        costs one call. A paragraph mistaken for code is English left in a
        Vietnamese page, and no later stage would ever say so."""
        assert not classify.is_literal_block(msgid)
        assert classify.classify(msgid).translatable

    def test_a_call_is_spaced_differently_from_a_sentence(self) -> None:
        """What separates ``sorted(d.keys())`` from ``popen() (in module os)``
        is that code puts a space after a comma and nowhere else."""
        assert classify.is_literal_block("sorted(d.keys(), key=len)")
        assert not classify.is_literal_block("sorted() (a builtin function)")

    def test_one_prompt_makes_the_whole_entry_a_transcript(self) -> None:
        """The lines around a prompt are the output of the command in it, and
        on their own they look like nothing at all."""
        assert classify.is_literal_block("$ python fibo.py 50\n0 1 1 2 3 5 8 13 21 34")

    def test_a_prompt_wearing_a_virtualenv_name(self) -> None:
        msgid = "(tutorial-env) $ python -m pip list\nnovas (3.1.1.3)\nnumpy (1.9.2)"
        assert classify.is_literal_block(msgid)


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


class TestTheRealCorpus:
    @pytest.mark.corpus
    def test_the_code_rules_stay_within_a_percent_of_the_corpus(self, upstream: Path) -> None:
        """The number that decides whether this rule is safe.

        It moved 917 entries, one per cent of 87 008, out of prose. A rule that
        grew to swallow ten per cent would be quietly deciding that a tenth of
        the documentation needs no translation, and nothing downstream would
        ever report that, because a passthrough entry looks exactly like a
        finished one.
        """
        msgids = [entry.msgid for one in sync.read_corpus(upstream) for entry in one.entries]
        counts = classify.counts(msgids)
        assert counts.total > 80_000
        assert counts.literal_block / counts.total < 0.05

    @pytest.mark.corpus
    def test_the_tutorial_code_a_real_run_wasted_calls_on(self, upstream: Path) -> None:
        """Tier 1 held 26 entries that three model calls each could not fix.
        All but two of them were this, and this is now free."""
        msgids = [entry.msgid for one in sync.read_corpus(upstream) for entry in one.entries]
        for msgid in ("import sound.effects.echo", "source tutorial-env/bin/activate"):
            assert msgid in msgids
            assert not classify.classify(msgid).translatable


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
