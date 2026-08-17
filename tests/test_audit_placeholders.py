"""``P01`` to ``P08`` over hand-written entries.

The fixtures are short and in the same typographic shape as the corpus. None of
them is a long CPython string, because a test that needs forty lines of context
to say what it is testing is a test nobody reads when it fails.
"""

from conftest import catalog_of, corpus_of, entry, findings
from pydocvi.audit import placeholders


def over(msgid: str, msgstr: str, **overrides: object) -> object:
    return corpus_of(catalog_of(entry(msgid, msgstr, **overrides)))


class TestP01:
    def test_a_dropped_span_is_found(self) -> None:
        corpus = over("Return a :class:`list`.", "Trả về một danh sách.")
        assert len(findings(placeholders.p01_spans_survive, corpus)) == 1

    def test_a_span_that_survived_is_not_reported(self) -> None:
        corpus = over("Return a :class:`list`.", "Trả về một :class:`list`.")
        assert findings(placeholders.p01_spans_survive, corpus) == []

    def test_moving_a_span_is_legal(self) -> None:
        """Vietnamese puts modifiers on the other side of what they modify, so a
        rule against moving a span would be a rule against translating well."""
        corpus = over("The :mod:`os` module and ``sys``.", "``sys`` và mô-đun :mod:`os` đó.")
        assert findings(placeholders.p01_spans_survive, corpus) == []

    def test_a_span_repeated_in_the_translation_is_found(self) -> None:
        corpus = over("Use ``sys``.", "Dùng ``sys`` và ``sys``.")
        assert len(findings(placeholders.p01_spans_survive, corpus)) == 1

    def test_a_span_that_appears_twice_upstream_must_appear_twice(self) -> None:
        """Counted, not merely present, because a set comparison reports a span
        that went from two occurrences to one as nothing at all."""
        corpus = over("``a`` then ``a`` again.", "``a`` rồi thôi.")
        assert len(findings(placeholders.p01_spans_survive, corpus)) == 1

    def test_the_finding_names_both_counts(self) -> None:
        corpus = over("Use ``sys``.", "Dùng nó.")
        assert "0 time(s)" in findings(placeholders.p01_spans_survive, corpus)[0].detail


class TestP02:
    def test_an_unrestored_marker_is_found(self) -> None:
        """A marker in the corpus renders as two characters nobody can type."""
        corpus = over("Return a :class:`list`.", "Trả về một ⟦1⟧.")
        assert len(findings(placeholders.p02_no_marker_survives, corpus)) == 1

    def test_a_restored_entry_is_clean(self) -> None:
        corpus = over("Return a :class:`list`.", "Trả về một :class:`list`.")
        assert findings(placeholders.p02_no_marker_survives, corpus) == []


class TestP03:
    def test_a_translated_target_is_found(self) -> None:
        """A translated target renders as a link to nothing and a reader
        following it lands nowhere."""
        corpus = over("See :func:`len`.", "Xem :func:`chiều dài`.")
        found = findings(placeholders.p03_role_targets_unchanged, corpus)
        assert len(found) == 1
        assert ":func:`len`" in found[0].detail

    def test_the_finding_names_what_came_back_instead(self) -> None:
        corpus = over("See :func:`len`.", "Xem :func:`chiều dài`.")
        assert findings(placeholders.p03_role_targets_unchanged, corpus)[0].got == (
            ":func:`chiều dài`"
        )

    def test_an_unchanged_target_is_clean(self) -> None:
        corpus = over("See :func:`len`.", "Xem :func:`len`.")
        assert findings(placeholders.p03_role_targets_unchanged, corpus) == []

    def test_a_role_with_a_display_text_keeps_its_target(self) -> None:
        corpus = over("See :ref:`the tutorial <tut>`.", "Xem :ref:`hướng dẫn <tut>`.")
        assert len(findings(placeholders.p03_role_targets_unchanged, corpus)) == 1

    def test_a_dropped_role_is_found(self) -> None:
        corpus = over("See :func:`len`.", "Xem hàm đó.")
        assert len(findings(placeholders.p03_role_targets_unchanged, corpus)) == 1


class TestP04:
    def test_a_lost_leading_space_is_found(self) -> None:
        """gettext concatenates these into rendered pages, so a lost space is two
        words run together a long way from here."""
        corpus = over(" Footnotes", "Chú thích")
        assert len(findings(placeholders.p04_whitespace_matches, corpus)) == 1

    def test_an_added_leading_space_is_found(self) -> None:
        corpus = over("Footnotes", " Chú thích")
        assert len(findings(placeholders.p04_whitespace_matches, corpus)) == 1

    def test_a_lost_trailing_newline_is_found(self) -> None:
        corpus = over("A line.\n", "Một dòng.")
        assert len(findings(placeholders.p04_whitespace_matches, corpus)) == 1

    def test_matching_edges_are_clean(self) -> None:
        corpus = over(" A line.\n", " Một dòng.\n")
        assert findings(placeholders.p04_whitespace_matches, corpus) == []


class TestP05:
    def test_a_dropped_specifier_is_found(self) -> None:
        """A dropped %s is a TypeError in whatever program copied the string,
        which makes this the one rule whose failures escape the docs entirely."""
        corpus = over("Cannot open %s for reading.", "Không thể mở tệp để đọc.")
        assert len(findings(placeholders.p05_format_specifiers_match, corpus)) == 1

    def test_a_preserved_specifier_is_clean(self) -> None:
        corpus = over("Cannot open %s.", "Không thể mở %s.")
        assert findings(placeholders.p05_format_specifiers_match, corpus) == []

    def test_reordering_two_specifiers_is_legal(self) -> None:
        corpus = over("%s wants %d.", "%d là cái %s cần.")
        assert findings(placeholders.p05_format_specifiers_match, corpus) == []


class TestP06:
    def test_a_translated_url_is_found(self) -> None:
        corpus = over(
            "See the `docs <https://docs.python.org/>`_.",
            "Xem `tài liệu <https://tài-liệu.python.org/>`_.",
        )
        assert len(findings(placeholders.p06_link_targets_unchanged, corpus)) == 1

    def test_translating_only_the_visible_text_is_correct(self) -> None:
        corpus = over(
            "See the `docs <https://docs.python.org/>`_.",
            "Xem `tài liệu <https://docs.python.org/>`_.",
        )
        assert findings(placeholders.p06_link_targets_unchanged, corpus) == []

    def test_the_finding_says_what_is_missing(self) -> None:
        corpus = over("See the `docs <https://a/>`_.", "Xem tài liệu.")
        assert "missing" in findings(placeholders.p06_link_targets_unchanged, corpus)[0].detail


class TestP07:
    DOCTEST = ">>> len([1, 2])\n2"

    def test_a_translated_comment_inside_a_doctest_is_found(self) -> None:
        """This is the classifier bug seen from the other end: an entry a model
        was asked to translate and should never have been shown."""
        corpus = over(
            ">>> x = [1]  # a list\n",
            ">>> x = [1]  # một danh sách\n",
        )
        assert len(findings(placeholders.p07_code_is_byte_identical, corpus)) == 1

    def test_a_copied_doctest_is_clean(self) -> None:
        corpus = over(self.DOCTEST, self.DOCTEST)
        assert findings(placeholders.p07_code_is_byte_identical, corpus) == []

    def test_prose_is_not_this_check_s_business(self) -> None:
        corpus = over("Return a list.", "Trả về một danh sách.")
        assert findings(placeholders.p07_code_is_byte_identical, corpus) == []


class TestP08:
    def test_a_fenced_answer_is_found(self) -> None:
        """A fence is a model formatting its answer rather than answering."""
        corpus = over("Return a list.", "```\nTrả về một danh sách.\n```")
        assert len(findings(placeholders.p08_no_fence, corpus)) == 1

    def test_a_horizontal_rule_is_found(self) -> None:
        corpus = over("Return a list.", "---\nTrả về một danh sách.")
        assert len(findings(placeholders.p08_no_fence, corpus)) == 1

    def test_an_ordinary_translation_is_clean(self) -> None:
        corpus = over("Return a list.", "Trả về một danh sách.")
        assert findings(placeholders.p08_no_fence, corpus) == []


class TestEveryCheckReportsWhereItLooked:
    def test_a_finding_carries_the_file_the_line_and_the_segment(self) -> None:
        """A finding a reviewer has to go and look up is a finding a reviewer
        skips, and there are 87 008 entries to skip."""
        one = entry("See :func:`len`.", "Xem :func:`chiều dài`.", line=42)
        corpus = corpus_of(catalog_of(one, name="library/functions.po"))
        found = findings(placeholders.p03_role_targets_unchanged, corpus)[0]
        assert found.path == "library/functions.po"
        assert found.line == 42
        assert found.segment == one.id
