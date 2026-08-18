"""The row model, the matcher, the two list rules and the two files.

The matcher tests are the ones that matter. There is one matcher and three
callers, so a test here that passes for the wrong reason is a wrong terminology
decision applied to 87 008 entries with nobody having decided anything.
"""

import pytest

from pydocvi import glossary, stale
from pydocvi.catalog import segment_id
from pydocvi.glossary import Glossary, GlossaryError, Matcher, Term
from pydocvi.memory import Memory, Segment


def term(en: str, vi: str, **overrides: object) -> Term:
    return Term(en=en, vi=vi, **overrides)  # type: ignore[arg-type]


def make(*terms: Term, version: int = 1) -> Glossary:
    return Glossary(version=version, terms=terms)


#: The pair the whole longest-first rule exists for.
NESTED = (
    term("context manager", "trình quản lý ngữ cảnh"),
    term("context", "ngữ cảnh"),
)


class TestTerm:
    def test_a_term_with_no_english_side_is_refused(self):
        with pytest.raises(ValueError, match="not a term"):
            term("  ", "ngữ cảnh")

    def test_a_term_with_no_rendering_is_refused(self):
        with pytest.raises(ValueError, match="no rendering"):
            term("context", "   ")

    def test_a_translated_row_expects_the_vietnamese(self):
        assert term("iterable", "khả lặp").rendering == "khả lặp"

    def test_a_kept_row_expects_the_english(self):
        assert term("decorator", "decorator", keep_en=True).rendering == "decorator"

    def test_a_row_without_context_is_always_in_force(self):
        assert term("list", "danh sách").applies()

    def test_a_context_matches_the_path(self):
        row = term("list", "danh sách", context="library/stdtypes")
        assert row.applies(where="library/stdtypes.po")
        assert not row.applies(where="tutorial/introduction.po")

    def test_a_context_matches_the_msgctxt(self):
        row = term("list", "danh sách", context="builtin")
        assert row.applies(msgctxt="builtin type")
        assert not row.applies(msgctxt="the html tag")

    def test_a_context_is_matched_case_insensitively(self):
        assert term("list", "danh sách", context="StdTypes").applies(where="library/stdtypes.po")

    def test_a_context_matches_nothing_when_nothing_was_given(self):
        assert not term("list", "danh sách", context="library").applies()


class TestGlossary:
    def test_terms_are_reachable_by_english(self):
        assert make(*NESTED).get("context").vi == "ngữ cảnh"

    def test_an_absent_term_is_none_rather_than_an_error(self):
        assert make(*NESTED).get("generator") is None

    def test_kept_and_translated_rows_are_separable(self):
        rows = make(term("decorator", "decorator", keep_en=True), term("list", "danh sách"))
        assert [row.en for row in rows.kept] == ["decorator"]
        assert [row.en for row in rows.translated] == ["list"]

    def test_a_kept_row_stands_alone_without_being_told_to(self):
        """A term that is English in every sentence is English on its own too,
        so the flag is only ever needed by rows that carry a rendering."""
        rows = make(term("decorator", "decorator", keep_en=True), term("list", "danh sách"))
        assert [row.en for row in rows.standalone] == ["decorator"]

    def test_a_translated_row_can_still_stand_alone(self):
        """The case the field was added for. ``float`` is ``số thực`` in a
        sentence and the name of a C type in the ``struct`` format table."""
        rows = make(term("float", "số thực", identifier=True), term("list", "danh sách"))
        assert [row.en for row in rows.standalone] == ["float"]
        assert [row.en for row in rows.translated] == ["float", "list"]
        assert rows.kept == ()

    def test_len_counts_rows(self):
        assert len(make(*NESTED)) == 2

    def test_iterating_yields_the_rows_in_file_order(self):
        assert [row.en for row in make(*NESTED)] == ["context manager", "context"]

    def test_replacing_the_rows_bumps_the_version(self):
        after = make(*NESTED).with_terms([*NESTED, term("iterable", "khả lặp")])
        assert after.version == 2

    def test_replacing_the_rows_with_the_same_rows_does_not_bump(self):
        after = make(*NESTED).with_terms(NESTED)
        assert after.version == 1

    def test_an_explicit_version_wins_over_the_bump(self):
        assert make(*NESTED).with_terms([], version=9).version == 9


class TestMatchOrder:
    def test_the_longer_term_comes_first(self):
        order = glossary.match_order(NESTED[::-1])
        assert [row.en for row in order] == ["context manager", "context"]

    def test_ties_are_broken_alphabetically_so_the_order_is_a_function_of_the_rows(self):
        rows = [term("beta", "b"), term("alpha", "a")]
        assert [row.en for row in glossary.match_order(rows)] == ["alpha", "beta"]

    def test_an_empty_glossary_has_an_empty_order(self):
        assert glossary.match_order([]) == ()


class TestMatching:
    def test_a_term_in_the_prose_is_found(self):
        assert Matcher(make(*NESTED))("a context manager protocol") == {"context manager"}

    def test_the_longer_term_wins_the_position_the_shorter_would_have_taken(self):
        found = Matcher(make(*NESTED))("use a context manager here")
        assert found == {"context manager"}

    def test_the_shorter_term_still_matches_where_the_longer_does_not(self):
        assert Matcher(make(*NESTED))("the context of the call") == {"context"}

    def test_both_match_when_both_appear(self):
        found = Matcher(make(*NESTED))("a context manager changes the context")
        assert found == {"context manager", "context"}

    def test_file_order_does_not_change_what_matches(self):
        forwards = Matcher(make(*NESTED))("a context manager")
        backwards = Matcher(make(*NESTED[::-1]))("a context manager")
        assert forwards == backwards == {"context manager"}

    def test_a_term_inside_another_word_does_not_match(self):
        assert Matcher(make(term("list", "danh sách")))("listen to the sublist") == frozenset()

    def test_a_hyphen_is_a_boundary_so_a_hyphenated_word_is_not_split(self):
        rows = make(term("built-in", "tích hợp sẵn"), term("built", "đã dựng"))
        assert Matcher(rows)("a built-in function") == {"built-in"}

    def test_an_underscore_is_a_boundary_so_an_identifier_is_not_a_term(self):
        assert Matcher(make(term("list", "danh sách")))("call list_of_things") == frozenset()

    def test_the_first_letter_may_differ_in_case(self):
        assert Matcher(make(term("iterable", "khả lặp")))("Iterable objects") == {"iterable"}

    def test_the_rest_of_the_term_may_not_differ_in_case(self):
        assert Matcher(make(term("iterable", "khả lặp")))("ITERABLE objects") == frozenset()

    def test_an_acronym_is_not_matched_by_its_lowercase_spelling(self):
        assert Matcher(make(term("URL", "URL", keep_en=True)))("a url here") == frozenset()

    def test_a_term_inside_an_inline_literal_is_code_and_does_not_count(self):
        found = Matcher(make(term("list", "danh sách")))("the ``list`` builtin")
        assert found == frozenset()

    def test_a_term_inside_a_role_is_code_and_does_not_count(self):
        found = Matcher(make(term("list", "danh sách")))("see :func:`list` for more")
        assert found == frozenset()

    def test_a_term_in_the_visible_text_of_a_link_does_count(self):
        rows = make(term("tutorial", "hướng dẫn"))
        assert Matcher(rows)("the `tutorial <https://x/y>`_ explains") == {"tutorial"}

    def test_an_empty_glossary_matches_nothing_without_building_a_pattern(self):
        assert Matcher(make())("a context manager") == frozenset()

    def test_a_term_is_reported_once_however_often_it_appears(self):
        rows = make(term("list", "danh sách"))
        assert Matcher(rows)("a list, another list, a third list") == {"list"}

    def test_a_term_at_the_very_start_and_end_of_the_string_matches(self):
        assert Matcher(make(term("list", "danh sách")))("list") == {"list"}

    def test_a_regex_metacharacter_in_a_term_is_a_literal(self):
        assert Matcher(make(term("f(x)", "f(x)", keep_en=True)))("call f(x) now") == {"f(x)"}


class TestContextualMatching:
    def test_a_contextual_row_matches_where_its_context_holds(self):
        rows = make(term("list", "danh sách", context="stdtypes"))
        assert Matcher(rows)("a list here", where="library/stdtypes.po") == {"list"}

    def test_a_contextual_row_is_silent_elsewhere(self):
        rows = make(term("list", "danh sách", context="stdtypes"))
        assert Matcher(rows)("a list here", where="tutorial/intro.po") == frozenset()

    def test_the_uncontextualised_row_still_matches_when_a_sibling_is_filtered_out(self):
        rows = make(
            term("list comprehension", "biểu thức danh sách", context="tutorial"),
            term("list", "danh sách"),
        )
        found = Matcher(rows)("a list comprehension", where="library/x.po")
        assert found == {"list"}


class TestSelect:
    def test_rows_come_back_in_match_order(self):
        rows = Matcher(make(*NESTED)).select("a context manager changes the context")
        assert [row.en for row in rows] == ["context manager", "context"]

    def test_the_rows_carry_their_renderings_for_the_prompt(self):
        rows = Matcher(make(*NESTED)).select("a context manager")
        assert rows[0].vi == "trình quản lý ngữ cảnh"

    def test_nothing_matched_is_an_empty_tuple(self):
        assert Matcher(make(*NESTED)).select("nothing here") == ()


class TestMissing:
    def test_a_rendering_that_arrived_is_not_missing(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        assert matcher.missing("an iterable object", "một đối tượng khả lặp") == ()

    def test_a_rendering_that_never_arrived_is_named(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        found = matcher.missing("an iterable object", "một đối tượng lặp được")
        assert [row.en for row in found] == ["iterable"]

    def test_a_kept_row_passes_when_the_english_survived(self):
        matcher = Matcher(make(term("decorator", "decorator", keep_en=True)))
        assert matcher.missing("a decorator", "một decorator") == ()

    def test_a_kept_row_fails_when_the_english_was_translated_away(self):
        matcher = Matcher(make(term("decorator", "decorator", keep_en=True)))
        assert len(matcher.missing("a decorator", "một trình trang trí")) == 1

    def test_a_term_the_english_never_used_is_never_missing(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        assert matcher.missing("a plain sentence", "một câu") == ()

    def test_normal_form_does_not_decide_whether_a_rendering_arrived(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        decomposed = "một đối tượng khả lặp"
        assert matcher.missing("an iterable", decomposed) == ()

    def test_a_rendering_hidden_inside_markup_does_not_count_as_arrived(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        assert len(matcher.missing("an iterable", "xem ``khả lặp``")) == 1

    def test_case_does_not_decide_whether_a_rendering_arrived(self):
        matcher = Matcher(make(term("iterable", "khả lặp")))
        assert matcher.missing("an iterable", "Khả lặp là một giao thức") == ()


class TestThreeCallers:
    """The matcher is the one passed to ``stale``, not a second implementation."""

    def test_stale_by_glossary_takes_this_matcher_unchanged(self):
        memory = Memory(
            [
                Segment(
                    id=segment_id("an iterable object"),
                    msgid="an iterable object",
                    msgstr="một đối tượng khả lặp",
                    source="machine",
                ),
                Segment(
                    id=segment_id("a plain sentence"),
                    msgid="a plain sentence",
                    msgstr="một câu",
                    source="machine",
                ),
            ]
        )
        found = stale.by_glossary(memory, ["iterable"], make(term("iterable", "khả lặp")).matcher())
        assert len(found) == 1

    def test_the_glossary_hands_out_its_own_matcher(self):
        assert isinstance(make(*NESTED).matcher(), Matcher)


class TestDiff:
    def test_an_added_row_is_reported(self):
        before = make(term("list", "danh sách"))
        after = make(term("list", "danh sách"), term("iterable", "khả lặp"), version=2)
        assert [row.en for row in glossary.diff(before, after).added] == ["iterable"]

    def test_a_removed_row_is_reported(self):
        before = make(term("list", "danh sách"), term("iterable", "khả lặp"))
        after = make(term("list", "danh sách"), version=2)
        assert [row.en for row in glossary.diff(before, after).removed] == ["iterable"]

    def test_a_changed_rendering_carries_both_sides(self):
        before = make(term("iterable", "lặp được"))
        after = make(term("iterable", "khả lặp"), version=2)
        ((was, now),) = glossary.diff(before, after).changed
        assert (was.vi, now.vi) == ("lặp được", "khả lặp")

    def test_flipping_keep_en_is_a_change(self):
        before = make(term("decorator", "trình trang trí"))
        after = make(term("decorator", "decorator", keep_en=True), version=2)
        assert len(glossary.diff(before, after).changed) == 1

    def test_flipping_identifier_is_a_change_too(self):
        """It moves entries in and out of ``L02`` and changes what the prompt
        says, so a run made before it is not a run made after it."""
        before = make(term("float", "số thực"))
        after = make(term("float", "số thực", identifier=True), version=2)
        assert len(glossary.diff(before, after).changed) == 1

    def test_adding_a_context_is_a_change(self):
        before = make(term("list", "danh sách"))
        after = make(term("list", "danh sách", context="stdtypes"), version=2)
        assert len(glossary.diff(before, after).changed) == 1

    def test_editing_a_note_is_not_a_change(self):
        before = make(term("list", "danh sách"))
        after = make(term("list", "danh sách", note="a longer explanation"), version=2)
        assert not glossary.diff(before, after)

    def test_an_unchanged_glossary_is_falsey_and_empty(self):
        found = glossary.diff(make(*NESTED), make(*NESTED))
        assert not found and len(found) == 0 and found.terms == frozenset()

    def test_the_affected_terms_include_additions(self):
        before = make(term("list", "danh sách"))
        after = make(term("list", "danh sách"), term("iterable", "khả lặp"), version=2)
        assert glossary.diff(before, after).terms == {"iterable"}

    def test_the_affected_terms_include_removals_and_changes(self):
        before = make(term("list", "danh sách"), term("iterable", "lặp được"))
        after = make(term("iterable", "khả lặp"), version=2)
        assert glossary.diff(before, after).terms == {"list", "iterable"}

    def test_the_versions_are_carried_so_a_report_can_name_them(self):
        found = glossary.diff(make(version=3), make(version=4))
        assert (found.old, found.new) == (3, 4)

    def test_len_counts_every_kind_of_change(self):
        before = make(term("a", "một"), term("b", "hai"))
        after = make(term("a", "khác"), term("c", "ba"), version=2)
        assert len(glossary.diff(before, after)) == 3


class TestCollisions:
    def test_two_terms_sharing_a_rendering_are_rejected(self):
        rows = make(term("mistake", "lỗi"), term("bug", "lỗi"))
        assert {problem.rule for problem in glossary.check(rows)} == {"G-e"}

    def test_both_sides_of_a_collision_are_named(self):
        rows = make(term("mistake", "lỗi"), term("bug", "lỗi"))
        assert {problem.en for problem in glossary.check(rows)} == {"mistake", "bug"}

    def test_a_collision_is_allowed_when_both_rows_carry_a_context(self):
        rows = make(
            term("mistake", "lỗi", context="tutorial"),
            term("bug", "lỗi", context="bugs.po"),
        )
        assert glossary.check(rows) == []

    def test_a_collision_where_only_one_row_is_contextual_is_still_rejected(self):
        rows = make(term("mistake", "lỗi", context="tutorial"), term("bug", "lỗi"))
        assert len(glossary.check(rows)) == 2

    def test_case_does_not_hide_a_collision(self):
        rows = make(term("mistake", "Lỗi"), term("bug", "lỗi"))
        assert len(glossary.check(rows)) == 2

    def test_two_kept_rows_do_not_collide_because_their_renderings_differ(self):
        rows = make(
            term("decorator", "decorator", keep_en=True),
            term("generator", "generator", keep_en=True),
        )
        assert glossary.check(rows) == []


class TestASingularAndItsPlural:
    """The pair the rule cannot forbid without forbidding the whole glossary.

    The matcher is whole word, so "file object" does not match inside "file
    objects" and both forms need a row of their own for both to be enforced.
    Vietnamese renders them the same. A rule that called that a collision would
    be a rule with no way to satisfy it.
    """

    def test_a_plural_may_share_the_rendering_of_its_singular(self):
        rows = make(term("file objects", "đối tượng tệp"), term("file object", "đối tượng tệp"))
        assert glossary.check(rows) == []

    def test_an_es_plural_may_too(self):
        rows = make(term("type aliases", "bí danh kiểu"), term("type alias", "bí danh kiểu"))
        assert glossary.check(rows) == []

    def test_a_y_to_ies_plural_may_too(self):
        rows = make(
            term("shared libraries", "thư viện dùng chung"),
            term("shared library", "thư viện dùng chung"),
        )
        assert glossary.check(rows) == []

    def test_the_ending_that_differs_need_not_be_the_last_word(self):
        rows = make(
            term("backwards compatibility", "tương thích ngược"),
            term("backward compatibility", "tương thích ngược"),
        )
        assert glossary.check(rows) == []

    def test_a_hyphen_is_not_a_difference(self):
        rows = make(
            term("multi-phase initialization", "khởi tạo nhiều giai đoạn"),
            term("multiphase initialization", "khởi tạo nhiều giai đoạn"),
        )
        assert glossary.check(rows) == []

    def test_two_terms_that_only_look_alike_still_collide(self):
        """ "params" is not a plural of "parameter", it is a different word."""
        rows = make(term("type parameters", "tham số kiểu"), term("type params", "tham số kiểu"))
        assert len(glossary.check(rows)) == 2

    def test_a_real_collision_beside_a_plural_is_still_caught(self):
        rows = make(
            term("file objects", "đối tượng tệp"),
            term("file object", "đối tượng tệp"),
            term("handle", "đối tượng tệp"),
        )
        assert {problem.en for problem in glossary.check(rows)} == {
            "file objects",
            "file object",
            "handle",
        }

    def test_the_plural_is_not_named_as_the_thing_its_singular_collides_with(self):
        """The detail names what a reader has to decide between, and the plural
        of the term itself is not one of those things.
        """
        rows = make(
            term("file objects", "đối tượng tệp"),
            term("file object", "đối tượng tệp"),
            term("handle", "đối tượng tệp"),
        )
        found = {problem.en: problem.detail for problem in glossary.check(rows)}
        assert "'file object'" not in found["file objects"]
        assert "'handle'" in found["file objects"]


class TestShadowing:
    def test_a_shorter_term_listed_above_a_longer_one_is_rejected(self):
        rows = make(term("context", "ngữ cảnh"), term("context manager", "trình quản lý"))
        assert [problem.rule for problem in glossary.check(rows)] == ["G-f"]

    def test_the_same_rows_in_match_order_pass(self):
        assert glossary.check(make(*NESTED)) == []

    def test_containment_is_on_whole_words(self):
        rows = make(term("list", "danh sách"), term("listener", "trình nghe"))
        assert glossary.check(rows) == []

    def test_a_clean_glossary_has_nothing_to_report(self):
        assert glossary.check(make(term("iterable", "khả lặp"))) == []

    def test_a_rejection_reads_as_a_sentence(self):
        rows = make(term("context", "ngữ cảnh"), term("context manager", "trình quản lý"))
        assert "'context'" in str(glossary.check(rows)[0])


class TestReading:
    def test_a_minimal_file_loads(self):
        loaded = glossary.loads('version: 7\nterms:\n  - en: "list"\n    vi: "danh sách"\n')
        assert loaded.version == 7 and loaded.get("list").vi == "danh sách"

    def test_every_field_survives(self):
        text = (
            "version: 1\nterms:\n"
            '  - en: "decorator"\n    vi: "decorator"\n    keep_en: true\n'
            "    identifier: true\n"
            '    context: "library"\n    note: "the community keeps the English"\n'
        )
        row = glossary.loads(text).terms[0]
        assert (row.keep_en, row.identifier, row.context, row.note) == (
            True,
            True,
            "library",
            "the community keeps the English",
        )

    def test_a_file_with_no_terms_is_a_glossary_with_no_rows(self):
        assert len(glossary.loads("version: 1\nterms:\n")) == 0

    def test_something_that_is_not_yaml_is_refused(self):
        with pytest.raises(GlossaryError, match="not readable as YAML"):
            glossary.loads("version: [unclosed\n")

    def test_a_list_at_the_top_level_is_refused(self):
        with pytest.raises(GlossaryError, match="mapping"):
            glossary.loads("- one\n- two\n")

    def test_a_missing_version_is_refused(self):
        with pytest.raises(GlossaryError, match="'version'"):
            glossary.loads('terms:\n  - en: "list"\n    vi: "danh sách"\n')

    def test_a_version_that_is_not_an_integer_is_refused(self):
        with pytest.raises(GlossaryError, match="'version'"):
            glossary.loads('version: "seven"\nterms: []\n')

    def test_terms_that_are_not_a_list_are_refused(self):
        with pytest.raises(GlossaryError, match="'terms'"):
            glossary.loads("version: 1\nterms: nope\n")

    def test_a_row_that_is_not_a_mapping_is_refused_by_number(self):
        with pytest.raises(GlossaryError, match="term 1"):
            glossary.loads("version: 1\nterms:\n  - nope\n")

    def test_an_unknown_field_is_refused_rather_than_ignored(self):
        text = 'version: 1\nterms:\n  - en: "list"\n    vi: "danh sách"\n    vn: "oops"\n'
        with pytest.raises(GlossaryError, match="unknown field"):
            glossary.loads(text)

    def test_a_row_with_no_rendering_is_refused_by_number(self):
        with pytest.raises(GlossaryError, match="term 1"):
            glossary.loads('version: 1\nterms:\n  - en: "list"\n')

    def test_a_row_with_an_empty_rendering_is_refused_by_number(self):
        with pytest.raises(GlossaryError, match="term 1"):
            glossary.loads('version: 1\nterms:\n  - en: "list"\n    vi: ""\n')


class TestWriting:
    def test_a_row_writes_the_two_fields_it_always_has(self):
        written = glossary.dumps(make(term("list", "danh sách")))
        assert '  - en: "list"\n    vi: "danh sách"\n' in written

    def test_an_absent_field_stays_absent_rather_than_becoming_null(self):
        assert "context" not in glossary.dumps(make(term("list", "danh sách")))

    def test_keep_en_is_written_only_when_it_is_true(self):
        assert "keep_en" not in glossary.dumps(make(term("list", "danh sách")))
        assert "keep_en: true" in glossary.dumps(make(term("decorator", "decorator", keep_en=True)))

    def test_identifier_is_written_only_when_it_is_true(self):
        assert "identifier" not in glossary.dumps(make(term("list", "danh sách")))
        assert "identifier: true" in glossary.dumps(make(term("float", "số thực", identifier=True)))

    def test_the_version_is_written(self):
        assert "version: 7\n" in glossary.dumps(make(version=7))

    def test_a_rendering_that_reads_as_a_boolean_is_quoted(self):
        assert '"no"' in glossary.dumps(make(term("no", "no", keep_en=True)))

    def test_a_quote_inside_a_value_is_escaped(self):
        written = glossary.dumps(make(term("say", 'nói "xin chào"')))
        assert '\\"xin ch' in written

    def test_a_backslash_inside_a_value_is_escaped(self):
        assert '"\\\\n"' in glossary.dumps(make(term("newline", "\\n", keep_en=True)))


class TestRoundTrip:
    """The one test that stops the hand-rolled writer drifting from the reader."""

    ROWS = (
        term("context manager", "trình quản lý ngữ cảnh"),
        term("decorator", "decorator", keep_en=True, note='the community keeps "decorator"'),
        term("list", "danh sách", context="library/stdtypes", note="a\\b"),
        term("float", "số thực", identifier=True),
        term("no", "no", keep_en=True),
        term("3.15", "3.15", keep_en=True),
    )

    def test_everything_written_reads_back_identically(self):
        before = make(*self.ROWS, version=7)
        assert glossary.loads(glossary.dumps(before)) == before

    def test_writing_what_was_read_produces_the_same_bytes(self):
        text = glossary.dumps(make(*self.ROWS, version=7))
        assert glossary.dumps(glossary.loads(text)) == text

    def test_a_version_that_reads_as_a_float_survives(self):
        assert glossary.loads(glossary.dumps(make(version=315))).version == 315


class TestFiles:
    def test_a_saved_glossary_loads_back(self, tmp_path):
        path = glossary.save(make(*NESTED, version=4), tmp_path / "m" / "glossary.yaml")
        assert glossary.load(path) == make(*NESTED, version=4)

    def test_saving_creates_the_directory(self, tmp_path):
        glossary.save(make(), tmp_path / "deep" / "deeper" / "glossary.yaml")
        assert (tmp_path / "deep" / "deeper" / "glossary.yaml").exists()

    def test_a_missing_file_names_the_path(self, tmp_path):
        absent = tmp_path / "nothing.yaml"
        with pytest.raises(GlossaryError, match="cannot read"):
            glossary.load(absent)


MARKDOWN = f"""# Glossary

Prose a machine cannot write.

## Terms

{glossary.TABLE_OPEN}
{glossary.TABLE_CLOSE}

## How to update this file

More prose.
"""


class TestTable:
    def test_the_table_has_a_row_per_term_plus_a_header_and_a_rule(self):
        rendered = glossary.table(make(*NESTED))
        assert rendered.count("\n| ") == 4

    def test_rows_are_rendered_in_match_order(self):
        rendered = glossary.table(make(*NESTED[::-1]))
        assert rendered.index("| context manager |") < rendered.index("| context |")

    def test_a_kept_row_says_so_rather_than_repeating_the_english_twice(self):
        rendered = glossary.table(make(term("decorator", "decorator", keep_en=True)))
        assert "| decorator | `decorator` (kept) |" in rendered

    def test_the_version_and_the_counts_are_stated(self):
        rendered = glossary.table(make(term("decorator", "decorator", keep_en=True), version=7))
        assert "Version 7. 1 terms, 1 of them kept in English." in rendered

    def test_a_standalone_row_says_so_where_a_reviewer_will_read_it(self):
        rendered = glossary.table(make(term("float", "số thực", identifier=True)))
        assert "| float | số thực | An entry that is only `float` names the thing" in rendered

    def test_a_kept_row_does_not_repeat_itself_in_the_note(self):
        """``standalone`` holds every kept row, and saying so on each of the 52
        of them would push the useful notes off the side of the table."""
        rendered = glossary.table(make(term("sys", "sys", keep_en=True, identifier=True)))
        assert "names the thing" not in rendered

    def test_a_context_becomes_a_note_a_reviewer_can_act_on(self):
        rendered = glossary.table(make(term("list", "danh sách", context="stdtypes")))
        assert "Only where the path or msgctxt contains `stdtypes`." in rendered

    def test_a_pipe_in_a_note_is_escaped_so_the_table_survives(self):
        rendered = glossary.table(make(term("or", "hoặc", note="the a | b form")))
        assert "a \\| b" in rendered


class TestRender:
    def test_the_table_lands_between_the_markers(self):
        rendered = glossary.render(MARKDOWN, make(*NESTED))
        assert "| context manager |" in rendered

    def test_the_prose_on_either_side_is_untouched(self):
        rendered = glossary.render(MARKDOWN, make(*NESTED))
        assert rendered.startswith("# Glossary\n\nProse a machine cannot write.")
        assert rendered.endswith("More prose.\n")

    def test_rendering_twice_changes_nothing_the_second_time(self):
        once = glossary.render(MARKDOWN, make(*NESTED))
        assert glossary.render(once, make(*NESTED)) == once

    def test_a_file_with_no_markers_is_refused(self):
        with pytest.raises(GlossaryError, match="must both be present"):
            glossary.render("# Glossary\n", make(*NESTED))

    def test_markers_in_the_wrong_order_are_refused(self):
        backwards = f"{glossary.TABLE_CLOSE}\n{glossary.TABLE_OPEN}\n"
        with pytest.raises(GlossaryError, match="in that order"):
            glossary.render(backwards, make(*NESTED))


class TestAgreement:
    def test_a_regenerated_file_agrees(self):
        rows = make(*NESTED)
        assert glossary.agrees(glossary.render(MARKDOWN, rows), rows) == []

    def test_a_file_generated_from_an_older_version_disagrees(self):
        stale_markdown = glossary.render(MARKDOWN, make(*NESTED))
        assert glossary.agrees(stale_markdown, make(*NESTED, version=2)) != []

    def test_a_hand_edit_to_the_table_disagrees(self):
        edited = glossary.render(MARKDOWN, make(*NESTED)).replace("ngữ cảnh", "bối cảnh")
        assert [problem.rule for problem in glossary.agrees(edited, make(*NESTED))] == ["G05"]

    def test_a_file_missing_its_markers_fails_g05_rather_than_raising(self):
        problems = glossary.agrees("# Glossary\n", make(*NESTED))
        assert [problem.rule for problem in problems] == ["G05"]

    def test_the_failure_says_how_to_fix_it(self):
        edited = glossary.render(MARKDOWN, make(*NESTED)).replace("ngữ cảnh", "bối cảnh")
        assert "glossary check --fix" in glossary.agrees(edited, make(*NESTED))[0].detail


class TestStats:
    def test_the_counts_add_up(self):
        rows = make(
            term("context manager", "trình quản lý ngữ cảnh"),
            term("decorator", "decorator", keep_en=True),
            term("list", "danh sách", context="stdtypes", note="the built-in type"),
            version=7,
        )
        found = glossary.stats(rows)
        assert (found.version, found.terms, found.kept, found.contextual, found.noted) == (
            7,
            3,
            1,
            1,
            1,
        )

    def test_the_longest_term_is_the_one_the_matcher_reaches_first(self):
        assert glossary.stats(make(*NESTED)).longest == "context manager"

    def test_terms_are_counted_by_how_many_words_they_have(self):
        assert glossary.stats(make(*NESTED)).by_words == {1: 1, 2: 1}

    def test_problems_are_counted_so_show_can_say_the_file_is_broken(self):
        rows = make(term("context", "ngữ cảnh"), term("context manager", "trình quản lý"))
        assert glossary.stats(rows).problems == 1

    def test_an_empty_glossary_has_no_longest_term(self):
        assert glossary.stats(make()).longest == ""
