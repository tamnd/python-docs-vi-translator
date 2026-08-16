import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import segment
from pydocvi.segment import RestorationError, protect, restore


class TestProtect:
    def test_a_role_becomes_one_marker(self) -> None:
        result = protect("Return a :class:`list` object.")
        assert result.text == "Return a ⟦1⟧ object."
        assert result.spans == (":class:`list`",)

    def test_an_inline_literal_becomes_one_marker(self) -> None:
        assert protect("Pass ``None`` here.").text == "Pass ⟦1⟧ here."

    def test_markers_are_numbered_in_order(self) -> None:
        result = protect("A :func:`f` and ``g`` and :mod:`h`.")
        assert result.text == "A ⟦1⟧ and ⟦2⟧ and ⟦3⟧."

    def test_a_repeated_span_gets_its_own_marker_each_time(self) -> None:
        """Reuse would be smaller and would also let a dropped second
        occurrence pass as a complete answer."""
        result = protect("Both :mod:`os` and :mod:`os` again.")
        assert result.text == "Both ⟦1⟧ and ⟦2⟧ again."
        assert result.spans == (":mod:`os`", ":mod:`os`")

    def test_a_role_wins_over_the_literal_inside_it(self) -> None:
        result = protect("See :ref:`the ``x`` thing`.")
        assert result.count == 1

    def test_prose_with_no_markup_is_untouched(self) -> None:
        assert protect("An ordinary sentence.").text == "An ordinary sentence."

    @pytest.mark.parametrize(
        "text",
        ["Use %s here.", "Use %(name)s here.", "Use {} here.", "Use {name} here.", "A |sub| here."],
    )
    def test_format_specifiers_and_substitutions_are_protected(self, text: str) -> None:
        assert protect(text).count == 1

    def test_a_bare_url_is_protected(self) -> None:
        assert protect("See https://example.com/x for more.").count == 1


class TestLinks:
    def test_the_text_stays_visible_and_the_target_does_not(self) -> None:
        result = protect("Read the `tutorial <https://docs.python.org/>`_ first.")
        assert result.text == "Read the ⟦1⟧tutorial⟦2⟧ first."
        assert result.spans == ("`", " <https://docs.python.org/>`_")

    def test_a_link_with_no_target_is_protected_the_same_way(self) -> None:
        assert protect("See `the glossary`_ for terms.").text == "See ⟦1⟧the glossary⟦2⟧ for terms."

    def test_the_url_inside_a_link_is_not_protected_twice(self) -> None:
        """Protected separately, the URL leaves a marker buried inside the link
        tail and restoration refuses a string nothing was wrong with."""
        result = protect("The `Docutils <https://docutils.sourceforge.io/>`_ project.")
        assert restore(result.text, result.spans) == (
            "The `Docutils <https://docutils.sourceforge.io/>`_ project."
        )

    def test_a_closing_role_backtick_does_not_open_a_link(self) -> None:
        """200 entries in the corpus have two roles with an underscore-led word
        between them, and a link pattern applied first swallows the prose."""
        source = "Calls :func:`__import__`, as the standard :func:`__import__` does."
        result = protect(source)
        assert result.text == "Calls ⟦1⟧, as the standard ⟦2⟧ does."
        assert restore(result.text, result.spans) == source


class TestRestore:
    def test_puts_the_spans_back(self) -> None:
        source = "Return a :class:`list` whose items are ``sorted``."
        result = protect(source)
        assert restore(result.text, result.spans) == source

    def test_reordered_markers_are_legal(self) -> None:
        """Vietnamese does not put modifiers where English does, so a
        translation that moves a marker is usually right."""
        result = protect("A :func:`f` then ``g``.")
        assert restore("⟦2⟧ rồi ⟦1⟧.", result.spans) == "``g`` rồi :func:`f`."

    def test_a_missing_marker_is_refused(self) -> None:
        result = protect("A :func:`f` and ``g``.")
        with pytest.raises(RestorationError, match="missing"):
            restore("Chỉ còn ⟦1⟧.", result.spans)

    def test_a_repeated_marker_is_refused(self) -> None:
        result = protect("A :func:`f`.")
        with pytest.raises(RestorationError, match="repeated"):
            restore("⟦1⟧ và ⟦1⟧.", result.spans)

    def test_an_invented_marker_is_refused(self) -> None:
        result = protect("A :func:`f`.")
        with pytest.raises(RestorationError, match="never had"):
            restore("⟦1⟧ và ⟦9⟧.", result.spans)

    def test_a_spaced_marker_is_refused_rather_than_repaired(self) -> None:
        """``⟦ 1 ⟧`` means the model retyped the marker instead of copying it,
        and guessing which span it meant is how a wrong cross-reference gets
        into a catalog with nobody having decided to put it there."""
        result = protect("A :func:`f`.")
        with pytest.raises(RestorationError, match="malformed"):
            restore("⟦1⟧ và ⟦ 2 ⟧.", result.spans)

    def test_a_span_the_model_typed_out_as_well_is_refused(self) -> None:
        """Every marker is present exactly once, so ``P01`` is content. The span
        still ends up in the result twice, because the model copied the marker
        and then helpfully wrote the role out beside it."""
        source = "The :mod:`os` module."
        result = protect(source)
        with pytest.raises(RestorationError, match="appears 2 time"):
            restore("Mô-đun ⟦1⟧ tức :mod:`os`.", result.spans, original=source)


class TestProperties:
    @given(st.text())
    @settings(max_examples=500)
    def test_protection_round_trips_for_arbitrary_text(self, value: str) -> None:
        result = protect(value)
        assert restore(result.text, result.spans) == value

    @given(st.text())
    @settings(max_examples=200)
    def test_the_protected_form_has_one_marker_per_span(self, value: str) -> None:
        result = protect(value)
        found = [int(n) for n in segment.PLACEHOLDER.findall(result.text)]
        assert sorted(found) == list(range(1, result.count + 1))


class TestHelpers:
    def test_spans_of_lists_the_markup(self) -> None:
        assert segment.spans_of("A :func:`f` and ``g``.") == (":func:`f`", "``g``")

    def test_strip_markup_leaves_the_prose(self) -> None:
        assert segment.strip_markup("the :mod:`os` module").split() == ["the", "module"]

    def test_placeholders_lists_the_markers(self) -> None:
        assert protect("A :func:`f` and ``g``.").placeholders() == ("⟦1⟧", "⟦2⟧")
