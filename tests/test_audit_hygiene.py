"""``H01`` to ``H06``: what is in the repository that should not be.

These are the only checks that read the working tree rather than parsed
catalogs, so every test here writes real files into ``tmp_path`` and hands the
corpus the list git would have handed it.

``H03`` gets the most tests, because it is the one with no second chance. A key
that reaches a public repository has to be rotated whatever anybody does
afterwards, and a check that is muted for being noisy is a check that was not
there on the day it mattered.
"""

from pathlib import Path

from conftest import corpus_of, findings
from pydocvi.audit import hygiene
from pydocvi.audit.model import Corpus

#: Assembled rather than written out, for the same reason ``conftest`` does it:
#: ``make secrets`` refuses to let any tracked file contain a key-shaped string,
#: and that rule is worth more without an allowlist in it.
SHAPED_LIKE_A_KEY = "sk-" + "0123456789abcdef"

README = '# Content\n\n<!-- counts: {"1": 1626} -->\n'


def tracked(where: Path, files: dict[str, str | bytes]) -> Corpus:
    """A corpus over files written into ``where``, named as git would name them.

    Written for real rather than faked, because four of these six checks stat or
    read the file rather than reasoning about its name.
    """
    paths = []
    for name, content in files.items():
        path = where / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        paths.append(path)
    return corpus_of(root=where, tracked=tuple(paths))


class TestH01:
    def test_a_compiled_catalog_is_found(self, tmp_path: Path) -> None:
        """A .mo drifts from its source the first time anybody edits the source
        and forgets to recompile."""
        corpus = tracked(tmp_path, {"library/one.mo": b"\xde\x12\x04\x95"})
        found = findings(hygiene.h01_no_mo_files, corpus)
        assert len(found) == 1
        assert found[0].path == "library/one.mo"

    def test_the_catalog_beside_it_is_clean(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"library/one.po": 'msgid ""\nmsgstr ""\n'})
        assert findings(hygiene.h01_no_mo_files, corpus) == []


class TestH02:
    def test_a_large_file_that_is_not_a_catalog_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"assets/diagram.png": b"x" * (hygiene.LARGE + 1)})
        found = findings(hygiene.h02_no_large_files, corpus)
        assert len(found) == 1
        assert "ceiling" in found[0].detail

    def test_a_catalog_of_the_same_size_is_ordinary(self, tmp_path: Path) -> None:
        """whatsnew/changelog.po is 5 126 KB and is a correct file, so the
        catalog ceiling has to clear it."""
        corpus = tracked(tmp_path, {"whatsnew/changelog.po": "x" * (hygiene.LARGE + 1)})
        assert findings(hygiene.h02_no_large_files, corpus) == []

    def test_a_catalog_over_the_catalog_ceiling_is_still_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"library/one.po": "x" * (hygiene.LARGE_CATALOG + 1)})
        assert len(findings(hygiene.h02_no_large_files, corpus)) == 1

    def test_a_file_git_lists_and_the_disk_does_not_have_is_skipped(self, tmp_path: Path) -> None:
        corpus = corpus_of(root=tmp_path, tracked=(tmp_path / "gone.po",))
        assert findings(hygiene.h02_no_large_files, corpus) == []

    def test_the_memory_is_allowed_to_be_big(self, tmp_path: Path) -> None:
        """It is every segment the project has and it only grows. This check
        fired on it the day it was committed, which is the right instinct and
        the wrong answer."""
        corpus = tracked(tmp_path, {hygiene.EXPECTED_LARGE: "x" * (hygiene.LARGE + 1)})
        assert findings(hygiene.h02_no_large_files, corpus) == []

    def test_nothing_else_under_manifests_gets_the_same_pass(self, tmp_path: Path) -> None:
        """One path, not a pattern, so the exemption cannot spread by being
        next to something that already has it."""
        corpus = tracked(tmp_path, {"manifests/glossary.yaml": "x" * (hygiene.LARGE + 1)})
        assert len(findings(hygiene.h02_no_large_files, corpus)) == 1


class TestH03:
    def test_a_key_shaped_string_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"manifests/routes.json": f'{{"key": "{SHAPED_LIKE_A_KEY}"}}'})
        found = findings(hygiene.h03_no_secrets, corpus)
        assert len(found) == 1
        assert found[0].line == 1

    def test_the_finding_never_quotes_what_it_matched(self, tmp_path: Path) -> None:
        """A check that printed the secret into a CI log to tell you the secret
        was in a file would have published it a second time, somewhere with a
        longer retention."""
        corpus = tracked(tmp_path, {"notes.md": SHAPED_LIKE_A_KEY})
        found = findings(hygiene.h03_no_secrets, corpus)[0]
        assert SHAPED_LIKE_A_KEY not in str(found)
        assert SHAPED_LIKE_A_KEY not in str(found.as_dict())

    def test_the_key_rule_applies_inside_a_catalog_too(self, tmp_path: Path) -> None:
        """The one rule with no exemption anywhere. A msgid comes from CPython
        and cannot carry our key, so a match in a catalog is a paste."""
        corpus = tracked(tmp_path, {"library/one.po": f'msgstr "{SHAPED_LIKE_A_KEY}"'})
        assert len(findings(hygiene.h03_no_secrets, corpus)) == 1

    def test_a_private_key_header_outside_a_catalog_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"notes.md": "-----BEGIN RSA PRIVATE KEY-----"})
        assert len(findings(hygiene.h03_no_secrets, corpus)) == 1

    def test_a_private_key_header_inside_a_catalog_is_the_ssl_page(self, tmp_path: Path) -> None:
        """library/ssl.po contains the literal PEM header as documentation text,
        and it matched on the first real run."""
        corpus = tracked(tmp_path, {"library/ssl.po": 'msgid "-----BEGIN PRIVATE KEY-----"'})
        assert findings(hygiene.h03_no_secrets, corpus) == []

    def test_a_host_and_port_outside_a_catalog_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"notes.md": "the box answers on 10.4.2.9:8103"})
        assert len(findings(hygiene.h03_no_secrets, corpus)) == 1

    def test_a_host_and_port_inside_a_catalog_is_a_curl_example(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"library/contextvars.po": 'msgid "curl 10.4.2.9:8080"'})
        assert findings(hygiene.h03_no_secrets, corpus) == []

    def test_loopback_tells_a_reader_nothing_and_is_not_reported(self, tmp_path: Path) -> None:
        """Loopback is every tutorial's example server, and matching it costs
        the check its credibility."""
        corpus = tracked(tmp_path, {"README.md": "runs on 127.0.0.1:8000"})
        assert findings(hygiene.h03_no_secrets, corpus) == []

    def test_the_documentation_address_blocks_are_not_reported(self, tmp_path: Path) -> None:
        """RFC 5737 set those three blocks aside so that documentation has
        addresses it can print."""
        text = "\n".join(("192.0.2.1:80", "198.51.100.7:443", "203.0.113.9:8080"))
        corpus = tracked(tmp_path, {"README.md": text})
        assert findings(hygiene.h03_no_secrets, corpus) == []

    def test_a_binary_file_is_skipped_rather_than_reported(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"assets/logo.png": b"\x89PNG\r\n\x1a\n\xff\xfe"})
        assert findings(hygiene.h03_no_secrets, corpus) == []


class TestH04:
    def test_a_file_that_does_not_end_in_a_newline_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"README.md": "# Content"})
        found = findings(hygiene.h04_text_is_well_formed, corpus)
        assert [one.detail for one in found] == ["does not end in a newline"]

    def test_a_file_that_ends_in_two_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"README.md": "# Content\n\n"})
        found = findings(hygiene.h04_text_is_well_formed, corpus)
        assert [one.detail for one in found] == ["ends in more than one newline"]

    def test_carriage_returns_are_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"README.md": b"# Content\r\n"})
        assert any(
            one.detail == "has carriage returns"
            for one in findings(hygiene.h04_text_is_well_formed, corpus)
        )

    def test_trailing_whitespace_is_found_with_its_line(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"README.md": "# Content\ntwo \nthree\n"})
        found = [one for one in findings(hygiene.h04_text_is_well_formed, corpus) if one.line]
        assert len(found) == 1
        assert found[0].line == 2

    def test_a_file_that_is_not_utf_8_is_found(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"README.md": b"caf\xe9\n"})
        found = findings(hygiene.h04_text_is_well_formed, corpus)
        assert len(found) == 1
        assert "is not UTF-8" in found[0].detail

    def test_a_long_line_is_not_reported(self, tmp_path: Path) -> None:
        """No wrapping rule reproduces the inherited corpus, 86.31 per cent
        being the best any of them manages, so a column bound would fail on
        thousands of correct lines. S08 covers the ground that matters."""
        corpus = tracked(tmp_path, {"library/one.po": f'msgid "{"x" * 400}"\n'})
        assert findings(hygiene.h04_text_is_well_formed, corpus) == []

    def test_a_binary_suffix_is_not_read_as_text(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"assets/logo.png": b"\x89PNG no trailing newline"})
        assert findings(hygiene.h04_text_is_well_formed, corpus) == []

    def test_an_empty_file_is_not_missing_a_newline(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"reports/quality.md": ""})
        assert findings(hygiene.h04_text_is_well_formed, corpus) == []


class TestH05:
    def test_a_tracked_locales_directory_is_found(self, tmp_path: Path) -> None:
        """It is where sphinx-intl puts a working copy of the catalogs, so a
        tracked one is a second, silently diverging set of the same 548 files."""
        corpus = tracked(tmp_path, {"locales/vi/one.po": 'msgid ""\n'})
        found = findings(hygiene.h05_no_working_copy, corpus)
        assert len(found) == 1
        assert "locales/" in found[0].detail

    def test_a_ds_store_is_found_by_name(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"MACHINE/.DS_Store": "x"})
        assert len(findings(hygiene.h05_no_working_copy, corpus)) == 1

    def test_each_file_is_reported_once(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"venv/lib/one.py": "x"})
        assert len(findings(hygiene.h05_no_working_copy, corpus)) == 1

    def test_an_ordinary_catalog_is_clean(self, tmp_path: Path) -> None:
        corpus = tracked(tmp_path, {"library/one.po": 'msgid ""\n'})
        assert findings(hygiene.h05_no_working_copy, corpus) == []


class TestH06:
    """The README is the only one of the two anybody reads without being asked
    to, which makes it the one most worth keeping true."""

    def test_a_readme_that_disagrees_with_the_report_is_found(self) -> None:
        report = '# Coverage\n\n<!-- counts: {"1": 1700} -->\n'
        found = findings(
            hygiene.h06_readme_matches_coverage, corpus_of(readme=README, coverage=report)
        )
        assert len(found) == 1
        assert "1626" in found[0].detail and "1700" in found[0].detail

    def test_a_readme_that_agrees_is_clean(self) -> None:
        report = '# Coverage\n\n<!-- counts: {"1": 1626} -->\n'
        corpus = corpus_of(readme=README, coverage=report)
        assert findings(hygiene.h06_readme_matches_coverage, corpus) == []

    def test_a_tier_only_one_of_them_knows_about_is_found(self) -> None:
        report = '# Coverage\n\n<!-- counts: {"1": 1626, "2": 40} -->\n'
        found = findings(
            hygiene.h06_readme_matches_coverage, corpus_of(readme=README, coverage=report)
        )
        assert len(found) == 1
        assert "the README says nothing" in found[0].detail

    def test_a_readme_with_no_counts_in_it_is_found(self) -> None:
        report = '# Coverage\n\n<!-- counts: {"1": 1626} -->\n'
        corpus = corpus_of(readme="# Content\n", coverage=report)
        assert len(findings(hygiene.h06_readme_matches_coverage, corpus)) == 1

    def test_a_run_with_neither_file_says_nothing(self) -> None:
        assert findings(hygiene.h06_readme_matches_coverage, corpus_of()) == []
