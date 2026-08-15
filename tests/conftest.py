"""Shared fixtures.

Nothing here touches the network or the fleet. Fixtures that need a real corpus
are guarded by the ``corpus`` marker and skip when the checkout is absent.
"""

from pathlib import Path

import pytest

UPSTREAM = Path.home() / "github" / "tamnd" / "python-docs-vi"


@pytest.fixture(scope="session")
def upstream() -> Path:
    """A local checkout of the upstream Transifex mirror."""
    if not (UPSTREAM / "bugs.po").exists():
        pytest.skip(f"no upstream checkout at {UPSTREAM}")
    return UPSTREAM


@pytest.fixture
def data_dir() -> Path:
    """Hand-written fixtures. None of them is longer than forty lines."""
    return Path(__file__).parent / "data"
