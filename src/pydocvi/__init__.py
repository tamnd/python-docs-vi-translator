"""Translate the CPython documentation into Vietnamese gettext catalogs."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("python-docs-vi-translator")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
