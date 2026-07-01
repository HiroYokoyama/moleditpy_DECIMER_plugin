"""
Shared test infrastructure for moleditpy_DECIMER_plugin.

Provides:
- ``mock_optional_imports()`` — context manager that intercepts heavy/optional
  dependencies (PyQt6, DECIMER, PIL, etc.) with MagicMock so tests run headlessly.
- ``load_plugin(path)`` — load a plugin .py file with deps already mocked.
- ``make_context()`` — build a stub PluginContext (MagicMock with non-None main window).
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]

BLOCKED_TOPS: frozenset[str] = frozenset(
    {
        "PyQt6",
        "DECIMER",
        "Pillow",
        "PIL",
        "numpy",
        "tensorflow",
        "moleditpy",
    }
)


class _MagicLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> MagicMock:
        m = MagicMock()
        m.__name__ = spec.name
        m.__spec__ = spec
        m.__path__ = []
        m.__package__ = spec.name.split(".")[0]
        return m  # type: ignore[return-value]

    def exec_module(self, module: object) -> None:
        pass


class _MagicFinder(importlib.abc.MetaPathFinder):
    _loader = _MagicLoader()

    def find_spec(
        self,
        fullname: str,
        path: object,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".")[0] in BLOCKED_TOPS:
            return importlib.machinery.ModuleSpec(fullname, self._loader)
        return None


@contextlib.contextmanager
def mock_optional_imports() -> Generator[None, None, None]:
    removed = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.split(".")[0] in BLOCKED_TOPS
    }
    finder = _MagicFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(removed)
        for k in list(sys.modules):
            if k.split(".")[0] in BLOCKED_TOPS and k not in removed:
                del sys.modules[k]


def load_plugin(path: Path) -> object:
    """Load *path* as an isolated module. Must be inside mock_optional_imports()."""
    mod_name = f"_smoke_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def make_context() -> MagicMock:
    """Return a stub PluginContext with a non-None main window."""
    ctx = MagicMock()
    ctx.get_main_window.return_value = MagicMock()
    return ctx
