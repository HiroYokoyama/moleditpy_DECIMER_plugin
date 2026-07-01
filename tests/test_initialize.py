"""
tests/test_initialize.py
Unit tests for decimer_plugin/__init__.py — metadata constants and initialize().
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import mock_optional_imports, make_context

ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / "decimer_plugin" / "__init__.py"


_mock_dialog_module = MagicMock()


def _load_init():
    """Load decimer_plugin/__init__.py as package 'decimer_plugin' with deps mocked."""
    with mock_optional_imports():
        spec = importlib.util.spec_from_file_location(
            "decimer_plugin",
            INIT_PATH,
            submodule_search_locations=[str(ROOT / "decimer_plugin")],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["decimer_plugin"] = mod
        sys.modules["decimer_plugin.dialog"] = _mock_dialog_module
        spec.loader.exec_module(mod)

    # Keep registered after context exits so callbacks can do relative imports
    sys.modules["decimer_plugin"] = mod
    sys.modules["decimer_plugin.dialog"] = _mock_dialog_module
    return mod


class TestPluginMetadata:
    def setup_method(self):
        self.mod = _load_init()

    def test_plugin_name(self):
        assert self.mod.PLUGIN_NAME == "DECIMER Image Importer"

    def test_plugin_version_is_nonempty_string(self):
        v = self.mod.PLUGIN_VERSION
        assert isinstance(v, str) and len(v) > 0

    def test_required_constants_present(self):
        for attr in (
            "PLUGIN_NAME",
            "PLUGIN_VERSION",
            "PLUGIN_AUTHOR",
            "PLUGIN_DESCRIPTION",
            "PLUGIN_SUPPORTED_MOLEDITPY_VERSION",
            "PLUGIN_DEPENDENCIES",
        ):
            assert hasattr(self.mod, attr), f"Missing constant: {attr}"

    def test_decimer_in_dependencies(self):
        deps = self.mod.PLUGIN_DEPENDENCIES
        assert isinstance(deps, list)
        assert any("DECIMER" in d for d in deps)

    def test_supported_version_targets_v4(self):
        ver = self.mod.PLUGIN_SUPPORTED_MOLEDITPY_VERSION
        assert "4" in ver


class TestInitializeRegistrations:
    def setup_method(self):
        self.mod = _load_init()

    def test_adds_menu_action(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        ctx.add_menu_action.assert_called_once()

    def test_menu_path_contains_import_keyword(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        path_arg = ctx.add_menu_action.call_args[0][0]
        assert "Import" in path_arg or "import" in path_arg

    def test_registers_drop_handler(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        ctx.register_drop_handler.assert_called_once()

    def test_drop_handler_priority_positive(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        call = ctx.register_drop_handler.call_args
        # priority may be passed as keyword or second positional arg
        priority = call.kwargs.get("priority") if call.kwargs else None
        if priority is None and len(call.args) > 1:
            priority = call.args[1]
        assert priority is not None and priority > 0


class TestDropHandler:
    def setup_method(self):
        self.mod = _load_init()
        ctx = make_context()
        self.mod.initialize(ctx)
        self.handler = ctx.register_drop_handler.call_args[0][0]
        # Patch _run_from_drop so tests never call DECIMER
        self.mod._run_from_drop = MagicMock()

    def test_ignores_mol_file(self):
        assert self.handler("structure.mol") is False

    def test_ignores_xyz_file(self):
        assert self.handler("coords.xyz") is False

    def test_ignores_txt_file(self):
        assert self.handler("data.txt") is False

    def test_ignores_svg_file(self):
        assert self.handler("image.svg") is False

    def test_accepts_png(self):
        assert self.handler("drawing.png") is True

    def test_accepts_jpg(self):
        assert self.handler("photo.jpg") is True

    def test_accepts_jpeg(self):
        assert self.handler("scan.jpeg") is True

    def test_accepts_uppercase_png(self):
        assert self.handler("STRUCTURE.PNG") is True

    def test_accepts_uppercase_jpg(self):
        assert self.handler("IMAGE.JPG") is True

    def test_calls_run_from_drop_on_accepted_file(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        handler = ctx.register_drop_handler.call_args[0][0]
        self.mod._run_from_drop = MagicMock()
        handler("molecule.png")
        self.mod._run_from_drop.assert_called_once()

    def test_does_not_call_run_from_drop_on_rejected_file(self):
        ctx = make_context()
        self.mod.initialize(ctx)
        handler = ctx.register_drop_handler.call_args[0][0]
        self.mod._run_from_drop = MagicMock()
        handler("molecule.mol")
        self.mod._run_from_drop.assert_not_called()


class TestOpenImportDialog:
    def setup_method(self):
        self.mod = _load_init()

    def test_show_dialog_calls_get_window(self):
        ctx = make_context()
        ctx.get_window.return_value = None
        self.mod.initialize(ctx)
        # Extract the menu callback
        show_cb = ctx.add_menu_action.call_args[0][1]
        show_cb()
        ctx.get_window.assert_called_with("dialog")

    def test_show_dialog_registers_new_window(self):
        ctx = make_context()
        ctx.get_window.return_value = None
        self.mod.initialize(ctx)
        show_cb = ctx.add_menu_action.call_args[0][1]
        show_cb()
        ctx.register_window.assert_called()
        window_id = ctx.register_window.call_args[0][0]
        assert window_id == "dialog"

    def test_show_dialog_raises_existing_visible_window(self):
        ctx = make_context()
        existing = MagicMock()
        existing.isVisible.return_value = True
        ctx.get_window.return_value = existing
        self.mod.initialize(ctx)
        show_cb = ctx.add_menu_action.call_args[0][1]
        show_cb()
        existing.raise_.assert_called_once()
        existing.activateWindow.assert_called_once()
        ctx.register_window.assert_not_called()
