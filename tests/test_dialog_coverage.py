"""
tests/test_dialog_coverage.py
Additional coverage for decimer_plugin/dialog.py: ImportFromImageDialog UI wiring
(_build_ui, _browse, _start_prediction, _on_done, _on_error, _close_progress,
_load_smiles), _import_error_message branches, _DECIMERWorker.__init__, and the
RuntimeError-on-progress-close branches inside run_decimer_async.

Uses its own richer (stateful) PyQt6 stubs so the dialog's real logic (text
storage, enabled flags, signal wiring) can be exercised, unlike the minimal
no-op stubs in test_dialog.py.
"""

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIALOG_PATH = ROOT / "decimer_plugin" / "dialog.py"


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, fn):
        self._slots.append(fn)

    def emit(self, *a, **kw):
        for fn in list(self._slots):
            fn(*a, **kw)


class _QLineEdit:
    def __init__(self, *a, **kw):
        self._text = ""
        self._placeholder = ""
        self._readonly = False

    def setPlaceholderText(self, t):
        self._placeholder = t

    def setReadOnly(self, v):
        self._readonly = v

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text

    def clear(self):
        self._text = ""


class _QPushButton:
    def __init__(self, *a, **kw):
        self._enabled = True
        self.clicked = _Signal()

    def setEnabled(self, v):
        self._enabled = v

    def isEnabled(self):
        return self._enabled

    def setFixedWidth(self, w):
        pass


class _QLabel:
    def __init__(self, *a, **kw):
        pass


class _QLayout:
    def __init__(self, *a, **kw):
        pass

    def addWidget(self, w):
        pass

    def addLayout(self, l):
        pass

    def addStretch(self):
        pass


class _QDialog:
    def __init__(self, parent=None):
        self._parent = parent
        self.closed = False

    def setWindowTitle(self, t):
        pass

    def resize(self, w, h):
        pass

    def close(self):
        self.closed = True

    def show(self):
        pass


class _QProgressDialog:
    def __init__(self, *a, **kw):
        self.closed = False

    def setWindowModality(self, m):
        pass

    def setMinimumDuration(self, d):
        pass

    def show(self):
        pass

    def close(self):
        self.closed = True


class _QFileDialog:
    @staticmethod
    def getOpenFileName(*a, **kw):
        return ("", "")


class _QMessageBox:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def warning(cls, *a, **kw):
        cls.calls.append(("warning", a, kw))

    @classmethod
    def critical(cls, *a, **kw):
        cls.calls.append(("critical", a, kw))


class _QThread:
    def __init__(self, parent=None):
        pass

    def start(self):
        pass


def _install_qt_stubs():
    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.pyqtSignal = lambda *a, **kw: _Signal()
    qt_core.Qt = MagicMock()
    qt_core.QThread = _QThread

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    qt_widgets.QVBoxLayout = _QLayout
    qt_widgets.QHBoxLayout = _QLayout
    qt_widgets.QPushButton = _QPushButton
    qt_widgets.QLabel = _QLabel
    qt_widgets.QLineEdit = _QLineEdit
    qt_widgets.QFileDialog = _QFileDialog
    qt_widgets.QMessageBox = _QMessageBox
    qt_widgets.QProgressDialog = _QProgressDialog

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtWidgets = qt_widgets

    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_qt_stubs()


def _load_dialog():
    mod_name = f"_decimer_dialog_cov_{abs(hash(DIALOG_PATH))}_{id(object())}"
    spec = importlib.util.spec_from_file_location(mod_name, DIALOG_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_dialog_mod = _load_dialog()


def _make_ctx():
    ctx = MagicMock()
    ctx.get_main_window.return_value = MagicMock()
    return ctx


def _make_sync_worker_class(smiles_result=None, error_msg=None):
    """A _DECIMERWorker stand-in that fires finished/failed synchronously on start()."""

    class _SyncWorker:
        def __init__(self, image_path, parent=None):
            self._image_path = image_path
            self.finished = _Signal()
            self.failed = _Signal()

        def start(self):
            if error_msg is not None:
                self.failed.emit(error_msg)
            else:
                self.finished.emit(smiles_result or "")

    return _SyncWorker


@pytest.fixture(autouse=True)
def _reset_message_box():
    _QMessageBox.reset()
    yield
    _QMessageBox.reset()


class TestImportErrorMessage:
    def test_decimer_missing_message(self):
        msg = _dialog_mod._import_error_message(ImportError("No module named 'DECIMER'"))
        assert "not installed" in msg
        assert "pip install DECIMER" in msg

    def test_empty_message_treated_as_missing(self):
        msg = _dialog_mod._import_error_message(ImportError(""))
        assert "not installed" in msg

    def test_dll_failure_message(self):
        msg = _dialog_mod._import_error_message(ImportError("DLL load failed while importing _pywrap_tensorflow"))
        assert "DLL" in msg
        assert "Visual C++" in msg

    def test_generic_dependency_message(self):
        msg = _dialog_mod._import_error_message(ImportError("No module named 'somepkg'"))
        assert "required dependency could not be imported" in msg
        assert "somepkg" in msg


class TestDECIMERWorkerInit:
    def test_init_stores_image_path(self):
        worker = _dialog_mod._DECIMERWorker("img.png")
        assert worker._image_path == "img.png"


class TestDialogBuildUI:
    def test_construct_dialog_builds_widgets(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        assert dlg._context is ctx
        assert dlg._worker is None
        assert dlg._progress is None
        assert dlg._path_edit.text() == ""
        assert dlg._btn_predict.isEnabled() is False
        assert dlg._btn_load.isEnabled() is False


class TestBrowse:
    def test_browse_sets_path_and_enables_predict(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        with patch.object(_dialog_mod.QFileDialog, "getOpenFileName", return_value=("mol.png", "")):
            dlg._browse()
        assert dlg._path_edit.text() == "mol.png"
        assert dlg._btn_predict.isEnabled() is True
        assert dlg._btn_load.isEnabled() is False

    def test_browse_cancel_leaves_state_unchanged(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        with patch.object(_dialog_mod.QFileDialog, "getOpenFileName", return_value=("", "")):
            dlg._browse()
        assert dlg._path_edit.text() == ""
        assert dlg._btn_predict.isEnabled() is False

    def test_browse_clears_previous_smiles(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._smiles_edit.setText("CCO")
        dlg._btn_load.setEnabled(True)
        with patch.object(_dialog_mod.QFileDialog, "getOpenFileName", return_value=("mol.png", "")):
            dlg._browse()
        assert dlg._smiles_edit.text() == ""
        assert dlg._btn_load.isEnabled() is False


class TestStartPrediction:
    def test_no_path_does_nothing(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._start_prediction()
        assert dlg._worker is None

    def test_success_populates_smiles_and_enables_load(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._path_edit.setText("mol.png")
        worker_cls = _make_sync_worker_class(smiles_result="c1ccccc1")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            dlg._start_prediction()
        assert dlg._smiles_edit.text() == "c1ccccc1"
        assert dlg._btn_load.isEnabled() is True
        assert dlg._btn_predict.isEnabled() is True
        assert dlg._progress is None

    def test_empty_result_shows_warning(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._path_edit.setText("blank.png")
        worker_cls = _make_sync_worker_class(smiles_result="")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            dlg._start_prediction()
        assert dlg._btn_load.isEnabled() is False
        assert any(c[0] == "warning" for c in _QMessageBox.calls)

    def test_error_shows_critical_message(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._path_edit.setText("bad.png")
        worker_cls = _make_sync_worker_class(error_msg="DECIMER is not installed.")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            dlg._start_prediction()
        assert dlg._btn_predict.isEnabled() is True
        assert any(c[0] == "critical" for c in _QMessageBox.calls)


class TestCloseProgress:
    def test_close_progress_none_is_noop(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._close_progress()
        assert dlg._progress is None

    def test_close_progress_normal(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._progress = _QProgressDialog()
        dlg._close_progress()
        assert dlg._progress is None

    def test_close_progress_swallows_runtime_error(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        fake = MagicMock()
        fake.close.side_effect = RuntimeError("already destroyed")
        dlg._progress = fake
        dlg._close_progress()
        assert dlg._progress is None


class TestLoadSmiles:
    def test_no_smiles_does_nothing(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._load_smiles()
        ctx.load_from_smiles.assert_not_called()

    def test_no_context_does_nothing(self):
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=None)
        dlg._smiles_edit.setText("CCO")
        dlg._load_smiles()  # must not raise

    def test_success_loads_and_closes(self):
        ctx = _make_ctx()
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._smiles_edit.setText("CCO")
        dlg._load_smiles()
        ctx.load_from_smiles.assert_called_once_with("CCO")
        ctx.show_status_message.assert_called_once()
        assert dlg.closed is True

    def test_exception_shows_critical(self):
        ctx = _make_ctx()
        ctx.load_from_smiles.side_effect = ValueError("bad smiles")
        dlg = _dialog_mod.ImportFromImageDialog(parent=None, context=ctx)
        dlg._smiles_edit.setText("!!!")
        dlg._load_smiles()
        assert any(c[0] == "critical" for c in _QMessageBox.calls)
        assert dlg.closed is False


class TestRunDecimerAsyncProgressCloseErrors:
    def test_on_done_swallows_progress_close_runtime_error(self):
        ctx = _make_ctx()
        fake_progress = MagicMock()
        fake_progress.close.side_effect = RuntimeError("gone")
        worker_cls = _make_sync_worker_class(smiles_result="CCO")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls), \
             patch.object(_dialog_mod, "QProgressDialog", return_value=fake_progress):
            _dialog_mod.run_decimer_async(ctx, "mol.png")
        ctx.load_from_smiles.assert_called_once_with("CCO")

    def test_on_error_swallows_progress_close_runtime_error(self):
        ctx = _make_ctx()
        fake_progress = MagicMock()
        fake_progress.close.side_effect = RuntimeError("gone")
        worker_cls = _make_sync_worker_class(error_msg="boom")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls), \
             patch.object(_dialog_mod, "QProgressDialog", return_value=fake_progress):
            _dialog_mod.run_decimer_async(ctx, "mol.png")
        ctx.load_from_smiles.assert_not_called()
