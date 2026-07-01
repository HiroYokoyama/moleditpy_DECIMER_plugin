"""
tests/test_dialog.py
Unit tests for decimer_plugin/dialog.py — worker logic and run_decimer_async.
"""

import subprocess
import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIALOG_PATH = ROOT / "decimer_plugin" / "dialog.py"


def _make_widget_class(class_name: str):
    """Create a simple stub widget class that ignores all args/calls."""
    class _Widget:
        def __init__(self, *a, **kw):
            pass
        def __getattr__(self, n):
            return lambda *a, **kw: None
    _Widget.__name__ = class_name
    _Widget.__qualname__ = class_name
    return _Widget


def _install_qt_stubs():
    """Install minimal PyQt6 stubs so dialog.py can be imported."""
    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()
    qt_core.Qt = MagicMock()

    class _QThread:
        def __init__(self, parent=None):
            pass
        def start(self):
            pass

    qt_core.QThread = _QThread

    class _QMessageBox:
        def __init__(self, *a, **kw): pass
        @classmethod
        def warning(cls, *a, **kw): pass
        @classmethod
        def critical(cls, *a, **kw): pass
        @classmethod
        def information(cls, *a, **kw): pass
        @classmethod
        def question(cls, *a, **kw): pass

    class _QFileDialog:
        def __init__(self, *a, **kw): pass
        @staticmethod
        def getOpenFileName(*a, **kw): return ("", "")

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in [
        "QDialog",
        "QVBoxLayout",
        "QHBoxLayout",
        "QPushButton",
        "QLabel",
        "QLineEdit",
        "QProgressDialog",
    ]:
        setattr(qt_widgets, name, _make_widget_class(name))

    qt_widgets.QMessageBox = _QMessageBox
    qt_widgets.QFileDialog = _QFileDialog

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtWidgets = qt_widgets

    sys.modules.setdefault("PyQt6", pyqt6)
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_qt_stubs()


def _load_dialog():
    mod_name = f"_decimer_dialog_under_test_{abs(hash(DIALOG_PATH))}"
    spec = importlib.util.spec_from_file_location(mod_name, DIALOG_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_dialog_mod = _load_dialog()


class TestPredictSmilesSubprocess:
    """Test _predict_smiles_subprocess error detection."""

    def _mock_result(self, returncode=0, stdout="", stderr=""):
        r = MagicMock()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_returns_stripped_smiles_on_success(self):
        with patch("subprocess.run", return_value=self._mock_result(stdout="c1ccccc1\n")):
            assert _dialog_mod._predict_smiles_subprocess("test.png") == "c1ccccc1"

    def test_returns_empty_string_when_prediction_empty(self):
        with patch("subprocess.run", return_value=self._mock_result(stdout="\n")):
            assert _dialog_mod._predict_smiles_subprocess("blank.png") == ""

    def test_raises_import_error_when_decimer_missing(self):
        r = self._mock_result(returncode=1, stderr="ModuleNotFoundError: No module named 'DECIMER'\n")
        with patch("subprocess.run", return_value=r):
            with pytest.raises(ImportError, match="DECIMER"):
                _dialog_mod._predict_smiles_subprocess("test.png")

    def test_raises_import_error_on_dll_failure(self):
        r = self._mock_result(returncode=1, stderr="ImportError: DLL load failed while importing _pywrap_tensorflow_internal")
        with patch("subprocess.run", return_value=r):
            with pytest.raises(ImportError):
                _dialog_mod._predict_smiles_subprocess("test.png")

    def test_raises_runtime_error_on_other_failure(self):
        r = self._mock_result(returncode=1, stderr="ValueError: image format not supported")
        with patch("subprocess.run", return_value=r):
            with pytest.raises(RuntimeError, match="image format"):
                _dialog_mod._predict_smiles_subprocess("test.png")

    def test_raises_runtime_error_with_code_when_stderr_empty(self):
        r = self._mock_result(returncode=2, stderr="")
        with patch("subprocess.run", return_value=r):
            with pytest.raises(RuntimeError, match="code 2"):
                _dialog_mod._predict_smiles_subprocess("test.png")


class TestDECIMERWorkerRun:
    """Test _DECIMERWorker.run() without actually running a thread."""

    def _make_worker(self, image_path="test.png"):
        worker = _dialog_mod._DECIMERWorker.__new__(_dialog_mod._DECIMERWorker)
        worker._image_path = image_path
        worker.finished = MagicMock()
        worker.failed = MagicMock()
        return worker

    def test_emits_smiles_on_success(self):
        worker = self._make_worker("benzene.png")
        with patch.object(_dialog_mod, "_predict_smiles_subprocess", return_value="c1ccccc1"):
            worker.run()
        worker.finished.emit.assert_called_once_with("c1ccccc1")
        worker.failed.emit.assert_not_called()

    def test_emits_empty_string_when_predict_returns_empty(self):
        worker = self._make_worker("blank.png")
        with patch.object(_dialog_mod, "_predict_smiles_subprocess", return_value=""):
            worker.run()
        worker.finished.emit.assert_called_once_with("")
        worker.failed.emit.assert_not_called()

    def test_emits_failed_on_import_error(self):
        worker = self._make_worker("img.png")
        with patch.object(_dialog_mod, "_predict_smiles_subprocess",
                          side_effect=ImportError("No module named 'DECIMER'")):
            worker.run()
        worker.failed.emit.assert_called_once()
        msg = worker.failed.emit.call_args[0][0]
        assert "DECIMER" in msg
        assert "install" in msg.lower()

    def test_emits_failed_on_timeout(self):
        worker = self._make_worker("slow.png")
        with patch.object(_dialog_mod, "_predict_smiles_subprocess",
                          side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=600)):
            worker.run()
        worker.failed.emit.assert_called_once()
        msg = worker.failed.emit.call_args[0][0]
        assert "timed out" in msg.lower() or "timeout" in msg.lower()
        worker.finished.emit.assert_not_called()

    def test_emits_failed_on_general_exception(self):
        worker = self._make_worker("bad.png")
        with patch.object(_dialog_mod, "_predict_smiles_subprocess",
                          side_effect=RuntimeError("corrupt image")):
            worker.run()
        worker.failed.emit.assert_called_once()
        worker.finished.emit.assert_not_called()


def _make_sync_worker_class(smiles_result=None, error_msg=None):
    """Return a _DECIMERWorker replacement that fires signals synchronously on start()."""

    class _SyncWorker:
        _finished_slots: list = []
        _failed_slots: list = []

        class finished:
            @classmethod
            def connect(cls, fn):
                _SyncWorker._finished_slots.append(fn)

        class failed:
            @classmethod
            def connect(cls, fn):
                _SyncWorker._failed_slots.append(fn)

        def __init__(self, image_path, parent=None):
            _SyncWorker._finished_slots = []
            _SyncWorker._failed_slots = []

        def start(self):
            if error_msg is not None:
                for fn in _SyncWorker._failed_slots:
                    fn(error_msg)
            else:
                for fn in _SyncWorker._finished_slots:
                    fn(smiles_result or "")

    return _SyncWorker


class TestRunDecimerAsync:
    """Test run_decimer_async (the drop-handler path) using a synchronous worker stub."""

    def _make_ctx(self):
        ctx = MagicMock()
        ctx.get_main_window.return_value = MagicMock()
        return ctx

    def test_calls_load_from_smiles_on_success(self):
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(smiles_result="CCO")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "ethanol.png")
        ctx.load_from_smiles.assert_called_once_with("CCO")

    def test_calls_show_status_message_on_success(self):
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(smiles_result="c1ccccc1")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "benzene.png")
        ctx.show_status_message.assert_called_once()

    def test_does_not_load_when_smiles_empty(self):
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(smiles_result="")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "blank.png")
        ctx.load_from_smiles.assert_not_called()

    def test_does_not_load_on_error(self):
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(error_msg="DECIMER is not installed.")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "img.png")
        ctx.load_from_smiles.assert_not_called()

    def test_does_not_load_on_runtime_error(self):
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(error_msg="model load failed")
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "bad.png")
        ctx.load_from_smiles.assert_not_called()

    def test_smiles_truncated_in_status_message(self):
        long_smiles = "C" * 100
        ctx = self._make_ctx()
        worker_cls = _make_sync_worker_class(smiles_result=long_smiles)
        with patch.object(_dialog_mod, "_DECIMERWorker", worker_cls):
            _dialog_mod.run_decimer_async(ctx, "big.png")
        msg = ctx.show_status_message.call_args[0][0]
        assert len(msg) < len(long_smiles) + 50
