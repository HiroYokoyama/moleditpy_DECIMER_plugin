
from ._qt_callbacks import connect_queued
import logging
import subprocess
import sys

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)
from PySide6.QtCore import Qt, QThread, Signal

_INSTALL_HINT = "Install with:\n  pip install DECIMER"

_PROGRESS_MSG = (
    "DECIMER: predicting structure from image…\n\n"
    "On first use, the model is downloaded automatically (~500 MB).\n"
    "This may take several minutes — the app stays responsive."
)

# Dialog destruction must not destroy a thread still waiting for DECIMER.
_RUNNING_WORKERS = set()


def _start_worker(worker):
    _RUNNING_WORKERS.add(worker)

    def release():
        _RUNNING_WORKERS.discard(worker)
        worker.deleteLater()

    connect_queued(worker.finished, release, cleanup=(worker.finished,))
    worker.start()

# Runs in a subprocess so TensorFlow never loads into MoleditPy's process,
# avoiding DLL conflicts with PySide6/PyVista/VTK.
_SUBPROCESS_SCRIPT = (
    "import sys;"
    "from DECIMER import predict_SMILES;"
    "result = predict_SMILES(sys.argv[1]);"
    "print(result if result else '', flush=True)"
)


def _smiles_problem(smiles: str) -> str:
    """Return a reason *smiles* cannot be loaded, or "" when it is usable.

    DECIMER is an OCR model, so an unparseable SMILES is a normal outcome for a
    blurry or crowded drawing. The host's load_from_smiles reports such a
    failure only on the status bar and returns without raising, so without this
    check the plugin overwrote that message with its own success text and closed
    the dialog as though the structure had loaded.
    """
    if not smiles:
        return "DECIMER returned an empty result."
    try:
        from rdkit import Chem
    except ImportError:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return f"RDKit could not parse the predicted SMILES: {exc}"
    if mol is None:
        return "The predicted SMILES is not valid and cannot be drawn."
    return ""


def _import_error_message(exc: ImportError) -> str:
    msg = str(exc)
    if "DECIMER" in msg or not msg:
        return f"DECIMER is not installed.\n\n{_INSTALL_HINT}"
    if "DLL load failed" in msg or "DLL" in msg or "_pywrap_tensorflow" in msg:
        return (
            "TensorFlow failed to load a Windows DLL required by DECIMER.\n\n"
            "Common causes:\n"
            "  • Missing Microsoft Visual C++ Redistributable\n"
            "  • CUDA / cuDNN version mismatch (if using GPU)\n"
            "  • Incompatible TensorFlow version\n\n"
            "Try installing the CPU-only build:\n"
            "  pip install tensorflow-cpu\n\n"
            f"Details: {msg}"
        )
    return (
        f"DECIMER is installed but a required dependency could not be imported:\n\n"
        f"  {msg}\n\n"
        f"Try reinstalling:\n  pip install --upgrade DECIMER"
    )


def _predict_smiles_subprocess(image_path: str) -> str:
    """Run predict_SMILES in a fresh subprocess to avoid DLL conflicts with MoleditPy."""
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, image_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr
        if "No module named 'DECIMER'" in stderr or 'No module named "DECIMER"' in stderr:
            raise ImportError("No module named 'DECIMER'")
        if "DLL load failed" in stderr or "_pywrap_tensorflow" in stderr:
            raise ImportError(stderr.strip())
        raise RuntimeError(stderr.strip() or f"Subprocess exited with code {result.returncode}")
    return result.stdout.strip()


class _DECIMERWorker(QThread):
    """Run DECIMER prediction in a subprocess via a background thread."""

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._image_path = image_path

    def run(self) -> None:
        try:
            smiles = _predict_smiles_subprocess(self._image_path)
            self.completed.emit(smiles)
        except ImportError as exc:
            logging.error("DECIMER plugin: import failed: %s", exc)
            self.failed.emit(_import_error_message(exc))
        except subprocess.TimeoutExpired:
            logging.error("DECIMER plugin: prediction timed out")
            self.failed.emit(
                "Prediction timed out (10 minutes).\n"
                "The model download may still be in progress — try again."
            )
        except Exception as exc:
            logging.exception("DECIMER plugin: prediction failed")
            self.failed.emit(str(exc))


class ImportFromImageDialog(QDialog):
    """Dialog: pick an image, run DECIMER in background, load the SMILES into MoleditPy."""

    def __init__(self, parent=None, context=None):
        super().__init__(parent)
        self._context = context
        self._worker: _DECIMERWorker | None = None
        self._progress: QProgressDialog | None = None
        self.setWindowTitle("Import from Image — DECIMER")
        self.resize(560, 160)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Row 1 – image path
        row1 = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            "Select a PNG or JPG image of a chemical structure…"
        )
        self._path_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse)
        row1.addWidget(self._path_edit)
        row1.addWidget(btn_browse)
        layout.addLayout(row1)

        # Row 2 – predicted SMILES output
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("SMILES:"))
        self._smiles_edit = QLineEdit()
        self._smiles_edit.setReadOnly(True)
        self._smiles_edit.setPlaceholderText("Predicted SMILES will appear here…")
        row2.addWidget(self._smiles_edit)
        layout.addLayout(row2)

        # Row 3 – action buttons
        row3 = QHBoxLayout()
        self._btn_predict = QPushButton("Predict Structure")
        self._btn_predict.setEnabled(False)
        self._btn_predict.clicked.connect(self._start_prediction)

        self._btn_load = QPushButton("Load into Editor")
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._load_smiles)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)

        row3.addWidget(self._btn_predict)
        row3.addStretch()
        row3.addWidget(self._btn_load)
        row3.addWidget(btn_close)
        layout.addLayout(row3)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Chemical Structure Image",
            "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._smiles_edit.clear()
            self._btn_predict.setEnabled(True)
            self._btn_load.setEnabled(False)

    def _start_prediction(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return

        self._btn_predict.setEnabled(False)
        self._btn_load.setEnabled(False)

        self._progress = QProgressDialog(_PROGRESS_MSG, None, 0, 0, self)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()

        self._worker = _DECIMERWorker(path)
        connect_queued(self._worker.completed, self._on_done, self, (str,), cleanup=(self._worker.finished,))
        connect_queued(self._worker.failed, self._on_error, self, (str,), cleanup=(self._worker.finished,))
        _start_worker(self._worker)

    def _on_done(self, smiles: str) -> None:
        self._close_progress()
        self._btn_predict.setEnabled(True)
        if smiles:
            self._smiles_edit.setText(smiles)
            self._btn_load.setEnabled(True)
        else:
            QMessageBox.warning(
                self,
                "No Structure Found",
                "DECIMER could not predict a SMILES from this image.\n"
                "Try a clearer image with a single chemical structure drawn.",
            )

    def _on_error(self, message: str) -> None:
        self._close_progress()
        self._btn_predict.setEnabled(True)
        QMessageBox.critical(self, "DECIMER Error", message)

    def _close_progress(self) -> None:
        if self._progress is not None:
            try:
                self._progress.close()
            except RuntimeError as exc:
                logging.warning("DECIMER plugin: could not close progress dialog: %s", exc)
            self._progress = None

    def _load_smiles(self) -> None:
        smiles = self._smiles_edit.text().strip()
        if not smiles or self._context is None:
            return
        problem = _smiles_problem(smiles)
        if problem:
            QMessageBox.warning(
                self,
                "Unusable Prediction",
                f"{problem}\n\nSMILES: {smiles}\n\n"
                "Try a clearer image, or edit the SMILES above before loading.",
            )
            return
        try:
            self._context.load_from_smiles(smiles)
            self._context.show_status_message("Structure loaded from image.", 4000)
            self.close()
        except Exception as exc:
            logging.exception("DECIMER plugin: failed to load SMILES into editor")
            QMessageBox.critical(
                self, "Import Error", f"Failed to load structure:\n{exc}"
            )


def run_decimer_async(context, image_path: str) -> None:
    mw = context.get_main_window()

    progress = QProgressDialog(_PROGRESS_MSG, None, 0, 0, mw)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()

    worker = _DECIMERWorker(image_path)

    def _on_done(smiles: str) -> None:
        try:
            progress.close()
        except RuntimeError as exc:
            logging.warning("DECIMER plugin: could not close progress dialog: %s", exc)
        if not smiles:
            QMessageBox.warning(
                mw,
                "No Structure Found",
                "DECIMER could not predict a structure from the dropped image.",
            )
            return
        problem = _smiles_problem(smiles)
        if problem:
            QMessageBox.warning(
                mw,
                "Unusable Prediction",
                f"{problem}\n\nSMILES: {smiles}\n\n"
                "Try a clearer image with a single structure drawn.",
            )
            return
        context.load_from_smiles(smiles)
        context.show_status_message(
            f"Structure loaded from image: {smiles[:60]}", 5000
        )

    def _on_error(msg: str) -> None:
        try:
            progress.close()
        except RuntimeError as exc:
            logging.warning("DECIMER plugin: could not close progress dialog: %s", exc)
        QMessageBox.critical(mw, "DECIMER Error", msg)

    connect_queued(worker.completed, _on_done, mw, (str,), cleanup=(worker.finished,))
    connect_queued(worker.failed, _on_error, mw, (str,), cleanup=(worker.finished,))
    _start_worker(worker)


run_decimer_sync = run_decimer_async
