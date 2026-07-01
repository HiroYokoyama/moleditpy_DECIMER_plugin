import logging

from PyQt6.QtWidgets import (
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal

_INSTALL_HINT = "Install with:\n  pip install DECIMER"

_PROGRESS_MSG = (
    "DECIMER: predicting structure from image…\n\n"
    "On first use, the model is downloaded automatically (~500 MB).\n"
    "This may take several minutes — the app stays responsive."
)


def _import_error_message(exc: ImportError) -> str:
    """Return a user-friendly message for an ImportError during DECIMER import."""
    msg = str(exc)
    if "DECIMER" in msg or not msg:
        return f"DECIMER is not installed.\n\n{_INSTALL_HINT}"
    if "DLL load failed" in msg or "DLL" in msg:
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
    # A transitive dependency (e.g. tensorflow, transformers) failed to import
    return (
        f"DECIMER is installed but a required dependency could not be imported:\n\n"
        f"  {msg}\n\n"
        f"Try reinstalling:\n  pip install --upgrade DECIMER"
    )


class _DECIMERWorker(QThread):
    """Run DECIMER.predict_SMILES in a background thread."""

    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._image_path = image_path

    def run(self) -> None:
        try:
            from DECIMER import predict_SMILES  # heavy: loads TF + downloads model
            smiles = predict_SMILES(self._image_path)
            self.finished.emit(smiles or "")
        except ImportError as exc:
            logging.error("DECIMER plugin: import failed: %s", exc)
            self.failed.emit(_import_error_message(exc))
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

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

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

        self._worker = _DECIMERWorker(path, parent=self)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

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
        try:
            self._context.load_from_smiles(smiles)
            self._context.show_status_message("Structure loaded from image.", 4000)
            self.close()
        except Exception as exc:
            logging.exception("DECIMER plugin: failed to load SMILES into editor")
            QMessageBox.critical(
                self, "Import Error", f"Failed to load structure:\n{exc}"
            )


# ------------------------------------------------------------------
# Drop-handler helper: async so the main thread never freezes
# ------------------------------------------------------------------


def run_decimer_async(context, image_path: str) -> None:
    """Start a background DECIMER prediction for a dropped image file.

    Returns immediately — progress dialog and result handling happen via signals.
    """
    mw = context.get_main_window()

    progress = QProgressDialog(_PROGRESS_MSG, None, 0, 0, mw)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()

    # parent=mw keeps the worker alive until the main window is destroyed
    worker = _DECIMERWorker(image_path, parent=mw)

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

    worker.finished.connect(_on_done)
    worker.failed.connect(_on_error)
    worker.start()


# Keep the old name as an alias so existing tests still pass
run_decimer_sync = run_decimer_async
