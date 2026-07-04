import logging

PLUGIN_NAME = "DECIMER Image Importer"
PLUGIN_VERSION = "1.0.1"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Import chemical structures from PNG/JPG/JPEG images using the DECIMER "
    "deep learning model (SMILES prediction from chemical structure drawings)."
)
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"
PLUGIN_DEPENDENCIES = ["DECIMER", "Pillow"]

_context = None

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def initialize(context):
    global _context
    _context = context

    context.add_menu_action("File/Import from Image (DECIMER)...", lambda: _open_import_dialog(context))

    def handle_drop(file_path: str) -> bool:
        import os
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            return False
        _run_from_drop(context, file_path)
        return True

    context.register_drop_handler(handle_drop, priority=5)


def _open_import_dialog(context) -> None:
    from .dialog import ImportFromImageDialog

    existing = context.get_window("dialog")
    if existing is not None:
        try:
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return
            existing.close()
            existing.deleteLater()
        except RuntimeError as exc:
            logging.warning("DECIMER plugin: could not close existing dialog: %s", exc)
        context.register_window("dialog", None)

    mw = context.get_main_window()
    dlg = ImportFromImageDialog(mw, context)
    context.register_window("dialog", dlg)
    dlg.show()


def _run_from_drop(context, image_path: str) -> None:
    from .dialog import run_decimer_async
    run_decimer_async(context, image_path)
