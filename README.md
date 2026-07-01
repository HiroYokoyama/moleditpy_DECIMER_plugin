# MoleditPy DECIMER Plugin

A [MoleditPy](https://github.com/HiroYokoyama/moleditpy) plugin that imports chemical structures from PNG/JPG images using the [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) deep learning model.

## Features

- **Menu action** — `File → Import from Image (DECIMER)...` opens a dialog to browse for an image, predict the SMILES, then load the structure into the editor
- **Drag-and-drop** — drop a `.png`, `.jpg`, or `.jpeg` file onto the MoleditPy window to predict and load automatically
- **Non-blocking UI** — prediction runs in a background thread; the editor stays responsive during the 10–30 s inference
- **Subprocess isolation** — TensorFlow loads in a fresh child process, avoiding DLL conflicts with PyQt6/PyVista on Windows (see [Windows note](#windows-tensorflow-dll-conflict) below)

## Requirements

- MoleditPy ≥ 4.0
- Python 3.11+
- [DECIMER](https://pypi.org/project/DECIMER/) (`pip install DECIMER`)
- Pillow (`pip install Pillow`)
- **Windows only:** `tensorflow-cpu` — see [below](#windows-tensorflow-dll-conflict)

## Installation

1. Install the plugin dependencies:

   ```bash
   pip install DECIMER Pillow
   ```

   On Windows, also install the CPU build of TensorFlow (required — see below):

   ```bash
   pip install tensorflow-cpu
   ```

2. Copy (or symlink) the `decimer_plugin/` directory into your MoleditPy plugins folder:

   ```
   %APPDATA%\moleditpy\plugins\decimer_plugin\
   ```

3. Restart MoleditPy. The menu item `File → Import from Image (DECIMER)...` should appear.

## First use — model download

On the first prediction, DECIMER automatically downloads its model weights (~500 MB) to `~/.data/DECIMER-V2`. This happens inside the background subprocess and may take several minutes depending on your connection. The progress dialog stays open until the download and inference finish. Subsequent predictions reuse the cached model and are much faster (~5–15 s).

## Windows — TensorFlow DLL conflict

### The problem

MoleditPy loads PyQt6, PyVista, and VTK at startup. These frameworks load several Windows system DLLs into the process. TensorFlow's `_pywrap_tensorflow_internal.pyd` relies on the same DLLs, and the versions already loaded by PyVista/VTK conflict with what TensorFlow expects. The result is:

```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

or, with `tensorflow-cpu`:

```
Failed to load _pywrap_tensorflow_common.dll: INITIALIZATION FAILED (0x45A)
```

This error occurs even when TensorFlow imports cleanly in a standalone Python session.

### The fix

The plugin runs `predict_SMILES` in a **fresh subprocess** (`sys.executable -c ...`). The child process starts with no PyQt6/PyVista DLLs already loaded, so TensorFlow initialises cleanly. Only the resulting SMILES string is passed back to MoleditPy over stdout.

This is handled transparently — no configuration is needed.

### Install tensorflow-cpu, not tensorflow

The GPU build of TensorFlow (`tensorflow`) additionally requires CUDA and cuDNN DLLs. Unless you have a compatible GPU and the correct CUDA toolkit installed, use the CPU build:

```bash
pip uninstall tensorflow -y
pip install tensorflow-cpu
```

TF ≥ 2.18 CPU builds require a CPU with **AVX2** support (Intel Haswell / 2013+ or AMD Ryzen). If your CPU is older, you will need to build TensorFlow from source or use an older release.

## Development

```bash
# Clone and run tests (no GPU, no DECIMER download needed — all mocked)
git clone https://github.com/HiroYokoyama/moleditpy_DECIMER_plugin
cd moleditpy_DECIMER_plugin
pip install pytest
python -m pytest tests/ -v
```

Tests run fully headlessly: PyQt6, DECIMER, TensorFlow, and PIL are all replaced with lightweight stubs. No model download or GPU is required.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
