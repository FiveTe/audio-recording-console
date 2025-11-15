# SoundSmith

SoundSmith is a PyQt-based acoustic lab for designing, emitting, recording, and analyzing custom tactile/audio signatures. It wraps a live spectrum analyzer, a session-based recording pipeline, and a pluggable analysis/modeling framework into a single desktop app so you can build reproducible datasets for downstream robotics or ML work.

## Features

- **Live signal generator** – Emit sine waves, single chirps, or 5-chirp bursts with adjustable frequency and amplitude for calibration or excitation sweeps.
- **Real-time spectrum + harmonics view** – See the FFT of the incoming audio stream, track up to 10 harmonics, and keep tabs on microphone gain before recording.
- **Session-aware recording workflow** – Create sessions, define rich labels + abbreviations, then capture either passive recordings or active chirp responses with countdown guidance and save/discard review.
- **Managed datasets on disk** – Each capture writes a WAV, averaged-spectrum JSON, and spectrogram PNG under `data/recordings/<session>/<abbr>_<amp>_<type>/`.
- **Modular analyzers** – Plug in as many analyzer modules as you like (FFT overlays, waveform inspections, etc.) and access them via the “Open Modular Analyzer” dialog.
- **Pluggable model trainers** – Register custom ML training routines that consume the entire recorded corpus.

## Repository Layout

```
soundsmith_app/
├── app.py                   # Main PyQt6 entry point
├── analyzers/               # Drop-in analyzer modules (auto-loaded)
│   └── default_analyzers.py
├── models/                  # Drop-in model training modules (auto-loaded)
│   └── default_models.py
└── data/
    ├── labels/labels.json   # Persistent {full_label: abbreviation} map
    └── recordings/          # Organized session/label folders with WAV/JSON/PNG
```

`app.py` automatically creates the `data`, `data/recordings`, `data/labels`, `analyzers`, and `models` folders (plus the default modules) the first time you launch the app.

## Prerequisites

- Python 3.10+
- Desktop OS with audio input/output devices accessible through PortAudio (SoundDevice backend)
- The following Python packages:
  - `PyQt6`
  - `numpy`
  - `scipy`
  - `sounddevice`
  - `matplotlib`

Install them with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install PyQt6 numpy scipy sounddevice matplotlib
```

> **Note:** `sounddevice` depends on PortAudio. On Linux you may need `sudo apt install libportaudio2` (or the equivalent package for your distro) before installing the Python wheel.

## Running the Application

1. Activate your environment and start the GUI:
   ```bash
   cd soundsmith_app
   python app.py
   ```
2. Select the desired input and output devices when prompted. These settings feed both the generator (`EmitterThread`) and analyzer (`AnalyzerThread`).
3. The main window opens with three panels:
   - **Live FFT + Harmonics** on the left.
   - **Output Control** for generating sine/chirp/5-chirp stimuli.
   - **Recording & Analysis** for session selection, label management, recording buttons, analyzer access, and basic model training.

Keep the terminal open; analyzer/model modules log useful status information there.

## Recording Workflow

1. **Create/Select a Session** – Use the editable “Session” combo box (e.g., `grasp_2025-02-10`).
2. **Define Labels (optional)** – Expand “Add New Label Definition” to register `Full Name → Abbreviation` pairs. The abbreviation becomes the folder prefix (e.g., `OPENPALM → OP`).
3. **Choose Label for Recording** – Pick the label you want the sample stored under.
4. **Record** – Click either:
   - `Record Passive` for ambient captures (10 s by default, microphone input only).
   - `Record Active (5 Chirps)` to emit the built-in 5-chirp waveform (6 s total) while recording the response.
5. **Countdown + Capture** – The UI counts down before recording starts, then displays the remaining seconds. The generator is automatically muted/unmuted depending on passive vs. active mode.
6. **Review & Save** – After processing, the app shows Save/Discard buttons. Saving writes:
   - `<base>.wav`
   - `<base>_data.json` (metadata + averaged spectrum)
   - `<base>_spectrogram.png`

   The files live under `data/recordings/<session>/<abbr>_<amp>_<type>/`. Metadata references absolute WAV paths so analyzers can reload audio later.
7. **Repeat** – The tree view lists all recordings grouped by label. Use it as the data source for analyzers and models.

## Using the Modular Analyzer

1. Click **📊 Open Modular Analyzer**.
2. Select one or more recordings per the analyzer’s requirements (e.g., the default “Comparison (Overlaid FFT)” accepts multiple, “Detailed 5-Chirp Analysis” requires exactly one).
3. Choose an analyzer module from the dropdown and press **Run Analysis**.
4. Results render in the main viewport and thumbnail bar. Click thumbnails to switch between figures generated by the module.

Analyzers receive the entire `recorded_signatures` map along with the selected IDs, FFT chunk size, and sample rate—ideal for advanced signal processing or ML visualization.

## Training Models

- Pick a registered model from the **Train Model** dropdown (defaults are placeholders).
- Press **Train New Model** to hand the full `recorded_signatures` dict to the selected training function.
- Implement your own trainers (see below) to integrate with scikit-learn, PyTorch, etc.

## Extending the App

Both analyzers and models are discovered dynamically. Add `.py` files under `analyzers/` or `models/` that call `@register_analyzer` / `@register_model` to appear in the UI menus on the next launch.

### Analyzer Example

```python
# analyzers/my_histogram.py
import numpy as np
from matplotlib.figure import Figure

@register_analyzer("Spectral Histogram")
def spectral_histogram(all_data, selected_ids, chunk_size, rate):
    uid = selected_ids[0]
    spectrum = all_data[uid]["spectral_data"]
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(spectrum, bins=50)
    ax.set_title("Magnitude Histogram")
    ax.set_xlabel("dB")
    ax.set_ylabel("Count")
    return {"Histogram": fig}
```

### Model Example

```python
# models/svm_classifier.py
from sklearn import svm

@register_model("SVM Classifier")
def train_svm_classifier(data):
    X, y = [], []
    for sample in data.values():
        X.append(sample["spectral_data"])
        y.append(sample["label"])
    clf = svm.SVC()
    clf.fit(X, y)
    print("Trained SVM with", len(X), "samples")
```

Modules execute with `register_*` utilities already injected into their namespace, so no imports from `app.py` are necessary.

## Data & Label Files

- **Labels:** `data/labels/labels.json` stores a JSON object (e.g., `{ "OpenPalm_Horizontal": "OP_H" }`). The app keeps this synchronized whenever you add labels or import legacy recordings.
- **Recordings:** Each `_data.json` file includes `base_filename`, `session`, `label`, `sound_type`, `amplitude`, `spectral_data`, `wav_path`, and `spec_path`. This format lets analyzers reload WAV files lazily.

Back up the `data/` directory to preserve your dataset. Deleting entries from disk removes them from the UI the next time you press “Rescan” (launching the app triggers a scan automatically).

## Troubleshooting

- **No audio devices detected:** Ensure microphones/speakers are connected and not locked by other apps. On headless machines, use virtual audio cables.
- **sounddevice `PortAudioError`:** Install OS-specific PortAudio libraries or select different devices at launch.
- **PyQt crashes on Wayland:** Set `QT_QPA_PLATFORM=xcb` before running (`QT_QPA_PLATFORM=xcb python app.py`).
- **Analyzer/model not showing up:** Verify the module file ends with `.py`, has no syntax errors, and calls the appropriate `register_*` decorator at import time.

## Next Steps

- Flesh out real ML models under `models/`.
- Create custom analyzer modules tailored to your signals.
- Automate dataset exports (e.g., to CSV or HDF5) using the metadata written to `_data.json`.

Happy measuring!
