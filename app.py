import sys
import json
import os
import glob
import io
import contextlib
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi, chirp
from scipy.io.wavfile import write as write_wav, read as read_wav

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QSlider, QLabel, QPushButton, QLineEdit, QListWidget,
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QInputDialog, QComboBox, QDialog, QSplitter, QScrollArea, QCheckBox,
    QFrame, QSizePolicy, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTabWidget, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QDateTime
from PyQt6.QtGui import QPixmap, QImage, QPainter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# =============================================================================
# APPLICATION PATHS & SETUP
# =============================================================================
try:
    APP_ROOT = os.path.dirname(os.path.realpath(__file__))
except NameError:
    APP_ROOT = os.path.dirname(os.path.realpath(sys.argv[0]))

DATA_DIR = os.path.join(APP_ROOT, "data")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
LABELS_FILE = os.path.join(LABELS_DIR, "labels.json")
ANALYZERS_DIR = os.path.join(APP_ROOT, "analyzers")
MODELS_DIR = os.path.join(APP_ROOT, "models")
LOOPBACK_FILE = os.path.join(DATA_DIR, "loopback_ref.npy")

# =============================================================================
# DYNAMIC MODULE REGISTRIES & DECORATORS
# =============================================================================
REGISTERED_ANALYZERS = {}
REGISTERED_MODELS = {}

def register_analyzer(name):
    """Decorator to register a new analyzer function."""
    def decorator(func):
        REGISTERED_ANALYZERS[name] = func
        return func
    return decorator

def register_model(name):
    """Decorator to register a new model training function."""
    def decorator(func):
        REGISTERED_MODELS[name] = func
        return func
    return decorator

# =============================================================================
# DUMMY MODULE CONTENT (for first run)
# =============================================================================

DEFAULT_ANALYZER_CODE = """
import numpy as np
from scipy.io.wavfile import read as read_wav
from matplotlib.figure import Figure
import os

# This module is dynamically loaded by app.py
# It expects 'register_analyzer' to be in its global scope.
print(f"Loading analyzers from {os.path.basename(__file__)}...")

@register_analyzer("Comparison (Overlaid FFT)")
def analyze_comparison(all_data, selected_ids, chunk_size, rate):
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    freq_bins = np.fft.rfftfreq(chunk_size, 1 / rate)
    
    for uid in selected_ids:
        data = all_data[uid]
        ax.plot(freq_bins, data['spectral_data'], label=data['base_filename'])

    ax.set_title("Comparison of Averaged Spectrums")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlim(20, rate / 2)
    ax.set_ylim(-80, 20)
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    
    return {"Overlaid FFT": fig}

@register_analyzer("Detailed 5-Chirp Analysis")
def analyze_chirps_detailed(all_data, selected_ids, chunk_size, rate):
    if len(selected_ids) != 1:
        raise ValueError("This analyzer requires exactly one recording to be selected.")
    
    uid = selected_ids[0]
    data = all_data[uid]
    
    if not os.path.exists(data['wav_path']):
        raise FileNotFoundError(f"Missing audio file: {data['wav_path']}")
        
    sr, raw_audio = read_wav(data['wav_path'])

    # -- 1. Waveform & Segmentation --
    fig_wave = Figure(figsize=(8, 4))
    ax_wave = fig_wave.add_subplot(111)
    ax_wave.plot(np.arange(len(raw_audio)) / sr, raw_audio, color='lightblue', zorder=1)
    ax_wave.set_title(f"Waveform: {data['base_filename']}")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)

    window_size, hop_size = 2048, 512
    rms = np.array([np.sqrt(np.mean(raw_audio[i:i+window_size]**2)) for i in range(0, len(raw_audio)-window_size, hop_size)])
    thresh = np.max(rms) * 0.5
    peaks = [i for i in range(1, len(rms)-1) if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] > rms[i+1]]
    min_dist = (sr * 0.9) / hop_size
    debounced = [peaks[0]] if peaks else []
    for p in peaks[1:]:
        if p - debounced[-1] > min_dist: debounced.append(p)

    chirp_segments = []
    if len(debounced) == 5:
        for i, p in enumerate(debounced):
            start = p * hop_size
            end = start + int(sr * 0.8)
            chirp_segments.append(raw_audio[start:end])
            ax_wave.axvspan(start/sr, end/sr, color=f'C{i}', alpha=0.3, zorder=2, label=f"Chirp {i+1}")
        ax_wave.legend()
    
    fig_wave.tight_layout()

    # -- 2. Overlaid FFT of segments --
    fig_fft = Figure(figsize=(8, 4))
    ax_fft = fig_fft.add_subplot(111)
    if chirp_segments:
        for i, seg in enumerate(chirp_segments):
            N = len(seg)
            freqs = np.fft.rfftfreq(N, 1/sr)
            mag = np.abs(np.fft.rfft(seg * np.hanning(N)))
            ax_fft.plot(freqs, 20*np.log10(mag + 1e-9), label=f'Chirp {i+1}', color=f'C{i}', alpha=0.8)
    else:
        ax_fft.text(0.5, 0.5, "Could not automatically segment 5 chirps", ha='center')

    ax_fft.set_title("Spectral Analysis of Individual Chirps")
    ax_fft.set_xscale('log')
    ax_fft.set_xlim(20, sr/2)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude [dB]")
    ax_fft.grid(True, which="both", ls="--")
    if chirp_segments: ax_fft.legend()
    fig_fft.tight_layout()

    # -- 3. Spectrogram --
    fig_spec = Figure(figsize=(8, 4))
    ax_spec = fig_spec.add_subplot(111)
    ax_spec.specgram(raw_audio, Fs=sr, NFFT=1024, cmap='viridis')
    ax_spec.set_title("Full Spectrogram")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.axis('tight')
    fig_spec.tight_layout()

    return {
        "Waveform Segmentation": fig_wave,
        "Chirp FFTs": fig_fft,
        "Full Spectrogram": fig_spec
    }
"""

DEFAULT_MODEL_CODE = """
import os
# This module is dynamically loaded by app.py
# It expects 'register_model' to be in its global scope.
print(f"Loading models from {os.path.basename(__file__)}...")

@register_model("SVM (Placeholder)")
def train_svm(data):
    print("--- Training SVM Model ---")
    labels = [d['label'] for d in data.values()]
    print(f"Received {len(labels)} data points.")
    print(f"Unique labels: {set(labels)}")
    print("Model training not implemented.")
    print("--------------------------")
    return None

@register_model("KNN (Placeholder)")
def train_knn(data):
    print("--- Training KNN Model ---")
    labels = [d['label'] for d in data.values()]
    print(f"Received {len(labels)} data points.")
    print(f"Unique labels: {set(labels)}")
    print("Model training not implemented.")
    print("--------------------------")
    return None
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories_and_files():
    """Creates all necessary app directories and default files."""
    print("Setting up application structure...")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(ANALYZERS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'w') as f:
            json.dump({}, f) # Now saves an empty dictionary
            print(f"Created {LABELS_FILE}")
            
    default_analyzer_file = os.path.join(ANALYZERS_DIR, "default_analyzers.py")
    if not os.path.exists(default_analyzer_file):
        with open(default_analyzer_file, 'w') as f:
            f.write(DEFAULT_ANALYZER_CODE)
            print(f"Created {default_analyzer_file}")
            
    default_model_file = os.path.join(MODELS_DIR, "default_models.py")
    if not os.path.exists(default_model_file):
        with open(default_model_file, 'w') as f:
            f.write(DEFAULT_MODEL_CODE)
            print(f"Created {default_model_file}")

def load_modules_from_dir(directory):
    """Dynamically loads .py files and registers decorators."""
    print(f"Loading modules from: {directory}")
    py_files = glob.glob(os.path.join(directory, "*.py"))
    
    for py_file in py_files:
        if os.path.basename(py_file) == "__init__.py":
            continue
        
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            
            # Provide decorators to the module's execution scope
            module_globals = {
                "register_analyzer": register_analyzer,
                "register_model": register_model,
                "__name__": f"loaded_module.{os.path.basename(py_file)}",
                "__file__": py_file,
            }
            exec(code, module_globals)
        except Exception as e:
            print(f"Failed to load module {py_file}: {e}")

def load_labels():
    """Loads label map {FullName: Abbr} from data/labels/labels.json."""
    try:
        with open(LABELS_FILE, 'r') as f:
            labels_data = json.load(f)
            if isinstance(labels_data, list):
                # Convert old list format to new dict format
                print("Old labels list format detected. Converting...")
                labels_map = {label: label.replace(" ", "_").upper() for label in labels_data}
                save_labels(labels_map) # Save back in new format
                return labels_map
            return labels_data # Should be a dict
    except Exception as e:
        print(f"Could not load labels file: {e}")
        return {}

def save_labels(labels_map):
    """Saves label map {FullName: Abbr} to data/labels/labels.json."""
    try:
        with open(LABELS_FILE, 'w') as f:
            json.dump(labels_map, f, indent=4, sort_keys=True)
    except Exception as e:
        print(f"Could not save labels file: {e}")

# =============================================================================
# --- CONFIGURATION (Global) ---
# =============================================================================
SAMPLING_RATE = 44100
CHUNK_SIZE = 2048
RECORDING_DURATION_PASSIVE_S = 10
RECORDING_DURATION_ACTIVE_S = 6
RECORDING_DURATION_HIGH_ENERGY_S = 12
RECORDING_DURATION_H5_HEC_S = 6

SOUND_TYPE_CODES = {
    "Sine Wave": "SIN",
    "Chirp": "CH",
    "5 Chirps": "5CH",
    "High Energy Chirp": "HEC",
    "Passive": "PAS"
}

# =============================================================================
# CORE CLASSES
# =============================================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, figure=None):
        if figure is None:
             figure = Figure(figsize=(width, height), dpi=dpi)
             self.axes = figure.add_subplot(111)
        else:
             self.axes = None # Axes are part of the passed figure
        super(MplCanvas, self).__init__(figure)

class EmitterThread(QThread):
    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.is_running = True
        self.start_idx = 0
        self.sound_type = "Sine Wave"
        self.frequency = 200.0
        self.amplitude = 0.5
        self.is_muted = True
        self.chirp_buffer = self._generate_chirp(duration=1.0)
        self.five_chirps_buffer = self._generate_5_chirps()
        self.high_energy_chirp_buffer = self._generate_high_energy_chirp()

    def _generate_chirp(self, duration=1.0, f0=20, f1=20000):
        t = np.linspace(0, duration, int(SAMPLING_RATE * duration), False)
        return chirp(t, f0=f0, f1=f1, t1=duration, method='linear').astype(np.float32)

    def _generate_5_chirps(self):
        segment = np.concatenate([self._generate_chirp(duration=0.8), np.zeros(int(SAMPLING_RATE * 0.2))])
        return np.tile(segment, 5)

    def _generate_high_energy_chirp(self):
        duration = RECORDING_DURATION_HIGH_ENERGY_S
        # Focus power where hand/bone response is strongest (mid-band region)
        return self._generate_chirp(duration=6, f0=300, f1=6000)

    def audio_callback(self, outdata, frames, time, status):
        if self.is_muted:
            outdata.fill(0)
            return

        if self.sound_type == "Sine Wave":
            t = (self.start_idx + np.arange(frames)) / SAMPLING_RATE
            wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
            outdata[:] = wave.reshape(-1, 1)
            self.start_idx += frames
        else:
            if self.sound_type == "High Energy Chirp":
                buffer = self.high_energy_chirp_buffer
            elif self.sound_type == "Chirp":
                buffer = self.chirp_buffer
            else:
                buffer = self.five_chirps_buffer
            num_samples = len(buffer)
            start = self.start_idx % num_samples
            end = start + frames
            if end < num_samples:
                outdata[:] = self.amplitude * buffer[start:end].reshape(-1, 1)
            else:
                outdata[:] = self.amplitude * np.concatenate((buffer[start:], buffer[:end % num_samples])).reshape(-1, 1)
            self.start_idx += frames

    def run(self):
        with sd.OutputStream(device=self.device_id, samplerate=SAMPLING_RATE, channels=1, callback=self.audio_callback):
            while self.is_running:
                self.msleep(100)

    def set_muted(self, muted):
        self.is_muted = muted

    def update_parameters(self, frequency, amplitude):
        self.frequency = frequency
        self.amplitude = amplitude

    def set_sound_type(self, sound_type):
        self.start_idx = 0
        self.sound_type = sound_type

    def stop(self):
        self.is_running = False

# =============================================================================
# CORRUPTION MONITOR THREAD
# =============================================================================
class CorruptionMonitor(QThread):
    status_update = pyqtSignal(str)

    def __init__(self, analyzer_thread, interval_ms=500):
        super().__init__()
        self.analyzer_thread = analyzer_thread
        self.interval_ms = interval_ms
        self.is_running = True

    def run(self):
        while self.is_running:
            chunk = getattr(self.analyzer_thread, "last_chunk", None)
            if chunk is not None:
                if not np.isfinite(chunk).all() or np.max(np.abs(chunk)) > 5:
                    self.status_update.emit("Warning: Live chunk contains invalid samples.")
                else:
                    self.status_update.emit("")
            self.msleep(self.interval_ms)

    def stop(self):
        self.is_running = False

# =============================================================================
# LOOPBACK CAPTURE THREAD
# =============================================================================
class LoopbackCaptureThread(QThread):
    finished_capture = pyqtSignal(np.ndarray)
    failed_capture = pyqtSignal(str)

    def __init__(self, device_id, duration_s):
        super().__init__()
        self.device_id = device_id
        self.duration_s = duration_s
        self.is_running = True

    def run(self):
        try:
            self.msleep(50)  # small delay to ensure device stable
            rec = sd.rec(int(self.duration_s * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, device=self.device_id, dtype='float32')
            sd.wait()
            if not self.is_running:
                return
            ref = rec[:, 0]
            self.finished_capture.emit(ref)
        except Exception as e:
            self.failed_capture.emit(str(e))

    def stop(self):
        self.is_running = False
class AnalyzerThread(QThread):
    new_data = pyqtSignal(np.ndarray, np.ndarray, list)
    recording_finished = pyqtSignal(str, str, str, float, np.ndarray, np.ndarray) # session, label, sound_type, amp, ...

    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.is_running = True
        self.filter_sos = None
        self.filter_zi = None
        self.audio_queue = []
        self.fundamental_freq = 200.0
        self.is_recording = False
        self.recording_buffer_fft = []
        self.recording_buffer_raw = []
        self.last_chunk = None
        self.reference_waveform = None
        self.use_reference = False

    def audio_callback(self, indata, frames, time, status):
        raw_audio = indata[:, 0]
        if self.is_recording:
            self.recording_buffer_raw.append(raw_audio.copy())
        if self.filter_sos is not None:
            filtered, self.filter_zi = sosfilt(self.filter_sos, raw_audio, zi=self.filter_zi)
            self.audio_queue.append(filtered)
        else:
            self.audio_queue.append(raw_audio)

    def run(self):
        stream = sd.InputStream(device=self.device_id, samplerate=SAMPLING_RATE, channels=1, blocksize=CHUNK_SIZE, callback=self.audio_callback)
        stream.start()
        freq_bins = np.fft.rfftfreq(CHUNK_SIZE, 1 / SAMPLING_RATE)
        window = np.hanning(CHUNK_SIZE)
        while self.is_running:
            if not self.audio_queue:
                self.msleep(20)
                continue
            data = self.audio_queue.pop(0)
            if len(data) == CHUNK_SIZE:
                self.last_chunk = data
                data_windowed = data * window
                db_magnitude = 20 * np.log10(np.abs(np.fft.rfft(data_windowed)) + 1e-9)
                if self.is_recording:
                    self.recording_buffer_fft.append(db_magnitude)
                harmonics_data = []
                if self.fundamental_freq > 0:
                    for i in range(1, 11):
                        harmonic_freq = self.fundamental_freq * i
                        if harmonic_freq < SAMPLING_RATE / 2:
                            idx = np.argmin(np.abs(freq_bins - harmonic_freq))
                            harmonics_data.append((freq_bins[idx], db_magnitude[idx]))
                self.new_data.emit(freq_bins, db_magnitude, harmonics_data)
            self.msleep(20)
        stream.stop()
        stream.close()

    def start_recording(self):
        self.recording_buffer_fft = []
        self.recording_buffer_raw = []
        self.is_recording = True

    def stop_and_process_recording(self, session_name, label, sound_type, amplitude):
        self.is_recording = False
        if self.recording_buffer_fft and self.recording_buffer_raw:
            averaged_spectrum = np.mean(self.recording_buffer_fft, axis=0)
            full_raw_recording = np.concatenate(self.recording_buffer_raw)
            self.recording_finished.emit(session_name, label, sound_type, amplitude, averaged_spectrum, full_raw_recording)
            self.recording_buffer_fft = []
            self.recording_buffer_raw = []

    def set_fundamental_frequency(self, freq):
        self.fundamental_freq = freq

    def stop(self):
        self.is_running = False

    def set_reference(self, ref_waveform):
        if ref_waveform is not None and len(ref_waveform) > 0:
            self.reference_waveform = ref_waveform.astype(np.float32).copy()

    def set_use_reference(self, use_ref: bool):
        self.use_reference = use_ref

# =============================================================================
# MODULAR ANALYZER DIALOG
# =============================================================================
class ModularAnalyzerDialog(QDialog):
    def __init__(self, recorded_signatures, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Modular Analyzer")
        self.setGeometry(150, 150, 1600, 900)
        self.all_data = recorded_signatures
        self.current_results = {}
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.Shape.StyledPanel)
        control_panel.setMaximumWidth(350)
        control_layout = QVBoxLayout(control_panel)
        
        control_layout.addWidget(QLabel("<b>1. Select Inputs:</b>"))
        self.input_list = QTreeWidget()
        self.input_list.setHeaderHidden(True)
        self.input_list.itemChanged.connect(self.on_analyzer_item_changed)
        
        # Group recordings by session, then label
        sessions_map = {}
        for uid, data in self.all_data.items():
            session = data.get('session', 'Unassigned Session')
            label = data.get('label', 'Uncategorized')
            sessions_map.setdefault(session, {}).setdefault(label, []).append(data)
            
        # Populate the tree: Session -> Label -> Recording
        for session in sorted(sessions_map.keys()):
            session_item = QTreeWidgetItem(self.input_list, [session])
            session_item.setFlags(session_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            
            for label in sorted(sessions_map[session].keys()):
                label_item = QTreeWidgetItem(session_item, [label])
                label_item.setFlags(label_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                label_item.setCheckState(0, Qt.CheckState.Unchecked)
                
                for data in sorted(sessions_map[session][label], key=lambda x: x['base_filename']):
                    child_item = QTreeWidgetItem(label_item, [data['base_filename']])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, data['wav_path']) # Store UID (wav_path)
                    child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child_item.setCheckState(0, Qt.CheckState.Unchecked)
        
        self.input_list.expandAll() # Start with all groups open
        control_layout.addWidget(self.input_list)

        control_layout.addWidget(QLabel("<b>2. Select Analyzer Module:</b>"))
        self.analyzer_combo = QComboBox()
        self.analyzer_combo.addItems(REGISTERED_ANALYZERS.keys()) # Populated dynamically
        control_layout.addWidget(self.analyzer_combo)

        self.run_btn = QPushButton("▶ Run Analysis")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        self.run_btn.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.run_btn)
        
        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # --- Gallery Tab ---
        gallery_widget = QWidget()
        gallery_layout = QVBoxLayout(gallery_widget)
        self.main_view_container = QVBoxLayout()
        self.current_main_canvas = None
        gallery_layout.addLayout(self.main_view_container, stretch=4)

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setFixedHeight(200)
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_content = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_content)
        self.thumb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.thumb_scroll.setWidget(self.thumb_content)
        gallery_layout.addWidget(self.thumb_scroll, stretch=1)

        # --- Details Tab ---
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_container = QWidget()
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.details_scroll.setWidget(self.details_container)

        self.tabs.addTab(gallery_widget, "Gallery")
        self.tabs.addTab(self.details_scroll, "Details")
        viewer_layout.addWidget(self.tabs)

        main_layout.addWidget(viewer_panel)

    def run_analysis(self):
        selected_ids = []
        root = self.input_list.invisibleRootItem()
        session_count = root.childCount()
        
        for i in range(session_count):
            session_item = root.child(i)
            label_count = session_item.childCount()
            for j in range(label_count):
                label_item = session_item.child(j)
                if label_item.checkState(0) == Qt.CheckState.Checked:
                    for k in range(label_item.childCount()):
                        rec_item = label_item.child(k)
                        selected_ids.append(rec_item.data(0, Qt.ItemDataRole.UserRole))
                    continue
                recording_count = label_item.childCount()
                for k in range(recording_count):
                    recording_item = label_item.child(k)
                    if recording_item.checkState(0) == Qt.CheckState.Checked:
                        selected_ids.append(recording_item.data(0, Qt.ItemDataRole.UserRole))
        
        if not selected_ids:
             QMessageBox.warning(self, "Error", "Please select at least one input recording.")
             return

        analyzer_name = self.analyzer_combo.currentText()
        analyzer_func = REGISTERED_ANALYZERS.get(analyzer_name)

        if not analyzer_func: return

        try:
            self.run_btn.setText("Running..."); self.run_btn.setEnabled(False); QApplication.processEvents()
            # Pass config globals to analyzer
            self.last_selected_ids = list(selected_ids)
            self.current_results = analyzer_func(self.all_data, selected_ids, CHUNK_SIZE, SAMPLING_RATE)
            self.update_viewer()
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", f"{e.__class__.__name__}: {e}")
        finally:
            self.run_btn.setText("▶ Run Analysis"); self.run_btn.setEnabled(True)

    def update_viewer(self):
        if self.current_main_canvas:
            self.main_view_container.removeWidget(self.current_main_canvas)
            self.current_main_canvas.close()
            self.current_main_canvas = None
        
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # Clear details layout
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not self.current_results: return

        first_key = None
        for name, fig in self.current_results.items():
            if first_key is None: first_key = name
            
            btn = QPushButton()
            btn.setFixedSize(240, 160)
            btn_layout = QVBoxLayout(btn)
            
            thumb_canvas = MplCanvas(figure=fig, width=3, height=2, dpi=50)
            thumb_canvas.draw()
            thumb_canvas.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) 
            
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; bg-color: rgba(255,255,255,150);")
            
            btn_layout.addWidget(thumb_canvas)
            btn_layout.addWidget(lbl)
            
            btn.clicked.connect(lambda checked, v=name: self.set_main_view(v))
            self.thumb_layout.addWidget(btn)

            # Details view: full-size stacked canvases with titles
            detail_box = QGroupBox(name)
            detail_box_layout = QVBoxLayout()
            detail_canvas = MplCanvas(figure=fig)
            detail_canvas.draw()
            detail_box_layout.addWidget(detail_canvas)
            # Add a small info label
            meta_text = f"Generated from {len(self.last_selected_ids) if hasattr(self, 'last_selected_ids') else 'N/A'} selected recordings."
            detail_box_layout.addWidget(QLabel(meta_text))
            detail_box.setLayout(detail_box_layout)
            self.details_layout.addWidget(detail_box)

        if first_key:
            self.set_main_view(first_key)

    def set_main_view(self, view_name):
        if self.current_main_canvas:
             self.main_view_container.removeWidget(self.current_main_canvas)
             self.current_main_canvas.close()

        fig = self.current_results.get(view_name)
        if fig:
            self.current_main_canvas = MplCanvas(figure=fig)
            self.main_view_container.addWidget(self.current_main_canvas)
            self.current_main_canvas.draw()

    def on_analyzer_item_changed(self, item, column):
        # If a label item is toggled, propagate to its children
        if item.childCount() > 0:
            state = item.checkState(0)
            for idx in range(item.childCount()):
                child = item.child(idx)
                child.setCheckState(0, state)

# =============================================================================
# TRAINING SELECTION DIALOG
# =============================================================================
class TrainingSelectionDialog(QDialog):
    def __init__(self, recorded_signatures, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selective Training")
        self.setGeometry(200, 200, 600, 500)
        self.all_data = recorded_signatures
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select recordings to include in training (Session → Label → Recording):"))

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self.on_item_changed)

        sessions_map = {}
        for uid, data in self.all_data.items():
            session = data.get('session', 'Unassigned Session')
            label = data.get('label', 'Uncategorized')
            sessions_map.setdefault(session, {}).setdefault(label, []).append(data)

        for session in sorted(sessions_map.keys()):
            session_item = QTreeWidgetItem(self.tree, [session])
            session_item.setFlags(session_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            for label in sorted(sessions_map[session].keys()):
                label_item = QTreeWidgetItem(session_item, [label])
                label_item.setFlags(label_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                label_item.setCheckState(0, Qt.CheckState.Unchecked)
                for data in sorted(sessions_map[session][label], key=lambda x: x['base_filename']):
                    child_item = QTreeWidgetItem(label_item, [data['base_filename']])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, data['wav_path'])
                    child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child_item.setCheckState(0, Qt.CheckState.Unchecked)

        self.tree.expandAll()
        layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Train with Selection")
        self.run_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def selected_ids(self):
        selected = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            session_item = root.child(i)
            for j in range(session_item.childCount()):
                label_item = session_item.child(j)
                if label_item.checkState(0) == Qt.CheckState.Checked:
                    for k in range(label_item.childCount()):
                        rec_item = label_item.child(k)
                        selected.append(rec_item.data(0, Qt.ItemDataRole.UserRole))
                    continue
                for k in range(label_item.childCount()):
                    rec_item = label_item.child(k)
                    if rec_item.checkState(0) == Qt.CheckState.Checked:
                        selected.append(rec_item.data(0, Qt.ItemDataRole.UserRole))
        return selected

    def on_item_changed(self, item, column):
        if item.childCount() > 0:
            state = item.checkState(0)
            for idx in range(item.childCount()):
                child = item.child(idx)
                child.setCheckState(0, state)

# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoundSmith - Acoustic Lab")
        self.setGeometry(100, 100, 1400, 850)
        
        # Load persistent labels (now a dict)
        self.all_labels_map = load_labels()
        
        self.recorded_signatures = {} # Init empty, will be populated by scan
        self.display_waveform = None  # waveform to show in UI (pending or last saved)
        self.display_waveform_dirty = False
        self.load_loopback_reference()
        
        self.latest_fft_data = None
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 5
        
        # New timer for recording countdown
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(1000)
        self.recording_timer.timeout.connect(self.update_recording_countdown)
        self.recording_countdown_value = 0
        
        self.current_recording_mode = None
        self.pending_recording = None # For Save/Discard logic
        
        if not self.select_devices():
            sys.exit()

        self.emitter_thread = EmitterThread(self.output_device_id)
        self.analyzer_thread = AnalyzerThread(self.input_device_id)
        self.analyzer_thread.new_data.connect(self.update_data)
        self.analyzer_thread.recording_finished.connect(self.handle_finished_recording)
        self.corruption_monitor = CorruptionMonitor(self.analyzer_thread)
        self.corruption_monitor.status_update.connect(self.on_corruption_status)
        self.loopback_thread = None

        self.initUI() # Creates UI elements
        self.refresh_device_lists() # Populate device combos based on current hardware and selection
        
        self.scan_and_load_recordings() # Populates UI lists from disk
        
        self.emitter_thread.start()
        self.analyzer_thread.start()
        self.corruption_monitor.start()

    def select_devices(self):
        devices = sd.query_devices()
        input_devices = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_input_channels'] > 0}
        output_devices = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_output_channels'] > 0}
        if not input_devices or not output_devices:
            QMessageBox.critical(self, "Error", "No suitable audio devices found.")
            return False
        input_name, ok1 = QInputDialog.getItem(self, "Select Input", "Select a device:", input_devices.keys(), 0, False)
        output_name, ok2 = QInputDialog.getItem(self, "Select Output", "Select a device:", output_devices.keys(), 0, False)
        if ok1 and ok2:
            self.input_device_id = input_devices[input_name]
            self.output_device_id = output_devices[output_name]
            # Adjust sampling rate to a value supported by both devices
            try:
                in_dev = sd.query_devices(self.input_device_id)
                out_dev = sd.query_devices(self.output_device_id)
                in_sr = in_dev.get("default_samplerate", 0) or 0
                out_sr = out_dev.get("default_samplerate", 0) or 0
                candidate_sr = int(min(in_sr, out_sr)) if in_sr and out_sr else int(max(in_sr, out_sr))
                if candidate_sr > 0:
                    global SAMPLING_RATE
                    SAMPLING_RATE = candidate_sr
                    print(f"Using sampling rate {SAMPLING_RATE} Hz (based on selected devices).")
            except Exception as e:
                print(f"Could not adjust sampling rate, keeping default {SAMPLING_RATE} Hz: {e}")
            return True
        return False

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        plot_and_harmonics_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=8, height=6)
        self.plot_ref = None
        harmonics_box = QGroupBox("🎶 Harmonics Display")
        harmonics_layout = QVBoxLayout()
        self.harmonics_table = QTableWidget(10, 2)
        self.harmonics_table.setHorizontalHeaderLabels(["Frequency (Hz)", "Amplitude (dB)"])
        self.harmonics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        harmonics_layout.addWidget(self.harmonics_table)

        # Waveform comparison: live chunk vs last saved recording
        self.waveform_compare_canvas = MplCanvas(self, width=5, height=2)
        self.clean_signal_canvas = MplCanvas(self, width=5, height=2)
        self.show_live_signal_chk = QCheckBox("Show live signal (loopback-removed)")
        self.show_live_signal_chk.setChecked(False)  # default off for performance
        harmonics_layout.addWidget(self.show_live_signal_chk)
        harmonics_layout.addWidget(self.clean_signal_canvas)
        harmonics_layout.addWidget(QLabel("Recording Preview"))
        harmonics_layout.addWidget(self.waveform_compare_canvas)

        harmonics_box.setLayout(harmonics_layout)
        plot_and_harmonics_layout.addWidget(self.canvas)
        plot_and_harmonics_layout.addWidget(harmonics_box)
        main_layout.addLayout(plot_and_harmonics_layout)

        controls_layout = QVBoxLayout()

        # --- Device Selection ---
        devices_box = QGroupBox("🎧 Audio Devices")
        devices_layout = QVBoxLayout()
        input_dev_layout = QHBoxLayout()
        input_dev_layout.addWidget(QLabel("Input:"))
        self.input_device_combo = QComboBox()
        input_dev_layout.addWidget(self.input_device_combo)
        devices_layout.addLayout(input_dev_layout)

        output_dev_layout = QHBoxLayout()
        output_dev_layout.addWidget(QLabel("Output:"))
        self.output_device_combo = QComboBox()
        output_dev_layout.addWidget(self.output_device_combo)
        devices_layout.addLayout(output_dev_layout)

        dev_btn_layout = QHBoxLayout()
        self.refresh_devices_btn = QPushButton("Refresh Devices")
        self.refresh_devices_btn.clicked.connect(self.refresh_device_lists)
        self.apply_devices_btn = QPushButton("Apply Devices")
        self.apply_devices_btn.clicked.connect(self.apply_device_selection)
        self.probe_rates_btn = QPushButton("Probe Sample Rates")
        self.probe_rates_btn.clicked.connect(self.probe_sampling_rates)
        dev_btn_layout.addWidget(self.refresh_devices_btn)
        dev_btn_layout.addWidget(self.apply_devices_btn)
        dev_btn_layout.addWidget(self.probe_rates_btn)
        devices_layout.addLayout(dev_btn_layout)
        devices_box.setLayout(devices_layout)
        controls_layout.addWidget(devices_box)
        
        output_box = QGroupBox("🔊 Output Control")
        output_layout = QVBoxLayout()

        play_stop_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ Play")
        self.play_button.setFixedHeight(40)
        self.play_button.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.play_button.clicked.connect(self.on_play)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setFixedHeight(40)
        self.stop_button.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)

        play_stop_layout.addWidget(self.play_button)
        play_stop_layout.addWidget(self.stop_button)
        output_layout.addLayout(play_stop_layout)

        sound_type_layout = QHBoxLayout()
        sound_type_layout.addWidget(QLabel("Sound Type:"))
        self.sound_type_combo = QComboBox()
        self.sound_type_combo.addItems(["Sine Wave", "Chirp", "5 Chirps", "High Energy Chirp"])
        self.sound_type_combo.currentIndexChanged.connect(self.sound_type_changed)
        sound_type_layout.addWidget(self.sound_type_combo)
        output_layout.addLayout(sound_type_layout)

        loopback_layout = QHBoxLayout()
        self.capture_loopback_btn = QPushButton("Capture Loopback")
        self.capture_loopback_btn.clicked.connect(self.capture_loopback_reference)
        self.capture_loopback_full_btn = QPushButton("Record Loopback (12s)")
        self.capture_loopback_full_btn.clicked.connect(lambda: self.capture_full_loopback(12))
        self.capture_loopback_full6_btn = QPushButton("Record Loopback (6s)")
        self.capture_loopback_full6_btn.clicked.connect(lambda: self.capture_full_loopback(6))
        self.apply_loopback_chk = QCheckBox("Subtract loopback")
        self.apply_loopback_chk.stateChanged.connect(self.toggle_loopback_subtraction)
        self.apply_loopback_chk.setEnabled(False)
        self.apply_loopback_chk.setToolTip("Loopback subtraction disabled; raw signal is used.")
        loopback_layout.addWidget(self.capture_loopback_btn)
        loopback_layout.addWidget(self.capture_loopback_full_btn)
        loopback_layout.addWidget(self.capture_loopback_full6_btn)
        loopback_layout.addWidget(self.apply_loopback_chk)
        output_layout.addLayout(loopback_layout)

        self.freq_label = QLabel(f"Frequency: {self.emitter_thread.frequency:.0f} Hz")
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(20, 20000)
        self.freq_slider.setValue(int(self.emitter_thread.frequency))
        self.freq_slider.valueChanged.connect(self.update_output_params)

        self.amp_label = QLabel(f"Amplitude: {self.emitter_thread.amplitude*100:.0f}%")
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(int(self.emitter_thread.amplitude * 100))
        self.amp_slider.valueChanged.connect(self.update_output_params)

        output_layout.addWidget(self.freq_label)
        output_layout.addWidget(self.freq_slider)
        output_layout.addWidget(self.amp_label)
        output_layout.addWidget(self.amp_slider)
        output_box.setLayout(output_layout)
        
        record_box = QGroupBox("🎙️ Recording & Analysis")
        record_layout = QVBoxLayout()
        
        # --- NEW: Session Selection ---
        record_layout.addWidget(QLabel("<b>1. Select/Create Session:</b>"))
        self.session_combo = QComboBox()
        self.session_combo.setEditable(True)
        self.session_combo.lineEdit().setPlaceholderText("e.g., Session1, TestRun, ...")
        record_layout.addWidget(self.session_combo)
        record_layout.addSpacing(10) # Add clearance
        # ----------------------------

        # --- NEW: Collapsible Add Label Box ---
        add_label_box = QGroupBox("2. Add New Label Definition")
        add_label_box.setCheckable(True)  # Make it collapsible
        add_label_box.setChecked(False) # Start collapsed
        add_label_layout = QVBoxLayout()
        
        self.add_label_widget = QWidget() # Container for inputs
        add_label_inputs_layout = QHBoxLayout(self.add_label_widget) # Layout for the container
        self.new_label_name_input = QLineEdit()
        self.new_label_name_input.setPlaceholderText("Full Name (e.g., OpenPalm_Horizontal)")
        self.new_label_abbr_input = QLineEdit()
        self.new_label_abbr_input.setPlaceholderText("Abbreviation (e.g., OP_H)")
        self.add_label_btn = QPushButton("Add")
        self.add_label_btn.clicked.connect(self.on_add_label)
        add_label_inputs_layout.addWidget(self.new_label_name_input, stretch=2)
        add_label_inputs_layout.addWidget(self.new_label_abbr_input, stretch=1)
        add_label_inputs_layout.addWidget(self.add_label_btn)
        
        add_label_layout.addWidget(self.add_label_widget)
        add_label_box.setLayout(add_label_layout)
        record_layout.addWidget(add_label_box)
        
        # Connect toggled signal to widget visibility
        add_label_box.toggled.connect(self.add_label_widget.setVisible)
        self.add_label_widget.setVisible(False) # Start hidden
        # ------------------------------------

        record_layout.addWidget(QLabel("<b>3. Select Label for Recording:</b>"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(False) # No longer editable
        self.label_combo.addItems(sorted(self.all_labels_map.keys())) # Populate from loaded map
        
        # --- Recording Buttons ---
        record_buttons_layout = QHBoxLayout()
        self.record_passive_button = QPushButton("Record Passive")
        self.record_passive_button.clicked.connect(self.start_passive_recording)
        self.record_active_button = QPushButton("Record Active (5 Chirps)")
        self.record_active_button.clicked.connect(self.start_active_recording)
        self.record_high_energy_button = QPushButton("Record High-Energy Chirp")
        self.record_high_energy_button.clicked.connect(self.start_high_energy_recording)
        self.record_hec6_button = QPushButton("Record HEC (6s)")
        self.record_hec6_button.clicked.connect(self.start_hec6_recording)
        record_buttons_layout.addWidget(self.record_passive_button)
        record_buttons_layout.addWidget(self.record_active_button)
        record_buttons_layout.addWidget(self.record_high_energy_button)
        record_buttons_layout.addWidget(self.record_hec6_button)
        
        self.record_buttons_widget = QWidget()
        self.record_buttons_widget.setLayout(record_buttons_layout)
        
        # --- Confirmation Buttons ---
        self.confirm_buttons_layout = QHBoxLayout()
        self.save_recording_btn = QPushButton("✅ Save")
        self.save_recording_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.save_recording_btn.clicked.connect(self.on_save_pending_recording)
        self.discard_recording_btn = QPushButton("❌ Discard")
        self.discard_recording_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.discard_recording_btn.clicked.connect(self.on_discard_pending_recording)
        self.confirm_buttons_layout.addWidget(self.save_recording_btn)
        self.confirm_buttons_layout.addWidget(self.discard_recording_btn)
        
        self.confirm_buttons_widget = QWidget()
        self.confirm_buttons_widget.setLayout(self.confirm_buttons_layout)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-style: italic; color: grey;")
        
        self.open_analyzer_btn = QPushButton("📊 Open Modular Analyzer")
        self.open_analyzer_btn.setFixedHeight(40)
        self.open_analyzer_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #2196F3; color: white;")
        self.open_analyzer_btn.clicked.connect(self.open_modular_analyzer)

        self.signature_list = QTreeWidget()
        self.signature_list.setHeaderHidden(True)
        # List is now populated by scan_and_load_recordings() after initUI

        record_layout.addWidget(self.label_combo)
        record_layout.addWidget(self.record_buttons_widget) # Add widget
        record_layout.addWidget(self.confirm_buttons_widget) # Add widget
        record_layout.addWidget(self.status_label)
        record_layout.addWidget(self.open_analyzer_btn)
        record_layout.addWidget(QLabel("Recorded Signatures:"))
        record_layout.addWidget(self.signature_list)
        record_box.setLayout(record_layout)
        
        train_box = QGroupBox("🧠 Train Model")
        train_tabs = QTabWidget()

        # Controls tab
        train_controls_widget = QWidget()
        train_layout = QVBoxLayout(train_controls_widget)
        train_layout.addWidget(QLabel("<i>Select data above, then choose model type:</i>"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(REGISTERED_MODELS.keys()) # Populated dynamically
        train_layout.addWidget(self.model_type_combo)
        train_btn = QPushButton("Train All Data")
        train_btn.clicked.connect(self.run_model_training)
        train_layout.addWidget(train_btn)
        train_selected_btn = QPushButton("Train with Selection...")
        train_selected_btn.clicked.connect(self.open_training_selection)
        train_layout.addWidget(train_selected_btn)
        train_layout.addStretch()

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.training_output = QPlainTextEdit()
        self.training_output.setReadOnly(True)
        self.training_output.setMinimumHeight(180)
        results_layout.addWidget(self.training_output)

        train_tabs.addTab(train_controls_widget, "Controls")
        train_tabs.addTab(results_widget, "Results")
        box_layout = QVBoxLayout()
        box_layout.addWidget(train_tabs)
        train_box.setLayout(box_layout)

        controls_layout.addWidget(output_box)
        controls_layout.addWidget(record_box)
        controls_layout.addWidget(train_box)
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.confirm_buttons_widget.hide() # Hide confirm buttons on start

    def on_add_label(self):
        full_name = self.new_label_name_input.text().strip()
        abbr = self.new_label_abbr_input.text().strip()

        if not full_name or not abbr:
            QMessageBox.warning(self, "Input Error", "Both Full Name and Abbreviation are required.")
            return

        if full_name in self.all_labels_map:
            QMessageBox.warning(self, "Input Error", f"The label '{full_name}' already exists.")
            return
            
        # Check if abbreviation is used by another full name
        for name, existing_abbr in self.all_labels_map.items():
            if existing_abbr == abbr:
                 QMessageBox.warning(self, "Input Error", f"Abbreviation '{abbr}' is already used by '{name}'.")
                 return

        # Add new label
        self.all_labels_map[full_name] = abbr
        save_labels(self.all_labels_map)
        
        # Update UI
        self.label_combo.addItem(full_name)
        self.label_combo.setCurrentText(full_name)
        
        # Clear inputs
        self.new_label_name_input.clear()
        self.new_label_abbr_input.clear()
        
        self.status_label.setText(f"Status: Added new label '{full_name}'")

    def scan_and_load_recordings(self):
        """Scans the RECORDINGS_DIR for all _data.json files and loads them."""
        print(f"Scanning for recordings in {RECORDINGS_DIR}...")
        
        # Clear existing session data
        self.recorded_signatures.clear()
        self.signature_list.clear() # Clear UI tree
        
        # --- NEW: Scan for sessions ---
        all_sessions = []
        try:
            all_sessions = [d for d in os.listdir(RECORDINGS_DIR) if os.path.isdir(os.path.join(RECORDINGS_DIR, d))]
        except FileNotFoundError:
            print(f"Recordings directory not found at {RECORDINGS_DIR}. Will be created.")
        except Exception as e:
            print(f"Error scanning for sessions: {e}")
        
        self.session_combo.addItems(sorted(all_sessions))
        self.session_combo.setCurrentText("")
        # ----------------------------

        data_files = glob.glob(os.path.join(RECORDINGS_DIR, "*", "**", "*_data.json"), recursive=True)
        
        labels_updated = False
        loaded_count = 0
        
        for data_path in data_files:
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                wav_path = data.get('wav_path')
                base_filename = data.get('base_filename')
                fixed_data = False

                # Attempt to repair broken absolute paths by anchoring to the data file location
                if (not wav_path or not os.path.exists(wav_path)) and base_filename:
                    base_dir = os.path.dirname(data_path)
                    candidate_wav = os.path.join(base_dir, f"{base_filename}.wav")
                    candidate_spec = os.path.join(base_dir, f"{base_filename}_spectrogram.png")
                    if os.path.exists(candidate_wav):
                        wav_path = candidate_wav
                        data["wav_path"] = wav_path
                        if os.path.exists(candidate_spec):
                            data["spec_path"] = candidate_spec
                        fixed_data = True
                
                # Check if file exists and has all required data
                if wav_path and base_filename and os.path.exists(wav_path):
                    self.recorded_signatures[wav_path] = data
                    
                    label_full_name = data.get('label')
                    if label_full_name:
                        # Auto-import labels from old sessions
                        if label_full_name not in self.all_labels_map:
                            print(f"Found orphaned label: '{label_full_name}'. Auto-importing.")
                            # Use label itself as abbreviation as fallback
                            self.all_labels_map[label_full_name] = label_full_name.replace(" ", "_").upper()
                            labels_updated = True
                    
                    loaded_count += 1

                    # Persist repaired paths so future loads are fast
                    if fixed_data:
                        try:
                            with open(data_path, 'w') as f:
                                json.dump(data, f, indent=4)
                        except Exception as e:
                            print(f"Could not persist repaired paths for {data_path}: {e}")
                else:
                    print(f"Skipping incomplete/missing data from {data_path}")
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
        
        print(f"Loaded {loaded_count} recordings.")
        
        # --- Populate Tree ---
        labels_map = {}
        for uid, data in self.recorded_signatures.items():
            label = data.get('label', 'Uncategorized')
            if label not in labels_map:
                labels_map[label] = []
            labels_map[label].append(data)
            
        for label in sorted(labels_map.keys()):
            label_item = QTreeWidgetItem(self.signature_list, [label])
            label_item.setFlags(label_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            
            for data in sorted(labels_map[label], key=lambda x: x['base_filename']):
                child_item = QTreeWidgetItem(label_item, [data['base_filename']])
                child_item.setData(0, Qt.ItemDataRole.UserRole, data['wav_path'])
        
        self.signature_list.expandAll()
        # ------------------------
        
        # Update the master label list and save if new labels were found
        if labels_updated:
            save_labels(self.all_labels_map)
            
        # Update combo box from the final map
        self.label_combo.clear()
        self.label_combo.addItems(sorted(self.all_labels_map.keys()))
        self.label_combo.setCurrentText("")

    def on_play(self):
        self.emitter_thread.set_muted(False)
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def on_stop(self):
        self.emitter_thread.set_muted(True)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def start_passive_recording(self):
        self._start_recording_flow(mode="passive")

    def start_active_recording(self):
        self._start_recording_flow(mode="active")

    def start_high_energy_recording(self):
        self._start_recording_flow(mode="high_energy")

    def start_hec6_recording(self):
        self._start_recording_flow(mode="hec6")

    def _start_recording_flow(self, mode):
        session_name = self.session_combo.currentText().strip()
        label_full_name = self.label_combo.currentText().strip()
        
        if not session_name:
            QMessageBox.warning(self, "Input Error", "Please select or create a Session.")
            return
            
        if not label_full_name:
            QMessageBox.warning(self, "Input Error", "Please select a Label before recording.")
            return
            
        self.current_session_name = session_name
        self.current_label_name = label_full_name
        self.current_recording_mode = mode
        self.was_playing_before_rec = not self.emitter_thread.is_muted

        self.record_buttons_widget.setEnabled(False)
        self.confirm_buttons_widget.hide() # Ensure confirm is hidden
        
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.countdown_value = 5
        self.status_label.setText(f"Recording in {self.countdown_value}...")
        self.countdown_timer.start()

    def update_countdown(self):
        self.countdown_value -= 1
        self.status_label.setText(f"Recording in {self.countdown_value}...")
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            self.begin_actual_recording()

    def begin_actual_recording(self):
        if self.current_recording_mode == "active":
            duration = RECORDING_DURATION_ACTIVE_S
            self.current_sound_type = "5 Chirps"
            self.current_rec_amplitude = self.emitter_thread.amplitude
            self.emitter_thread.set_sound_type("5 Chirps")
            self.emitter_thread.set_muted(False)
        elif self.current_recording_mode == "high_energy":
            duration = RECORDING_DURATION_HIGH_ENERGY_S
            self.current_sound_type = "High Energy Chirp"
            self.current_rec_amplitude = self.emitter_thread.amplitude
            self.emitter_thread.set_sound_type("High Energy Chirp")
            self.emitter_thread.set_muted(False)
        elif self.current_recording_mode == "hec6":
            duration = RECORDING_DURATION_H5_HEC_S
            self.current_sound_type = "High Energy Chirp"
            self.current_rec_amplitude = self.emitter_thread.amplitude
            self.emitter_thread.set_sound_type("High Energy Chirp")
            self.emitter_thread.set_muted(False)
        else:
            duration = RECORDING_DURATION_PASSIVE_S
            self.current_sound_type = "Passive"
            self.current_rec_amplitude = 0.0
            self.emitter_thread.set_muted(True)

        # Start recording countdown
        self.recording_countdown_value = duration
        self.status_label.setText(f"Recording... {self.recording_countdown_value}s left")
        self.recording_timer.start()
        
        self.analyzer_thread.start_recording()
        # This master timer still controls the actual stop
        QTimer.singleShot(duration * 1000, self.finish_recording)

    def finish_recording(self):
        self.recording_timer.stop() # Stop the countdown UI timer
        self.status_label.setText("Processing...")
        # Stop output playback after capture to avoid continuous chirps
        self.emitter_thread.set_muted(True)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.analyzer_thread.stop_and_process_recording(self.current_session_name, self.current_label_name, self.current_sound_type, self.current_rec_amplitude)

    def get_filepaths(self, session_name, full_label_name, label_abbr, sound_type, amplitude):
        type_code = SOUND_TYPE_CODES.get(sound_type, "UNK")
        amp_str = f"{int(amplitude * 100)}p"
        
        # Folder name uses abbreviation
        label_folder_name = f"{label_abbr}_{amp_str}_{type_code}"
        # New session path
        session_path = os.path.join(RECORDINGS_DIR, session_name)
        full_folder_path = os.path.join(session_path, label_folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        counter = 1
        while True:
            # File name uses full label name
            candidate_base = f"{full_label_name}_{amp_str}_{type_code}_{counter:02d}"
            wav_path = os.path.join(full_folder_path, f"{candidate_base}.wav")
            if not os.path.exists(wav_path):
                spec_path = os.path.join(full_folder_path, f"{candidate_base}_spectrogram.png")
                return candidate_base, wav_path, spec_path
            counter += 1

    # --- NEW: Slot for recording countdown ---
    def update_recording_countdown(self):
        self.recording_countdown_value -= 1
        if self.recording_countdown_value >= 0:
            self.status_label.setText(f"Recording... {self.recording_countdown_value}s left")
        else:
            self.recording_timer.stop()
    # ---------------------------------------

    def handle_finished_recording(self, session_name, label_full_name, sound_type, amplitude, averaged_spectrum, raw_audio):
        # Store data temporarily
        self.pending_recording = {
            "session": session_name,
            "label": label_full_name, # This is the Full Name
            "sound_type": sound_type,
            "amplitude": amplitude,
            "averaged_spectrum": averaged_spectrum,
            "raw_audio": raw_audio
        }
        # Prepare waveform display before save
        self.display_waveform = raw_audio
        self.display_waveform_dirty = True
        
        # Show confirmation buttons
        self.status_label.setText(f"Confirm: Save '{label_full_name}'?")
        self.record_buttons_widget.hide()
        self.confirm_buttons_widget.show()

    def on_save_pending_recording(self):
        if not self.pending_recording:
            return

        # Get data from temp storage
        data = self.pending_recording
        session_name = data["session"]
        full_label_name = data["label"]
        sound_type = data["sound_type"]
        amplitude = data["amplitude"]
        
        # Get abbreviation from map
        label_abbr = self.all_labels_map.get(full_label_name)
        if not label_abbr:
            print(f"Warning: No abbreviation found for '{full_label_name}'. Using sanitized name as folder.")
            label_abbr = full_label_name.replace(" ", "_").upper()

        base_filename, wav_path, spec_path = self.get_filepaths(session_name, full_label_name, label_abbr, sound_type, amplitude)
        
        # Create metadata JSON path
        data_path = wav_path.replace(".wav", "_data.json")

        data_for_storage = {
            'base_filename': base_filename,
            'session': session_name, # Store session name
            'label': full_label_name, # Store full name in JSON
            'sound_type': sound_type,
            'amplitude': amplitude,
            'spectral_data': data["averaged_spectrum"].tolist(),
            'wav_path': wav_path,
            'spec_path': spec_path
        }
        
        # Save all files
        try:
            write_wav(wav_path, SAMPLING_RATE, data["raw_audio"])
            self.save_spectrogram_image(data["raw_audio"], spec_path)
            with open(data_path, 'w') as f:
                json.dump(data_for_storage, f, indent=4)
        except Exception as e:
            print(f"Failed to save recording files: {e}")
            self.status_label.setText(f"Error: Failed to save files for {base_filename}")
            self._restore_ui_state()
            self.pending_recording = None
            return

        # Add to in-memory session
        self.recorded_signatures[wav_path] = data_for_storage
        self.display_waveform = data["raw_audio"]
        self.display_waveform_dirty = True
        
        # --- Add to Tree ---
        parent_item = None
        matches = self.signature_list.findItems(full_label_name, Qt.MatchFlag.MatchFixedString, 0)
        if matches:
            parent_item = matches[0]
        else:
            parent_item = QTreeWidgetItem(self.signature_list, [full_label_name])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.signature_list.sortItems(0, Qt.SortOrder.AscendingOrder)

        child_item = QTreeWidgetItem(parent_item, [base_filename])
        child_item.setData(0, Qt.ItemDataRole.UserRole, wav_path)
        parent_item.setExpanded(True)
        # ------------------------

        # If it was a new session, add it to the combo
        if self.session_combo.findText(session_name) == -1:
            self.session_combo.addItem(session_name)
            self.session_combo.setCurrentText(session_name)

        self.status_label.setText(f"Saved: {base_filename}")
        self._restore_ui_state()
        self.pending_recording = None

    def on_discard_pending_recording(self):
        self.pending_recording = None
        self.display_waveform = None
        self.display_waveform_dirty = False
        self.status_label.setText("Status: Recording discarded.")
        self._restore_ui_state()

    def capture_loopback_reference(self):
        chunk = getattr(self.analyzer_thread, "last_chunk", None)
        if chunk is None:
            QMessageBox.warning(self, "Loopback", "No audio chunk available to capture. Start stream first.")
            return
        self.analyzer_thread.set_reference(chunk)
        # Persist to disk
        try:
            np.save(LOOPBACK_FILE, chunk.astype(np.float32))
        except Exception as e:
            print(f"Failed to save loopback reference: {e}")
        self.status_label.setText("Status: Loopback reference captured.")

    def capture_full_loopback(self, duration=12):
        if self.loopback_thread and self.loopback_thread.isRunning():
            QMessageBox.information(self, "Loopback", "Loopback capture already running.")
            return
        # Save current output state, then play HEC during capture
        self._prev_sound_type = self.sound_type_combo.currentText()
        self._prev_muted = self.emitter_thread.is_muted
        self._prev_amp = self.emitter_thread.amplitude
        self.emitter_thread.set_sound_type("High Energy Chirp")
        self.emitter_thread.set_muted(False)
        self.capture_loopback_full_btn.setEnabled(False)
        self.capture_loopback_full6_btn.setEnabled(False)
        self.status_label.setText(f"Status: Recording loopback ({duration}s)...")
        self.loopback_thread = LoopbackCaptureThread(self.input_device_id, duration)
        self.loopback_thread.finished_capture.connect(self._on_loopback_captured)
        self.loopback_thread.failed_capture.connect(self._on_loopback_failed)
        self.loopback_thread.start()

    def toggle_loopback_subtraction(self, state):
        # Loopback subtraction is disabled by design; keep checkbox inactive.
        self.apply_loopback_chk.setChecked(False)
        self.status_label.setText("Status: Loopback subtraction is disabled (using raw signal).")

    def load_loopback_reference(self):
        if os.path.exists(LOOPBACK_FILE):
            try:
                ref = np.load(LOOPBACK_FILE)
                self.analyzer_thread.set_reference(ref)
                print(f"Loaded loopback reference from {LOOPBACK_FILE}")
            except Exception as e:
                print(f"Could not load loopback reference: {e}")

    def _on_loopback_captured(self, ref):
        self.capture_loopback_full_btn.setEnabled(True)
        self.capture_loopback_full6_btn.setEnabled(True)
        # Restore output state
        if hasattr(self, "_prev_sound_type"):
            self.emitter_thread.set_sound_type(self._prev_sound_type)
        if hasattr(self, "_prev_muted"):
            self.emitter_thread.set_muted(self._prev_muted)
        self.analyzer_thread.set_reference(ref)
        try:
            np.save(LOOPBACK_FILE, ref.astype(np.float32))
        except Exception as e:
            print(f"Failed to save loopback reference: {e}")
        self.status_label.setText("Status: Full loopback (12s) captured.")
        self.display_waveform = ref
        self.display_waveform_dirty = True

    def _on_loopback_failed(self, msg):
        self.capture_loopback_full_btn.setEnabled(True)
        self.capture_loopback_full6_btn.setEnabled(True)
        if hasattr(self, "_prev_sound_type"):
            self.emitter_thread.set_sound_type(self._prev_sound_type)
        if hasattr(self, "_prev_muted"):
            self.emitter_thread.set_muted(self._prev_muted)
        self.status_label.setText(f"Status: Loopback capture failed: {msg}")

    def _restore_ui_state(self):
        # Show record buttons, hide confirm buttons
        self.confirm_buttons_widget.hide()
        self.record_buttons_widget.show()
        self.record_buttons_widget.setEnabled(True)
        self.current_recording_mode = None
        
        # Restore play/stop state
        if self.was_playing_before_rec:
            self.on_play()
        else:
            self.on_stop()

    def save_spectrogram_image(self, raw_audio, path):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.specgram(raw_audio, Fs=SAMPLING_RATE, NFFT=1024, cmap='viridis')
        ax.axis('off')
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def open_modular_analyzer(self):
        dialog = ModularAnalyzerDialog(self.recorded_signatures, self)
        dialog.exec()

    def run_model_training(self):
        model_name = self.model_type_combo.currentText()
        model_func = REGISTERED_MODELS.get(model_name)
        
        if not model_func:
            QMessageBox.critical(self, "Error", f"Could not find model function for '{model_name}'")
            return
            
        if not self.recorded_signatures:
            QMessageBox.information(self, "No Data", "Please record or load some data before training.")
            return
            
        try:
            # Pass all recorded data to the model function
            self._run_model_with_logging(model_func, self.recorded_signatures, model_name)
            QMessageBox.information(self, "Training", f"Ran training for '{model_name}'. See console for output.")
        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"{e.__class__.__name__}: {e}")
            self._append_training_log(f"[ERROR] {model_name}: {e}")

    def _append_training_log(self, text):
        if hasattr(self, "training_output") and self.training_output is not None:
            self.training_output.appendPlainText(text)

    def _run_model_with_logging(self, model_func, data, model_name):
        buf = io.StringIO()
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            model_func(data)
        output = buf.getvalue()
        self._append_training_log(f"[{timestamp}] {model_name}\n{output}\n---")

    # --- Device management helpers ---
    def refresh_device_lists(self):
        devices = sd.query_devices()
        self.input_devices_map = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_input_channels'] > 0}
        self.output_devices_map = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_output_channels'] > 0}

        self.input_device_combo.blockSignals(True)
        self.output_device_combo.blockSignals(True)
        self.input_device_combo.clear()
        self.output_device_combo.clear()
        self.input_device_combo.addItems(self.input_devices_map.keys())
        self.output_device_combo.addItems(self.output_devices_map.keys())

        # Set current selections to active device ids if present
        def set_current(combo, mapping, dev_id):
            for text, idx in mapping.items():
                if idx == dev_id:
                    combo.setCurrentText(text)
                    return
        if hasattr(self, "input_device_id"):
            set_current(self.input_device_combo, self.input_devices_map, self.input_device_id)
        if hasattr(self, "output_device_id"):
            set_current(self.output_device_combo, self.output_devices_map, self.output_device_id)
        self.input_device_combo.blockSignals(False)
        self.output_device_combo.blockSignals(False)

    def apply_device_selection(self):
        if getattr(self.analyzer_thread, "is_recording", False):
            QMessageBox.warning(self, "Busy", "Stop recording before changing devices.")
            return

        in_text = self.input_device_combo.currentText()
        out_text = self.output_device_combo.currentText()
        if in_text not in getattr(self, "input_devices_map", {}) or out_text not in getattr(self, "output_devices_map", {}):
            QMessageBox.warning(self, "Selection Error", "Please select valid input/output devices.")
            return

        new_in = self.input_devices_map[in_text]
        new_out = self.output_devices_map[out_text]
        if new_in == self.input_device_id and new_out == self.output_device_id:
            self.status_label.setText("Status: Devices unchanged.")
            return

        # Find a mutually supported sample rate before stopping threads
        supported_rate = self.pick_supported_rate(new_in, new_out)
        if supported_rate is None:
            QMessageBox.warning(self, "Sample Rate", "Could not find a common supported sample rate for the selected devices.")
            return

        # Stop existing threads with timeout to avoid UI freeze
        self.emitter_thread.stop()
        self.analyzer_thread.stop()
        if not self.emitter_thread.wait(2000):
            self.emitter_thread.terminate()
        if not self.analyzer_thread.wait(2000):
            self.analyzer_thread.terminate()

        # Update IDs and recreate threads
        self.input_device_id = new_in
        self.output_device_id = new_out

        # Set agreed sampling rate
        global SAMPLING_RATE
        SAMPLING_RATE = supported_rate
        print(f"Using sampling rate {SAMPLING_RATE} Hz (based on selected devices).")

        self.emitter_thread = EmitterThread(self.output_device_id)
        self.analyzer_thread = AnalyzerThread(self.input_device_id)
        self.analyzer_thread.new_data.connect(self.update_data)
        self.analyzer_thread.recording_finished.connect(self.handle_finished_recording)

        # Restore UI-selected sound parameters
        self.emitter_thread.update_parameters(self.freq_slider.value(), self.amp_slider.value() / 100.0)
        self.emitter_thread.set_sound_type(self.sound_type_combo.currentText())
        if self.stop_button.isEnabled():  # was playing
            self.emitter_thread.set_muted(False)
        else:
            self.emitter_thread.set_muted(True)

        # Start new threads
        self.emitter_thread.start()
        self.analyzer_thread.start()

        # Re-apply fundamental frequency if in sine mode
        if self.sound_type_combo.currentText() == "Sine Wave":
            self.analyzer_thread.set_fundamental_frequency(self.freq_slider.value())
        else:
            self.analyzer_thread.set_fundamental_frequency(0)

        self.status_label.setText(f"Status: Devices applied (in {in_text}, out {out_text})")

    def probe_sampling_rates(self):
        in_text = self.input_device_combo.currentText()
        out_text = self.output_device_combo.currentText()
        if in_text not in getattr(self, "input_devices_map", {}) or out_text not in getattr(self, "output_devices_map", {}):
            QMessageBox.warning(self, "Selection Error", "Please select valid input/output devices.")
            return

        in_id = self.input_devices_map[in_text]
        out_id = self.output_devices_map[out_text]
        candidate_rates = [44100, 48000, 32000, 22050, 96000, 192000]
        ok_rates = []
        for rate in candidate_rates:
            try:
                sd.check_input_settings(device=in_id, samplerate=rate, channels=1)
                sd.check_output_settings(device=out_id, samplerate=rate, channels=1)
                ok_rates.append(rate)
            except Exception:
                continue

        if ok_rates:
            msg = "Supported sample rates: " + ", ".join(f"{r} Hz" for r in ok_rates)
            QMessageBox.information(self, "Sample Rates", msg)
            self.status_label.setText(f"Status: Available rates: {ok_rates}")
        else:
            QMessageBox.warning(self, "Sample Rates", "No matching sample rates found in test list.")

    def pick_supported_rate(self, in_id, out_id):
        defaults = []
        try:
            defaults.append(int(sd.query_devices(in_id).get("default_samplerate", 0) or 0))
            defaults.append(int(sd.query_devices(out_id).get("default_samplerate", 0) or 0))
        except Exception:
            pass
        candidates = [r for r in defaults if r] + [48000, 44100, 96000, 192000, 32000, 22050]
        seen = set()
        unique_candidates = [r for r in candidates if not (r in seen or seen.add(r))]
        for rate in unique_candidates:
            try:
                sd.check_input_settings(device=in_id, samplerate=rate, channels=1)
                sd.check_output_settings(device=out_id, samplerate=rate, channels=1)
                return rate
            except Exception:
                continue
        return None

    def open_training_selection(self):
        if not self.recorded_signatures:
            QMessageBox.information(self, "No Data", "Please record or load data before training.")
            return

        dialog = TrainingSelectionDialog(self.recorded_signatures, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_ids = dialog.selected_ids()
            if not selected_ids:
                QMessageBox.warning(self, "No Selection", "Please select at least one recording.")
                return

            subset = {uid: self.recorded_signatures[uid] for uid in selected_ids if uid in self.recorded_signatures}
            model_name = self.model_type_combo.currentText()
            model_func = REGISTERED_MODELS.get(model_name)
            if not model_func:
                QMessageBox.critical(self, "Error", f"Could not find model function for '{model_name}'")
                return
            try:
                self._run_model_with_logging(model_func, subset, model_name)
                QMessageBox.information(self, "Training", f"Ran training for '{model_name}' on selected data. See console for output.")
            except Exception as e:
                QMessageBox.critical(self, "Model Error", f"{e.__class__.__name__}: {e}")
                self._append_training_log(f"[ERROR] {model_name} (selection): {e}")

    def sound_type_changed(self):
        sound_type = self.sound_type_combo.currentText()
        self.emitter_thread.set_sound_type(sound_type)
        is_sine_mode = (sound_type == "Sine Wave")
        self.freq_slider.setEnabled(is_sine_mode)
        self.freq_label.setEnabled(is_sine_mode)
        self.harmonics_table.setEnabled(is_sine_mode)
        if is_sine_mode:
            self.analyzer_thread.set_fundamental_frequency(self.freq_slider.value())
        else:
            self.analyzer_thread.set_fundamental_frequency(0)
            self.harmonics_table.clearContents()

    def update_output_params(self):
        freq = self.freq_slider.value()
        amp = self.amp_slider.value() / 100.0
        self.freq_label.setText(f"Frequency: {freq:.0f} Hz")
        self.amp_label.setText(f"Amplitude: {amp*100:.0f}%")
        self.emitter_thread.update_parameters(freq, amp)
        if self.sound_type_combo.currentText() == "Sine Wave":
            self.analyzer_thread.set_fundamental_frequency(freq)

    def update_data(self, freq_bins, db_magnitude, harmonics_data):
        self.latest_fft_data = db_magnitude
        if self.plot_ref is None:
            self.canvas.axes.set_title("Live Audio Spectrum")
            self.canvas.axes.set_xlabel("Frequency (Hz)")
            self.canvas.axes.set_ylabel("Amplitude [dB]")
            self.canvas.axes.set_xlim(20, SAMPLING_RATE / 2)
            self.canvas.axes.set_ylim(-80, 20)
            self.canvas.axes.set_xscale('log')
            self.canvas.axes.grid(True, which="both", ls="--")
            self.plot_ref, = self.canvas.axes.plot(freq_bins, db_magnitude)
        else:
            self.plot_ref.set_ydata(db_magnitude)
        self.canvas.draw()
        
        self.harmonics_table.clearContents()
        if self.harmonics_table.isEnabled():
            for i, (freq, amp) in enumerate(harmonics_data):
                self.harmonics_table.setItem(i, 0, QTableWidgetItem(f"{freq:.2f}"))
                self.harmonics_table.setItem(i, 1, QTableWidgetItem(f"{amp:.2f}"))

        # Current signal plot (loopback-removed)
        if self.show_live_signal_chk.isChecked():
            self.clean_signal_canvas.axes.clear()
            last_chunk = getattr(self.analyzer_thread, "last_chunk", None)
            if last_chunk is not None:
                t = np.arange(len(last_chunk)) / SAMPLING_RATE
                self.clean_signal_canvas.axes.plot(t, last_chunk, color="C1", alpha=0.8)
                self.clean_signal_canvas.axes.set_title("Current Signal (loopback-removed)")
                self.clean_signal_canvas.axes.set_xlabel("Time (s)")
                self.clean_signal_canvas.axes.set_ylabel("Amplitude")
                self.clean_signal_canvas.draw()

        # --- Waveform comparison (only last saved recording to reduce UI load) ---
        self.waveform_compare_canvas.axes.clear()
        if self.display_waveform is not None and self.display_waveform_dirty:
            N = min(len(self.display_waveform), SAMPLING_RATE // 2)
            t_saved = np.arange(N) / SAMPLING_RATE
            self.waveform_compare_canvas.axes.plot(t_saved, self.display_waveform[:N], label="Recording Preview", alpha=0.8)
            self.waveform_compare_canvas.axes.legend()
            self.waveform_compare_canvas.axes.set_title("Recording Preview")
            self.waveform_compare_canvas.axes.set_xlabel("Time (s)")
            self.waveform_compare_canvas.axes.set_ylabel("Amplitude")
            self.waveform_compare_canvas.draw()
            self.display_waveform_dirty = False

    def on_corruption_status(self, message):
        if message:
            self.status_label.setText(f"Status: {message}")

    def closeEvent(self, event):
        self.emitter_thread.stop()
        self.analyzer_thread.stop()
        self.corruption_monitor.stop()
        if self.loopback_thread and self.loopback_thread.isRunning():
            self.loopback_thread.stop()
        self.emitter_thread.wait()
        self.analyzer_thread.wait()
        self.corruption_monitor.wait()
        if self.loopback_thread:
            self.loopback_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # --- Setup and Module Loading ---
    setup_directories_and_files()
    load_modules_from_dir(ANALYZERS_DIR)
    load_modules_from_dir(MODELS_DIR)
    print(f"Loaded Analyzers: {list(REGISTERED_ANALYZERS.keys())}")
    print(f"Loaded Models: {list(REGISTERED_MODELS.keys())}")
    
    window = MainWindow()
    if hasattr(window, 'input_device_id'):
        window.show()
        sys.exit(app.exec())
