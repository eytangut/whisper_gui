#!/usr/bin/env python3
"""Whisper GUI: faster-whisper + Hugging Face model download (PySide6).

    pip install -U pyside6 faster-whisper huggingface_hub
    python whisper_gui.py
"""
import ctypes
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime

# A --windowed PyInstaller build has no stdout/stderr, and some libraries write to them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import faster_whisper
from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # faster-whisper < 1.1
    BatchedInferencePipeline = None
from huggingface_hub import HfApi, snapshot_download
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from tqdm.auto import tqdm

MODELS = {
    "ivrit-ai/whisper-large-v3-turbo-ct2": "Turbo: much faster, slightly less accurate. Best choice on CPU.",
    "ivrit-ai/whisper-large-v3-ct2": "Large v3: most accurate, slowest.",
}
MEDIA_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
             ".mp4", ".mkv", ".webm", ".mov", ".avi")
RTL_LANGS = {"he", "ar", "fa", "ur"}

_MODELS = {}  # (spec, device, compute, threads) -> [WhisperModel, BatchedInferencePipeline | None]
_DEVNULL = open(os.devnull, "w")


# ----------------------------------------------------------------- helpers
def hms(t):
    t = int(round(max(t, 0)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def ts(t, sep=","):
    ms = int(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def fmt_seg(s):
    return f"[{ts(s['start'], '.')} → {ts(s['end'], '.')}] {s['text']}"


def render(segs, fmt):
    if fmt == "srt":
        return "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n"
                         for i, s in enumerate(segs, 1))
    if fmt == "vtt":
        return "WEBVTT\n\n" + "\n".join(
            f"{ts(s['start'], '.')} --> {ts(s['end'], '.')}\n{s['text']}\n" for s in segs)
    if fmt == "json":
        return json.dumps(segs, ensure_ascii=False, indent=2)
    return "\n".join(s["text"] for s in segs)


def write_out(path, segs, fmt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(segs, fmt))


def probe(path):
    """Audio duration in seconds (reads the header only), or None."""
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return c.duration / av.time_base
    except Exception:
        pass
    return None


def hbox(*items):
    """Row helper: str -> label, '-' -> stretch, (widget, stretch) -> weighted widget."""
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    for it in items:
        if isinstance(it, str):
            lay.addStretch() if it == "-" else lay.addWidget(QLabel(it))
        elif isinstance(it, tuple):
            lay.addWidget(it[0], it[1])
        else:
            lay.addWidget(it)
    return lay


def getv(w):
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    return w.text()


def setv(w, v):
    try:
        if isinstance(w, QComboBox):
            w.setCurrentText(str(v))
        elif isinstance(w, QCheckBox):
            w.setChecked(str(v).lower() in ("true", "1"))
        elif isinstance(w, QSpinBox):
            w.setValue(int(v))
        else:
            w.setText(str(v))
    except (TypeError, ValueError):
        pass


# ----------------------------------------------------------- timing history
class Timings:
    """Append-only JSONL log of every run. Used to predict future run times."""

    def __init__(self):
        d = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(d, exist_ok=True)
        self.dir, self.path = d, os.path.join(d, "timings.jsonl")
        self.records = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.records.append(json.loads(line))
                    except ValueError:
                        pass

    def add(self, rec):
        self.records.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def clear(self):
        self.records.clear()
        if os.path.exists(self.path):
            os.remove(self.path)

    def median(self, field, match, kind="transcribe", last=10):
        vals = [r[field] for r in self.records
                if r.get("kind") == kind and r.get(field)
                and all(r.get(k) == v for k, v in match.items())][-last:]
        return (statistics.median(vals), len(vals)) if vals else (None, 0)

    def best(self, field, exact, loose):
        """Median over runs with identical settings, else over runs with the same model/device/compute."""
        v, n = self.median(field, exact)
        return (v, n) if v else self.median(field, loose)


# ------------------------------------------------------------------ worker
class Cancelled(Exception):
    pass


def make_tqdm(on_bytes):
    """tqdm subclass that reports downloaded bytes (used by huggingface_hub)."""
    class ByteTqdm(tqdm):
        def __init__(self, *a, **kw):
            kw.pop("name", None)  # newer huggingface_hub passes this
            kw["file"], kw["disable"] = _DEVNULL, False
            super().__init__(*a, **kw)
            if self.unit == "B" and kw.get("initial"):
                on_bytes(kw["initial"])

        def update(self, n=1):
            if self.unit == "B" and n:
                on_bytes(n)
            return super().update(n)
    return ByteTqdm


class Worker(QThread):
    bar = Signal(str, int, str)           # which bar, percent (-1 = busy), detail
    status = Signal(str)
    log = Signal(str)
    loadstart = Signal(float)             # predicted load seconds (0 = unknown)
    filestart = Signal(int)
    segment = Signal(int, object)
    result = Signal(int, object, object)  # file index, segments, timing record
    record = Signal(object)
    failed = Signal(str)

    def __init__(self, spec, files, opts, pred, download_only=False):
        super().__init__()
        self.spec, self.files, self.o, self.pred = spec, files, opts, pred
        self.download_only = download_only
        self.cancel = False
        self.load_secs, self.load_cold, self.errors = 0.0, False, 0

    def stop(self):
        self.cancel = True

    def run(self):
        t0 = time.time()
        try:
            path = self.fetch()
            if self.download_only:
                self.status.emit("Download finished")
                return
            entry = self.load(path)
            for i, f in enumerate(self.files):
                if self.cancel:
                    raise Cancelled
                self.one(i, f, entry)
            msg = (f"Done: {len(self.files) - self.errors}/{len(self.files)} file(s) "
                   f"in {hms(time.time() - t0)}")
            self.status.emit(msg)
            self.log.emit(msg)
        except Cancelled:
            self.status.emit("Cancelled")
            self.log.emit("Cancelled")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    # -- stage 1: get the model files (local folder / cache / Hugging Face)
    @staticmethod
    def cached(spec):
        try:
            p = snapshot_download(spec, local_files_only=True)
        except Exception:
            return None
        ok = all(os.path.exists(os.path.join(p, f)) for f in ("model.bin", "config.json"))
        return p if ok else None

    def fetch(self):
        spec = self.spec
        if os.path.isdir(spec):
            self.bar.emit("dl", 100, "Local folder, nothing to download")
            return spec
        p = self.cached(spec)
        if p:
            self.bar.emit("dl", 100, "Already downloaded")
            self.log.emit(f"Using cached model: {p}")
            return p

        self.status.emit("Fetching file list from Hugging Face...")
        self.bar.emit("dl", -1, "Contacting huggingface.co...")
        info = HfApi().model_info(spec, files_metadata=True)
        files = [s for s in info.siblings
                 if not s.rfilename.endswith(".md") and s.rfilename != ".gitattributes"]
        total = sum(s.size or 0 for s in files) or 1
        state, lock = {"done": 0, "last": 0.0}, threading.Lock()
        t0 = time.time()

        def on_bytes(n):
            if self.cancel:
                raise Cancelled
            with lock:
                state["done"] += n
                now = time.time()
                if now - state["last"] < 0.1:
                    return
                state["last"], d = now, state["done"]
            speed = d / max(now - t0, 1e-3)
            eta = (total - d) / speed if speed else 0
            self.bar.emit("dl", min(99, int(100 * d / total)),
                          f"{size(d)} / {size(total)} · {size(speed)}/s · ETA {hms(eta)}")

        self.status.emit(f"Downloading {spec} ({size(total)})...")
        self.log.emit(f"Downloading {spec}: {len(files)} files, {size(total)}")
        path = snapshot_download(spec, allow_patterns=[s.rfilename for s in files],
                                 tqdm_class=make_tqdm(on_bytes))
        secs = time.time() - t0
        self.bar.emit("dl", 100, f"Downloaded {size(total)} in {hms(secs)}")
        self.log.emit(f"Downloaded to {path}")
        self.record.emit({"kind": "download", "ts": datetime.now().isoformat(timespec="seconds"),
                          "model": spec, "bytes": total, "seconds": round(secs, 2),
                          "bytes_per_second": round(total / max(secs, 1e-3))})
        return path

    # -- stage 2: load into memory (cached across runs)
    def load(self, path):
        o = self.o
        key = (self.spec, o["device"], o["compute"], o["threads"])
        if key not in _MODELS:
            self.status.emit("Loading model into memory...")
            self.loadstart.emit(float(self.pred.get("load") or 0))
            t0 = time.time()
            _MODELS[key] = [WhisperModel(path, device=o["device"], compute_type=o["compute"],
                                         cpu_threads=o["threads"]), None]
            self.load_secs, self.load_cold = time.time() - t0, True
            self.bar.emit("load", 100, f"Loaded in {hms(self.load_secs)}")
            self.log.emit(f"Model loaded in {self.load_secs:.1f}s")
        else:
            self.bar.emit("load", 100, "Already loaded")
        entry = _MODELS[key]
        if o["batched"] and entry[1] is None:
            entry[1] = BatchedInferencePipeline(model=entry[0])
        return entry

    # -- stage 3: transcribe one file
    def one(self, i, f, entry):
        o, n = self.o, len(self.files)
        name = os.path.basename(f)
        self.filestart.emit(i)
        self.status.emit(f"Transcribing {name} ({i + 1}/{n})")
        self.bar.emit("tr", -1, "Preparing audio (decoding, silence detection)...")
        out = []
        try:
            kw = dict(language=o["language"], beam_size=o["beam"], vad_filter=o["vad"],
                      word_timestamps=o["words"], initial_prompt=o["prompt"] or None,
                      hotwords=o["hotwords"] or None)
            t0 = time.time()
            if o["batched"]:
                segs, info = entry[1].transcribe(f, batch_size=o["batch_size"], **kw)
            else:
                segs, info = entry[0].transcribe(f, condition_on_previous_text=o["cond"], **kw)
            dur = max(info.duration, 0.1)
            lang = f"{info.language} ({info.language_probability:.0%})"
            for s in segs:
                if self.cancel:
                    raise Cancelled
                seg = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                if o["words"]:
                    seg["words"] = [{"start": w.start, "end": w.end, "word": w.word,
                                     "probability": w.probability} for w in (s.words or [])]
                out.append(seg)
                self.segment.emit(i, seg)

                done, el = min(s.end, dur), time.time() - t0
                if done >= 20 and el > 0:
                    eta = (dur - done) * el / done          # live speed
                elif self.pred.get("rtf"):
                    eta = (dur - done) * self.pred["rtf"]   # history
                else:
                    eta = None
                txt = f"{hms(done)} / {hms(dur)} · {done / max(el, 1e-3):.1f}× realtime · {lang}"
                if eta is not None:
                    txt += f" · ETA {hms(eta)}"
                self.bar.emit("tr", int(100 * done / dur), txt)
                self.bar.emit("overall", int(100 * (i + done / dur) / n), f"File {i + 1}/{n}")

            secs = time.time() - t0
            self.bar.emit("tr", 100, f"{hms(dur)} of audio in {hms(secs)} ({dur / max(secs, 1e-3):.1f}× realtime)")
            self.bar.emit("overall", int(100 * (i + 1) / n), f"File {i + 1}/{n}")
            self.log.emit(f"{name}: {hms(dur)} of audio in {hms(secs)}")
            self.result.emit(i, out, {
                "kind": "transcribe", "ts": datetime.now().isoformat(timespec="seconds"),
                "model": self.spec, "device": o["device"], "compute": o["compute"],
                "threads": o["threads"], "batched": o["batched"], "batch_size": o["batch_size"],
                "beam": o["beam"], "vad": o["vad"], "words": o["words"],
                "audio_seconds": round(dur, 2), "transcribe_seconds": round(secs, 2),
                "rtf": round(secs / dur, 4) if dur >= 10 else None,  # short clips are mostly overhead
                "load_seconds": round(self.load_secs, 2) if i == 0 else 0,
                "load_cold": bool(self.load_cold and i == 0),
                "segments": len(out), "language": info.language,
                "os": platform.platform(), "cpu_count": os.cpu_count(),
                "faster_whisper": getattr(faster_whisper, "__version__", "?"),
            })
        except Cancelled:
            if out:
                self.result.emit(i, out, None)  # keep what was transcribed so far
            raise
        except Exception as e:
            self.errors += 1
            self.log.emit(f"ERROR {name}: {type(e).__name__}: {e}")


def keep_awake(on):
    """Windows: stop the PC sleeping during a long transcription."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def install_crash_handler():
    """Windowed exes swallow exceptions; log them to a file and show a dialog instead."""
    def hook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        try:
            d = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "crash.log"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now()} ---\n{text}")
            QMessageBox.critical(None, "Unexpected error", f"{ev}\n\nDetails were saved to {d}")
        except Exception:
            pass
    sys.excepthook = hook


# ------------------------------------------------------------ timing dialog
class TimingDialog(QDialog):
    def __init__(self, parent, timings):
        super().__init__(parent)
        self.setWindowTitle("Timing history")
        self.resize(900, 380)
        groups = {}
        for r in timings.records:
            if r.get("kind") == "transcribe":
                k = (r["model"], r["device"], r["compute"], r["batched"], r["threads"])
                groups.setdefault(k, []).append(r)
        heads = ["Model", "Device", "Compute", "Batched", "Threads", "Runs", "Audio", "Speed", "Load"]
        t = QTableWidget(len(groups), len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, (k, rs) in enumerate(groups.items()):
            rtfs = [r["rtf"] for r in rs if r.get("rtf")]
            loads = [r["load_seconds"] for r in rs if r.get("load_cold") and r.get("load_seconds")]
            vals = [k[0], k[1], k[2], "yes" if k[3] else "no", k[4] or "auto", len(rs),
                    hms(sum(r["audio_seconds"] for r in rs)),
                    f"{1 / statistics.median(rtfs):.2f}× realtime" if rtfs else "-",
                    hms(statistics.median(loads)) if loads else "-"]
            for c, v in enumerate(vals):
                t.setItem(row, c, QTableWidgetItem(str(v)))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)

        b_open = QPushButton("Open data folder")
        b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(timings.dir)))
        b_clear = QPushButton("Clear history")
        b_clear.clicked.connect(lambda: self.clear(parent, timings))
        path = QLabel(timings.path)
        path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay = QVBoxLayout(self)
        lay.addWidget(t)
        lay.addLayout(hbox(b_open, b_clear, "-"))
        lay.addWidget(path)

    def clear(self, parent, timings):
        if QMessageBox.question(self, "Clear history", "Delete all timing data?") == QMessageBox.Yes:
            timings.clear()
            parent.refresh_estimate()
            self.close()


# --------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper GUI")
        self.resize(1000, 900)
        self.setAcceptDrops(True)
        self.timings = Timings()
        self.settings = QSettings("WhisperGUI", "WhisperGUI")
        self.job = None
        self.results, self.durations = {}, {}
        self.load_t0 = self.load_pred = 0.0
        self.load_timer = QTimer(self, interval=200)
        self.load_timer.timeout.connect(self.tick_load)

        # --- model
        self.model = QComboBox(editable=True)
        for i, (rid, tip) in enumerate(MODELS.items()):
            self.model.addItem(rid)
            self.model.setItemData(i, tip, Qt.ToolTipRole)
        self.model.setToolTip("Hugging Face repo ID (downloaded automatically) or a local CTranslate2 folder")
        b_dir = QPushButton("Local folder...")
        b_dir.clicked.connect(self.pick_model_dir)
        self.b_dl = QPushButton("Download only")
        self.b_dl.setToolTip("Fetch the model into the Hugging Face cache without transcribing")
        self.b_dl.clicked.connect(lambda: self.start(download_only=True))
        self.g_model = QGroupBox("Model")
        QVBoxLayout(self.g_model).addLayout(hbox("Hugging Face repo or folder", (self.model, 1), b_dir, self.b_dl))

        # --- options
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda", "auto"])
        self.compute = QComboBox()
        self.compute.addItems(["int8", "float16", "int8_float16", "float32", "default"])
        self.threads = QSpinBox()
        self.threads.setRange(0, 64)
        self.threads.setSpecialValueText("default (4)")
        self.threads.setValue(max(4, (os.cpu_count() or 8) // 2))  # ~physical cores; CTranslate2's own default is 4
        self.threads.setToolTip("Try a few values: the Timing history table compares them")
        self.lang = QComboBox(editable=True)
        self.lang.addItems(["he", "en", "ar", "ru", "es", "fr", "de", "auto"])
        self.beam = QSpinBox()
        self.beam.setRange(1, 10)
        self.beam.setValue(5)
        self.vad = QCheckBox("VAD (skip silence)")
        self.vad.setChecked(True)
        self.words = QCheckBox("Word timestamps")
        self.cond = QCheckBox("Condition on previous text")
        self.cond.setToolTip("Off reduces repetition loops/hallucinations; on can improve consistency")
        self.batched = QCheckBox("Batched mode")
        self.batched.setToolTip("Mostly worth it on GPU. Needs faster-whisper >= 1.1; ignores 'condition on previous text'.")
        self.batched.setEnabled(BatchedInferencePipeline is not None)
        self.bsize = QSpinBox()
        self.bsize.setRange(1, 64)
        self.bsize.setValue(8)
        self.prompt = QLineEdit(placeholderText="Initial prompt: a natural sentence with names/terms, only affects the start (optional)")
        self.hotwords = QLineEdit(placeholderText="Hotwords: names/terms to favor across the whole file, e.g. כהן Kubernetes (optional)")
        self.g_opts = QGroupBox("Options")
        lay = QVBoxLayout(self.g_opts)
        lay.addLayout(hbox("Device", self.device, "Compute", self.compute, "CPU threads", self.threads,
                           "Language", self.lang, "-"))
        lay.addLayout(hbox("Beam", self.beam, self.vad, self.words, self.cond, "-"))
        lay.addLayout(hbox(self.batched, "batch size", self.bsize, "-"))
        lay.addWidget(self.prompt)
        lay.addWidget(self.hotwords)

        # --- queue
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue.setMinimumHeight(90)
        self.queue.currentRowChanged.connect(self.show_current)
        b_add, b_rm, b_clr = QPushButton("Add files..."), QPushButton("Remove selected"), QPushButton("Clear")
        b_add.clicked.connect(self.pick_files)
        b_rm.clicked.connect(self.remove_selected)
        b_clr.clicked.connect(self.clear_queue)
        self.qbtns = [b_add, b_rm, b_clr]
        g_q = QGroupBox("Files (drag and drop here; folders work too)")
        lq = QVBoxLayout(g_q)
        lq.addWidget(self.queue)
        lq.addLayout(hbox(b_add, b_rm, b_clr, "-"))

        # --- auto-save
        self.fmt = {f: QCheckBox(f) for f in ("txt", "srt", "vtt", "json")}
        self.fmt["txt"].setChecked(True)
        self.outdir = QLineEdit(placeholderText="Next to each audio file")
        b_out = QPushButton("Browse...")
        b_out.clicked.connect(self.pick_outdir)
        self.g_out = QGroupBox("Auto-save results (existing files are overwritten)")
        QVBoxLayout(self.g_out).addLayout(hbox("Formats", *self.fmt.values(), "Folder", (self.outdir, 1), b_out))

        # --- controls
        self.b_start = QPushButton("Transcribe")
        self.b_start.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        self.b_start.clicked.connect(lambda: self.start())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(lambda: self.job and self.job.stop())
        self.b_unload = QPushButton("Unload model")
        self.b_unload.setToolTip("Free the RAM used by the loaded model")
        self.b_unload.clicked.connect(self.unload)
        b_hist = QPushButton("Timing history...")
        b_hist.clicked.connect(lambda: TimingDialog(self, self.timings).exec())
        self.est = QLabel()
        self.est.setWordWrap(True)

        # --- progress
        gp = QGroupBox("Progress")
        grid = QGridLayout(gp)
        self.bars = {}
        for r, (k, name) in enumerate([("overall", "Overall"), ("dl", "Download"),
                                       ("load", "Load model"), ("tr", "Transcribe")]):
            b, d = QProgressBar(), QLabel("")
            b.setRange(0, 100)
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(b, r, 1)
            grid.addWidget(d, r, 2)
            self.bars[k] = (b, d)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)

        # --- output
        self.out = QTextEdit(readOnly=True)
        self.logview = QPlainTextEdit(readOnly=True)
        self.logview.setMaximumBlockCount(5000)
        b_save, b_copy = QPushButton("Save as..."), QPushButton("Copy")
        b_save.clicked.connect(self.save_current)
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.out.toPlainText()))
        self.rtl = QCheckBox("Right-to-left view")
        self.rtl.toggled.connect(self.set_rtl)
        self.lang.currentTextChanged.connect(lambda t: self.rtl.setChecked(t.strip().lower() in RTL_LANGS))
        page = QWidget()
        lp = QVBoxLayout(page)
        lp.addWidget(self.out)
        lp.addLayout(hbox(b_save, b_copy, self.rtl, "-"))
        tabs = QTabWidget()
        tabs.addTab(page, "Transcript")
        tabs.addTab(self.logview, "Log")
        tabs.setMinimumHeight(240)

        central = QWidget()
        v = QVBoxLayout(central)
        for w in (self.g_model, self.g_opts, g_q, self.g_out):
            v.addWidget(w)
        v.addLayout(hbox(self.b_start, self.b_stop, self.b_unload, b_hist, "-"))
        v.addWidget(self.est)
        v.addWidget(gp)
        v.addWidget(tabs, 1)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        # --- settings persistence and live estimate
        self.persist = {"model": self.model, "device": self.device, "compute": self.compute,
                        "threads": self.threads, "lang": self.lang, "rtl": self.rtl, "beam": self.beam,
                        "vad": self.vad, "words": self.words, "cond": self.cond,
                        "batched": self.batched, "bsize": self.bsize, "prompt": self.prompt, "hotwords": self.hotwords,
                        "outdir": self.outdir}
        self.persist.update({f"fmt_{k}": c for k, c in self.fmt.items()})
        for k, w in self.persist.items():
            v = self.settings.value(k)
            if v is not None:
                setv(w, v)
        if BatchedInferencePipeline is None:
            self.batched.setChecked(False)
        for w in (self.model, self.device, self.compute):
            w.currentTextChanged.connect(self.refresh_estimate)
        self.threads.valueChanged.connect(self.refresh_estimate)
        self.batched.toggled.connect(self.refresh_estimate)

        self.statusBar().showMessage("Ready")
        self.logline(f"faster-whisper {getattr(faster_whisper, '__version__', '?')} · timing data: {self.timings.path}")
        self.refresh_estimate()

    # ---------------------------------------------------------- queue / files
    def paths(self):
        return [self.queue.item(i).data(Qt.UserRole) for i in range(self.queue.count())]

    def set_mark(self, i, mark):
        p = self.queue.item(i).data(Qt.UserRole)
        d = self.durations.get(p)
        self.queue.item(i).setText(f"{mark}{os.path.basename(p)}" + (f"   [{hms(d)}]" if d else ""))

    def add_paths(self, ps):
        have = set(self.paths())
        for p in ps:
            if os.path.isdir(p):
                self.add_paths(sorted(os.path.join(r, f) for r, _, fs in os.walk(p)
                                      for f in fs if f.lower().endswith(MEDIA_EXT)))
                continue
            if p in have or not os.path.isfile(p):
                continue
            it = QListWidgetItem()
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            self.queue.addItem(it)
            have.add(p)
            self.durations[p] = probe(p)
            self.set_mark(self.queue.count() - 1, "")
        self.refresh_estimate()

    def pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(
            self, "Add audio/video", "",
            "Media (" + " ".join("*" + e for e in MEDIA_EXT) + ");;All files (*)")
        self.add_paths(fs)

    def remove_selected(self):
        for it in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(it))
        self.refresh_estimate()

    def clear_queue(self):
        self.queue.clear()
        self.results.clear()
        self.out.clear()
        self.refresh_estimate()

    def pick_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select CTranslate2 model folder")
        if d:
            self.model.setCurrentText(d)

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.outdir.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls()])

    # -------------------------------------------------------------- settings
    def opts(self):
        lang = self.lang.currentText().strip().lower()
        return dict(device=self.device.currentText(), compute=self.compute.currentText(),
                    threads=self.threads.value(), language=None if lang in ("", "auto") else lang,
                    beam=self.beam.value(), vad=self.vad.isChecked(),
                    batched=self.batched.isChecked() and BatchedInferencePipeline is not None,
                    batch_size=self.bsize.value(), words=self.words.isChecked(),
                    cond=self.cond.isChecked(), prompt=self.prompt.text().strip(),
                    hotwords=self.hotwords.text().strip())

    def predict(self):
        o, spec = self.opts(), self.model.currentText().strip()
        exact = {"model": spec, "device": o["device"], "compute": o["compute"],
                 "batched": o["batched"], "threads": o["threads"]}
        loose = {k: exact[k] for k in ("model", "device", "compute")}
        rtf, n = self.timings.best("rtf", exact, loose)
        warm = (spec, o["device"], o["compute"], o["threads"]) in _MODELS
        load = 0 if warm else (self.timings.best("load_seconds", {**exact, "load_cold": True},
                                                 {**loose, "load_cold": True})[0] or 0)
        return {"rtf": rtf, "n": n, "load": load}

    def refresh_estimate(self, *_):
        if not hasattr(self, "est"):
            return
        ps = self.paths()
        known = [self.durations[p] for p in ps if self.durations.get(p)]
        if not ps:
            self.est.setText("Add audio files to see a time estimate.")
            return
        total = sum(known)
        text = f"Audio: {hms(total)}" + (" (some durations unknown)" if len(known) < len(ps) else "")
        p = self.predict()
        if p["rtf"] and known:
            text += f"  ·  Estimated time: ~{hms(p['load'] + total * p['rtf'])} (based on {p['n']} earlier run(s))"
        else:
            text += "  ·  No timing history for these settings yet; estimates start after the first run."
        self.est.setText(text)

    def set_rtl(self, on):
        opt = self.out.document().defaultTextOption()
        opt.setTextDirection(Qt.RightToLeft if on else Qt.LeftToRight)
        self.out.document().setDefaultTextOption(opt)

    # ------------------------------------------------------------------ run
    def busy(self, b):
        for w in (self.g_model, self.g_opts, self.g_out, self.b_start, self.b_unload, *self.qbtns):
            w.setEnabled(not b)
        self.b_stop.setEnabled(b)
        keep_awake(b)

    def start(self, download_only=False):
        if self.job and self.job.isRunning():
            return
        spec = self.model.currentText().strip()
        files = [] if download_only else self.paths()
        if not spec or (not os.path.isdir(spec) and spec.count("/") != 1):
            QMessageBox.warning(self, "Model", "Enter a Hugging Face repo ID like user/name, or choose a local folder.")
            return
        if not download_only and not files:
            QMessageBox.warning(self, "No files", "Add at least one audio file.")
            return
        self.results.clear()
        self.out.clear()
        for i in range(self.queue.count()):
            self.set_mark(i, "")
        for k in self.bars:
            self.on_bar(k, 0, "")
        j = self.job = Worker(spec, files, self.opts(), self.predict(), download_only)
        j.bar.connect(self.on_bar)
        j.status.connect(self.statusBar().showMessage)
        j.log.connect(self.logline)
        j.loadstart.connect(self.on_loadstart)
        j.filestart.connect(self.on_filestart)
        j.segment.connect(self.on_segment)
        j.result.connect(self.on_result)
        j.record.connect(self.on_record)
        j.failed.connect(self.on_failed)
        j.finished.connect(self.on_finished)
        self.busy(True)
        j.start()

    def on_bar(self, which, pct, detail):
        bar, lab = self.bars[which]
        if pct < 0:
            bar.setRange(0, 0)
        else:
            bar.setRange(0, 100)
            bar.setValue(pct)
        lab.setText(detail)
        if which == "load" and pct >= 100:
            self.load_timer.stop()

    def on_loadstart(self, pred):
        self.load_t0, self.load_pred = time.time(), pred
        self.load_timer.start()
        self.tick_load()

    def tick_load(self):
        el = time.time() - self.load_t0
        if self.load_pred > 0:
            self.on_bar("load", min(95, int(100 * el / self.load_pred)),
                        f"Loading... {hms(el)} (usually ~{hms(self.load_pred)})")
        else:
            self.on_bar("load", -1, f"Loading... {hms(el)}")

    def on_filestart(self, i):
        p = self.paths()[i]
        self.results[p] = []
        self.set_mark(i, "▶ ")
        self.queue.setCurrentRow(i)

    def on_segment(self, i, seg):
        self.results[self.paths()[i]].append(seg)
        if self.queue.currentRow() == i:
            self.out.append(fmt_seg(seg))

    def on_result(self, i, segs, rec):
        p = self.paths()[i]
        self.results[p] = segs
        self.set_mark(i, "✓ " if rec else "◐ ")
        if rec:
            self.timings.add(rec)
        else:
            self.logline(f"Stopped early: saving partial transcript of {os.path.basename(p)}")
        self.autosave(p, segs)

    def on_record(self, rec):
        self.timings.add(rec)

    def on_failed(self, msg):
        self.logline("ERROR " + msg)
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Error", msg)

    def on_finished(self):
        self.load_timer.stop()
        self.busy(False)
        for i in range(self.queue.count()):
            if self.queue.item(i).text().startswith("▶"):
                self.set_mark(i, "✗ ")
        self.refresh_estimate()
        QApplication.alert(self)  # flash the taskbar button

    def unload(self):
        if self.job and self.job.isRunning():
            return
        _MODELS.clear()
        gc.collect()
        self.logline("Model unloaded")
        self.refresh_estimate()

    # --------------------------------------------------------------- output
    def show_current(self, row):
        if row < 0:
            return
        segs = self.results.get(self.queue.item(row).data(Qt.UserRole), [])
        self.out.setPlainText("\n".join(fmt_seg(s) for s in segs))

    def autosave(self, path, segs):
        out_dir = self.outdir.text().strip() or os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        for f, cb in self.fmt.items():
            if cb.isChecked():
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    target = os.path.join(out_dir, f"{base}.{f}")
                    write_out(target, segs, f)
                    self.logline(f"Saved {target}")
                except OSError as e:
                    self.logline(f"ERROR saving {base}.{f}: {e}")

    def save_current(self):
        row = self.queue.currentRow()
        segs = self.results.get(self.queue.item(row).data(Qt.UserRole)) if row >= 0 else None
        if not segs:
            return
        base = os.path.splitext(self.queue.item(row).data(Qt.UserRole))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", base + ".txt",
            "Text (*.txt);;SubRip (*.srt);;WebVTT (*.vtt);;JSON (*.json)")
        if path:
            write_out(path, segs, os.path.splitext(path)[1].lstrip(".").lower() or "txt")
            self.statusBar().showMessage(f"Saved {path}")

    def logline(self, s):
        self.logview.appendPlainText(f"{datetime.now():%H:%M:%S}  {s}")

    def closeEvent(self, e):
        if self.job and self.job.isRunning():
            self.job.stop()
            self.job.wait(8000)
        for k, w in self.persist.items():
            self.settings.setValue(k, getv(w))
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("WhisperGUI")
    install_crash_handler()
    win = Main()
    win.show()
    sys.exit(app.exec())
