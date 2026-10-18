#!/usr/bin/env python3
"""Whisper GUI, wxPython edition: faster-whisper + Hugging Face model download.

    pip install -U wxPython faster-whisper huggingface_hub
    python whisper_gui_wx.py
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
import wx
import wx.lib.scrolledpanel as scrolled
from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # faster-whisper < 1.1
    BatchedInferencePipeline = None
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

MODELS = {
    "ivrit-ai/whisper-large-v3-turbo-ct2": "Turbo: much faster, slightly less accurate. Best choice on CPU.",
    "ivrit-ai/whisper-large-v3-ct2": "Large v3: most accurate, slowest.",
}
MEDIA_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
             ".mp4", ".mkv", ".webm", ".mov", ".avi")
RTL_LANGS = {"he", "ar", "fa", "ur"}
FORMATS = ("txt", "srt", "vtt", "json")

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


def keep_awake(on):
    """Windows: stop the PC sleeping during a long transcription."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def data_dir():
    d = wx.StandardPaths.Get().GetUserDataDir()  # Windows: %APPDATA%\WhisperGUI
    os.makedirs(d, exist_ok=True)
    return d


def install_crash_handler():
    """Windowed exes swallow exceptions; log them to a file and show a dialog instead."""
    def hook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        try:
            d = data_dir()
            with open(os.path.join(d, "crash.log"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now()} ---\n{text}")
            wx.MessageBox(f"{ev}\n\nDetails were saved to {d}", "Unexpected error", wx.ICON_ERROR)
        except Exception:
            pass
    sys.excepthook = hook


def getv(w):
    return w.GetValue()


def setv(w, v):
    try:
        if isinstance(w, wx.CheckBox):
            w.SetValue(str(v).lower() in ("true", "1"))
        elif isinstance(w, wx.SpinCtrl):
            w.SetValue(int(v))
        else:
            w.SetValue(str(v))
    except (TypeError, ValueError, RuntimeError):
        pass


def hrow(parent, *items):
    """Row helper: str -> label, '-' -> stretch, (widget, stretch) -> weighted widget."""
    s = wx.BoxSizer(wx.HORIZONTAL)
    for it in items:
        if isinstance(it, str):
            if it == "-":
                s.AddStretchSpacer(1)
            else:
                s.Add(wx.StaticText(parent, label=it), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        elif isinstance(it, tuple):
            s.Add(it[0], it[1], wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        else:
            s.Add(it, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    return s


class Settings:
    def __init__(self, d):
        self.path = os.path.join(d, "settings_wx.json")
        try:
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
        except OSError:
            pass


# ----------------------------------------------------------- timing history
class Timings:
    """Append-only JSONL log of every run. Used to predict future run times."""

    def __init__(self, d):
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


class Sig:
    """Calls a UI method on the main thread (the wx stand-in for a Qt signal)."""

    def __init__(self, ui, fn):
        self.ui, self.fn = ui, fn

    def emit(self, *a):
        if not self.ui.closing:
            wx.CallAfter(self.fn, *a)


class Worker(threading.Thread):
    def __init__(self, ui, spec, files, opts, pred, download_only=False):
        super().__init__(daemon=True)
        self.ui = ui
        self.bar = Sig(ui, ui.on_bar)               # which bar, percent (-1 = busy), detail
        self.status = Sig(ui, ui.set_status)
        self.log = Sig(ui, ui.logline)
        self.loadstart = Sig(ui, ui.on_loadstart)   # predicted load seconds (0 = unknown)
        self.filestart = Sig(ui, ui.on_filestart)
        self.segment = Sig(ui, ui.on_segment)
        self.result = Sig(ui, ui.on_result)         # file index, segments, timing record
        self.record = Sig(ui, ui.on_record)
        self.failed = Sig(ui, ui.on_failed)
        self.spec, self.files, self.o, self.pred = spec, files, opts, pred
        self.download_only = download_only
        self.cancel = False
        self.load_secs, self.load_cold, self.errors = 0.0, False, 0

    def stop(self):
        self.cancel = True

    def run(self):
        try:
            self.work()
        finally:
            if not self.ui.closing:
                wx.CallAfter(self.ui.on_finished)

    def work(self):
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


# ------------------------------------------------------------ timing dialog
class TimingDialog(wx.Dialog):
    def __init__(self, parent, timings):
        super().__init__(parent, title="Timing history", size=(900, 400),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.parent, self.timings = parent, timings
        groups = {}
        for r in timings.records:
            if r.get("kind") == "transcribe":
                k = (r["model"], r["device"], r["compute"], r["batched"], r["threads"])
                groups.setdefault(k, []).append(r)
        heads = [("Model", 260), ("Device", 60), ("Compute", 90), ("Batched", 60), ("Threads", 60),
                 ("Runs", 50), ("Audio", 70), ("Speed", 120), ("Load", 60)]
        lc = wx.ListCtrl(self, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        for c, (h, w) in enumerate(heads):
            lc.InsertColumn(c, h, width=w)
        for k, rs in groups.items():
            rtfs = [r["rtf"] for r in rs if r.get("rtf")]
            loads = [r["load_seconds"] for r in rs if r.get("load_cold") and r.get("load_seconds")]
            vals = [k[0], k[1], k[2], "yes" if k[3] else "no", k[4] or "default", len(rs),
                    hms(sum(r["audio_seconds"] for r in rs)),
                    f"{1 / statistics.median(rtfs):.2f}× realtime" if rtfs else "-",
                    hms(statistics.median(loads)) if loads else "-"]
            idx = lc.InsertItem(lc.GetItemCount(), str(vals[0]))
            for c, v in enumerate(vals[1:], 1):
                lc.SetItem(idx, c, str(v))

        b_open = wx.Button(self, label="Open data folder")
        b_open.Bind(wx.EVT_BUTTON, lambda e: wx.LaunchDefaultApplication(timings.dir))
        b_clear = wx.Button(self, label="Clear history")
        b_clear.Bind(wx.EVT_BUTTON, self.on_clear)
        path = wx.StaticText(self, label=timings.path)
        v = wx.BoxSizer(wx.VERTICAL)
        v.Add(lc, 1, wx.EXPAND | wx.ALL, 6)
        v.Add(hrow(self, b_open, b_clear), 0, wx.ALL, 6)
        v.Add(path, 0, wx.LEFT | wx.BOTTOM, 8)
        self.SetSizer(v)

    def on_clear(self, e):
        if wx.MessageBox("Delete all timing data?", "Clear history", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            self.timings.clear()
            self.parent.refresh_estimate()
            self.EndModal(wx.ID_OK)


class FileDrop(wx.FileDropTarget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

    def OnDropFiles(self, x, y, filenames):
        self.ui.add_paths(list(filenames))
        return True


# --------------------------------------------------------------- main window
class Main(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Whisper GUI", size=(1000, 900))
        self.closing = False
        self.timings = Timings(data_dir())
        self.settings = Settings(data_dir())
        self.job = None
        self.files, self.results, self.durations = [], {}, {}
        self.cur = -1
        self.pulsing, self.load_active = set(), False
        self.load_t0 = self.load_pred = 0.0

        P = self.panel = scrolled.ScrolledPanel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        def group(title):
            box = wx.StaticBoxSizer(wx.VERTICAL, P, title)
            root.Add(box, 0, wx.EXPAND | wx.ALL, 4)
            return box, box.GetStaticBox()

        def add(box, sizer_or_win):
            box.Add(sizer_or_win, 0, wx.EXPAND | wx.ALL, 3)

        # --- model
        gm, S = group("Model")
        self.g_model = S
        first = next(iter(MODELS))
        self.model = wx.ComboBox(S, choices=list(MODELS), style=wx.CB_DROPDOWN, value=first)
        self.model.SetToolTip("Hugging Face repo ID (downloaded automatically) or a local CTranslate2 folder")
        self.model_help = wx.StaticText(S, label=MODELS[first])
        b_dir = wx.Button(S, label="Local folder...")
        b_dir.Bind(wx.EVT_BUTTON, self.pick_model_dir)
        b_dl = wx.Button(S, label="Download only")
        b_dl.SetToolTip("Fetch the model into the Hugging Face cache without transcribing")
        b_dl.Bind(wx.EVT_BUTTON, lambda e: self.start(download_only=True))
        add(gm, hrow(S, "Hugging Face repo or folder", (self.model, 1), b_dir, b_dl))
        gm.Add(self.model_help, 0, wx.LEFT | wx.BOTTOM, 8)

        # --- options
        go, S = group("Options")
        self.g_opts = S
        self.device = wx.ComboBox(S, choices=["cpu", "cuda", "auto"], style=wx.CB_READONLY, value="cpu")
        self.compute = wx.ComboBox(S, choices=["int8", "float16", "int8_float16", "float32", "default"],
                                   style=wx.CB_READONLY, value="int8")
        self.threads = wx.SpinCtrl(S, min=0, max=64, size=(70, -1),
                                   initial=max(4, (os.cpu_count() or 8) // 2))
        self.threads.SetToolTip("0 = CTranslate2 default (4). Try a few values: the Timing history table compares them")
        self.lang = wx.ComboBox(S, choices=["he", "en", "ar", "ru", "es", "fr", "de", "auto"],
                                style=wx.CB_DROPDOWN, value="he")
        self.beam = wx.SpinCtrl(S, min=1, max=10, initial=5, size=(60, -1))
        self.vad = wx.CheckBox(S, label="VAD (skip silence)")
        self.vad.SetValue(True)
        self.words = wx.CheckBox(S, label="Word timestamps")
        self.cond = wx.CheckBox(S, label="Condition on previous text")
        self.cond.SetToolTip("Off reduces repetition loops/hallucinations; on can improve consistency")
        self.batched = wx.CheckBox(S, label="Batched mode")
        self.batched.SetToolTip("Mostly worth it on GPU. Needs faster-whisper >= 1.1.")
        self.batched.Enable(BatchedInferencePipeline is not None)
        self.bsize = wx.SpinCtrl(S, min=1, max=64, initial=8, size=(60, -1))
        self.prompt = wx.TextCtrl(S)
        self.prompt.SetHint("Initial prompt: a natural sentence with names/terms, only affects the start (optional)")
        self.hotwords = wx.TextCtrl(S)
        self.hotwords.SetHint("Hotwords: names/terms to favor across the whole file, e.g. כהן Kubernetes (optional)")
        add(go, hrow(S, "Device", self.device, "Compute", self.compute, "CPU threads", self.threads,
                     "Language", self.lang))
        add(go, hrow(S, "Beam", self.beam, self.vad, self.words, self.cond))
        add(go, hrow(S, self.batched, "batch size", self.bsize))
        add(go, self.prompt)
        add(go, self.hotwords)

        # --- queue
        gq, S = group("Files (drag and drop here; folders work too)")
        self.queue = wx.ListBox(S, style=wx.LB_EXTENDED, size=(-1, 110))
        self.queue.Bind(wx.EVT_LISTBOX, self.on_queue_select)
        b_add, b_rm, b_clr = (wx.Button(S, label=t) for t in ("Add files...", "Remove selected", "Clear"))
        b_add.Bind(wx.EVT_BUTTON, self.pick_files)
        b_rm.Bind(wx.EVT_BUTTON, self.remove_selected)
        b_clr.Bind(wx.EVT_BUTTON, self.clear_queue)
        self.qbtns = [b_add, b_rm, b_clr]
        gq.Add(self.queue, 1, wx.EXPAND | wx.ALL, 3)
        add(gq, hrow(S, b_add, b_rm, b_clr))

        # --- auto-save
        gs, S = group("Auto-save results (existing files are overwritten)")
        self.g_out = S
        self.fmt = {f: wx.CheckBox(S, label=f) for f in FORMATS}
        self.fmt["txt"].SetValue(True)
        self.outdir = wx.TextCtrl(S)
        self.outdir.SetHint("Next to each audio file")
        b_out = wx.Button(S, label="Browse...")
        b_out.Bind(wx.EVT_BUTTON, self.pick_outdir)
        add(gs, hrow(S, "Formats", *self.fmt.values(), "Folder", (self.outdir, 1), b_out))

        # --- controls
        self.b_start = wx.Button(P, label="Transcribe")
        self.b_start.SetFont(self.b_start.GetFont().Bold())
        self.b_start.Bind(wx.EVT_BUTTON, lambda e: self.start())
        self.b_stop = wx.Button(P, label="Stop")
        self.b_stop.Enable(False)
        self.b_stop.Bind(wx.EVT_BUTTON, lambda e: self.job and self.job.stop())
        self.b_unload = wx.Button(P, label="Unload model")
        self.b_unload.SetToolTip("Free the RAM used by the loaded model")
        self.b_unload.Bind(wx.EVT_BUTTON, self.unload)
        b_hist = wx.Button(P, label="Timing history...")
        b_hist.Bind(wx.EVT_BUTTON, lambda e: TimingDialog(self, self.timings).ShowModal())
        root.Add(hrow(P, self.b_start, self.b_stop, self.b_unload, b_hist), 0, wx.ALL, 6)
        self.est = wx.StaticText(P, style=wx.ST_NO_AUTORESIZE | wx.ST_ELLIPSIZE_END, size=(-1, 22))
        root.Add(self.est, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # --- progress
        gp, S = group("Progress")
        grid = wx.FlexGridSizer(4, 3, 4, 8)
        grid.AddGrowableCol(1, 2)
        grid.AddGrowableCol(2, 3)
        self.bars = {}
        for k, name in (("overall", "Overall"), ("dl", "Download"), ("load", "Load model"), ("tr", "Transcribe")):
            g = wx.Gauge(S, range=100, size=(-1, 18))
            d = wx.StaticText(S, label="", style=wx.ST_NO_AUTORESIZE | wx.ST_ELLIPSIZE_END, size=(300, -1))
            grid.Add(wx.StaticText(S, label=name), 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(g, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
            grid.Add(d, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
            self.bars[k] = (g, d)
        add(gp, grid)

        # --- output
        nb = wx.Notebook(P)
        page = wx.Panel(nb)
        self.out = wx.TextCtrl(page, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        b_save, b_copy = wx.Button(page, label="Save as..."), wx.Button(page, label="Copy")
        b_save.Bind(wx.EVT_BUTTON, self.save_current)
        b_copy.Bind(wx.EVT_BUTTON, self.copy_out)
        self.rtl = wx.CheckBox(page, label="Right-to-left view")
        pv = wx.BoxSizer(wx.VERTICAL)
        pv.Add(self.out, 1, wx.EXPAND | wx.ALL, 3)
        pv.Add(hrow(page, b_save, b_copy, self.rtl), 0, wx.ALL, 3)
        page.SetSizer(pv)
        self.logview = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        nb.AddPage(page, "Transcript")
        nb.AddPage(self.logview, "Log")
        nb.SetMinSize((-1, 260))
        root.Add(nb, 1, wx.EXPAND | wx.ALL, 4)

        P.SetSizer(root)
        P.SetupScrolling(scroll_x=False)

        # --- settings persistence
        self.persist = {"model": self.model, "device": self.device, "compute": self.compute,
                        "threads": self.threads, "lang": self.lang, "rtl": self.rtl, "beam": self.beam,
                        "vad": self.vad, "words": self.words, "cond": self.cond,
                        "batched": self.batched, "bsize": self.bsize, "prompt": self.prompt,
                        "hotwords": self.hotwords, "outdir": self.outdir}
        self.persist.update({f"fmt_{k}": c for k, c in self.fmt.items()})
        for k, w in self.persist.items():
            if k in self.settings.data:
                setv(w, self.settings.data[k])
        if BatchedInferencePipeline is None:
            self.batched.SetValue(False)
        self.set_rtl(self.rtl.GetValue())
        self.on_model()

        # --- events (bound after loading settings so they don't fire during init)
        for w in (self.device, self.compute):
            w.Bind(wx.EVT_COMBOBOX, self.refresh_estimate)
        self.model.Bind(wx.EVT_COMBOBOX, self.on_model)
        self.model.Bind(wx.EVT_TEXT, self.on_model)
        self.lang.Bind(wx.EVT_COMBOBOX, self.on_lang)
        self.lang.Bind(wx.EVT_TEXT, self.on_lang)
        self.threads.Bind(wx.EVT_SPINCTRL, self.refresh_estimate)
        self.batched.Bind(wx.EVT_CHECKBOX, self.refresh_estimate)
        self.rtl.Bind(wx.EVT_CHECKBOX, lambda e: self.set_rtl(self.rtl.GetValue()))
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.timer = wx.Timer(self)  # animates busy bars and the model-load bar
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(100)

        self.set_drop(self)
        self.CreateStatusBar()
        self.SetStatusText("Ready")
        self.logline(f"faster-whisper {getattr(faster_whisper, '__version__', '?')} · timing data: {self.timings.path}")
        self.refresh_estimate()
        self.Centre()

    # ---------------------------------------------------------- queue / files
    def set_drop(self, w):
        try:
            w.SetDropTarget(FileDrop(self))
        except Exception:
            pass
        for c in w.GetChildren():
            self.set_drop(c)

    def set_mark(self, i, mark):
        p = self.files[i]
        d = self.durations.get(p)
        self.queue.SetString(i, f"{mark}{os.path.basename(p)}" + (f"   [{hms(d)}]" if d else ""))

    def add_paths(self, ps):
        for p in ps:
            if os.path.isdir(p):
                self.add_paths(sorted(os.path.join(r, f) for r, _, fs in os.walk(p)
                                      for f in fs if f.lower().endswith(MEDIA_EXT)))
                continue
            if p in self.files or not os.path.isfile(p):
                continue
            self.files.append(p)
            self.durations[p] = probe(p)
            self.queue.Append("")
            self.set_mark(len(self.files) - 1, "")
        self.refresh_estimate()

    def pick_files(self, e=None):
        wild = "Media|" + ";".join("*" + x for x in MEDIA_EXT) + "|All files|*.*"
        with wx.FileDialog(self, "Add audio/video", wildcard=wild,
                           style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.add_paths(dlg.GetPaths())

    def remove_selected(self, e=None):
        for i in sorted(self.queue.GetSelections(), reverse=True):
            self.queue.Delete(i)
            del self.files[i]
        self.cur = -1
        self.refresh_estimate()

    def clear_queue(self, e=None):
        self.queue.Clear()
        self.files.clear()
        self.results.clear()
        self.out.Clear()
        self.cur = -1
        self.refresh_estimate()

    def pick_model_dir(self, e=None):
        with wx.DirDialog(self, "Select CTranslate2 model folder") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.model.SetValue(dlg.GetPath())

    def pick_outdir(self, e=None):
        with wx.DirDialog(self, "Select output folder") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.outdir.SetValue(dlg.GetPath())

    def on_queue_select(self, e):
        sel = self.queue.GetSelections()
        self.cur = sel[0] if sel else -1
        self.show_current(self.cur)

    # -------------------------------------------------------------- settings
    def opts(self):
        lang = self.lang.GetValue().strip().lower()
        return dict(device=self.device.GetValue(), compute=self.compute.GetValue(),
                    threads=self.threads.GetValue(), language=None if lang in ("", "auto") else lang,
                    beam=self.beam.GetValue(), vad=self.vad.GetValue(),
                    batched=self.batched.GetValue() and BatchedInferencePipeline is not None,
                    batch_size=self.bsize.GetValue(), words=self.words.GetValue(),
                    cond=self.cond.GetValue(), prompt=self.prompt.GetValue().strip(),
                    hotwords=self.hotwords.GetValue().strip())

    def predict(self):
        o, spec = self.opts(), self.model.GetValue().strip()
        exact = {"model": spec, "device": o["device"], "compute": o["compute"],
                 "batched": o["batched"], "threads": o["threads"]}
        loose = {k: exact[k] for k in ("model", "device", "compute")}
        rtf, n = self.timings.best("rtf", exact, loose)
        warm = (spec, o["device"], o["compute"], o["threads"]) in _MODELS
        load = 0 if warm else (self.timings.best("load_seconds", {**exact, "load_cold": True},
                                                 {**loose, "load_cold": True})[0] or 0)
        return {"rtf": rtf, "n": n, "load": load}

    def refresh_estimate(self, e=None):
        if not hasattr(self, "est"):
            return
        ps = self.files
        known = [self.durations[p] for p in ps if self.durations.get(p)]
        if not ps:
            self.est.SetLabel("Add audio files to see a time estimate.")
            return
        total = sum(known)
        text = f"Audio: {hms(total)}" + (" (some durations unknown)" if len(known) < len(ps) else "")
        p = self.predict()
        if p["rtf"] and known:
            text += f"  ·  Estimated time: ~{hms(p['load'] + total * p['rtf'])} (based on {p['n']} earlier run(s))"
        else:
            text += "  ·  No timing history for these settings yet; estimates start after the first run."
        self.est.SetLabel(text)

    def on_model(self, e=None):
        self.model_help.SetLabel(MODELS.get(self.model.GetValue().strip(), ""))
        self.refresh_estimate()

    def on_lang(self, e=None):
        on = self.lang.GetValue().strip().lower() in RTL_LANGS
        self.rtl.SetValue(on)
        self.set_rtl(on)

    def set_rtl(self, on):
        try:
            self.out.SetLayoutDirection(wx.Layout_RightToLeft if on else wx.Layout_LeftToRight)
        except Exception:
            pass

    # ------------------------------------------------------------------ run
    def busy(self, b):
        for w in (self.g_model, self.g_opts, self.g_out, self.b_start, self.b_unload, *self.qbtns):
            w.Enable(not b)
        self.b_stop.Enable(b)
        keep_awake(b)

    def start(self, download_only=False):
        if self.job and self.job.is_alive():
            return
        spec = self.model.GetValue().strip()
        files = [] if download_only else list(self.files)
        if not spec or (not os.path.isdir(spec) and spec.count("/") != 1):
            wx.MessageBox("Enter a Hugging Face repo ID like user/name, or choose a local folder.",
                          "Model", wx.ICON_WARNING)
            return
        if not download_only and not files:
            wx.MessageBox("Add at least one audio file.", "No files", wx.ICON_WARNING)
            return
        self.results.clear()
        self.out.Clear()
        for i in range(len(self.files)):
            self.set_mark(i, "")
        for k in self.bars:
            self.on_bar(k, 0, "")
        self.job = Worker(self, spec, files, self.opts(), self.predict(), download_only)
        self.busy(True)
        self.job.start()

    def on_bar(self, which, pct, detail):
        gauge, lab = self.bars[which]
        if pct < 0:
            self.pulsing.add(which)
            gauge.Pulse()
        else:
            self.pulsing.discard(which)
            gauge.SetValue(min(pct, 100))
        lab.SetLabel(detail)
        if which == "load" and pct >= 100:
            self.load_active = False

    def on_timer(self, e):
        for k in list(self.pulsing):
            self.bars[k][0].Pulse()
        if self.load_active:
            self.tick_load()

    def on_loadstart(self, pred):
        self.load_t0, self.load_pred, self.load_active = time.time(), pred, True
        self.tick_load()

    def tick_load(self):
        el = time.time() - self.load_t0
        if self.load_pred > 0:
            self.on_bar("load", min(95, int(100 * el / self.load_pred)),
                        f"Loading... {hms(el)} (usually ~{hms(self.load_pred)})")
        else:
            self.on_bar("load", -1, f"Loading... {hms(el)}")

    def on_filestart(self, i):
        self.results[self.files[i]] = []
        self.set_mark(i, "▶ ")
        for s in self.queue.GetSelections():
            self.queue.Deselect(s)
        self.queue.SetSelection(i)
        self.cur = i
        self.show_current(i)

    def on_segment(self, i, seg):
        self.results[self.files[i]].append(seg)
        if self.cur == i:
            self.out.AppendText(fmt_seg(seg) + "\n")

    def on_result(self, i, segs, rec):
        p = self.files[i]
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
        self.SetStatusText("Failed")
        wx.MessageBox(msg, "Error", wx.ICON_ERROR)

    def on_finished(self):
        self.load_active = False
        self.busy(False)
        for i in range(self.queue.GetCount()):
            if self.queue.GetString(i).startswith("▶"):
                self.set_mark(i, "✗ ")
        self.refresh_estimate()
        self.RequestUserAttention()  # flash the taskbar button

    def set_status(self, s):
        self.SetStatusText(s)

    def unload(self, e=None):
        if self.job and self.job.is_alive():
            return
        _MODELS.clear()
        gc.collect()
        self.logline("Model unloaded")
        self.refresh_estimate()

    # --------------------------------------------------------------- output
    def show_current(self, row):
        if row < 0 or row >= len(self.files):
            return
        segs = self.results.get(self.files[row], [])
        self.out.SetValue("\n".join(fmt_seg(s) for s in segs))

    def autosave(self, path, segs):
        out_dir = self.outdir.GetValue().strip() or os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        for f, cb in self.fmt.items():
            if cb.GetValue():
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    target = os.path.join(out_dir, f"{base}.{f}")
                    write_out(target, segs, f)
                    self.logline(f"Saved {target}")
                except OSError as ex:
                    self.logline(f"ERROR saving {base}.{f}: {ex}")

    def save_current(self, e=None):
        row = self.cur
        segs = self.results.get(self.files[row]) if 0 <= row < len(self.files) else None
        if not segs:
            return
        base = os.path.splitext(self.files[row])[0]
        wild = "Text (*.txt)|*.txt|SubRip (*.srt)|*.srt|WebVTT (*.vtt)|*.vtt|JSON (*.json)|*.json"
        with wx.FileDialog(self, "Save transcript", defaultDir=os.path.dirname(base),
                           defaultFile=os.path.basename(base) + ".txt", wildcard=wild,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            if ext not in FORMATS:
                ext = FORMATS[dlg.GetFilterIndex()]
                path += "." + ext
        write_out(path, segs, ext)
        self.SetStatusText(f"Saved {path}")

    def copy_out(self, e=None):
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(self.out.GetValue()))
            wx.TheClipboard.Close()

    def logline(self, s):
        self.logview.AppendText(f"{datetime.now():%H:%M:%S}  {s}\n")

    def on_close(self, e):
        self.closing = True
        self.timer.Stop()
        if self.job and self.job.is_alive():
            self.job.stop()
            self.job.join(8)
        self.settings.data = {k: getv(w) for k, w in self.persist.items()}
        self.settings.save()
        self.Destroy()


if __name__ == "__main__":
    app = wx.App(False)
    app.SetAppName("WhisperGUI")
    install_crash_handler()
    frame = Main()
    frame.Show()
    app.MainLoop()
