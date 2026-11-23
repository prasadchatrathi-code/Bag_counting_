"""Demo / operator web UI.

  mise run serve            # http://127.0.0.1:8008

Two modes:
  * Recorded runs: browse every `out/<run>/` that has a summary.json; play the
    annotated video, step through the auditable event log (thumbnail, camera
    clock, direction) and compare against hand-counted ground truth where a
    truth file exists for that clip.
  * Live: start the pipeline on a clip (or, later, an RTSP URL) and watch the
    annotated frames and the event feed arrive as they are produced. This is
    the same Pipeline class the CLI uses, driven through its callbacks.
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Config
from .evaluate import evaluate
from .pipeline import Pipeline, parse_line

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
CLIPS = ROOT / "clips"
TRUTH = ROOT / "data" / "truth"
STATIC = Path(__file__).parent / "static"
CLIP_META = ROOT / "configs" / "clips.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """On shutdown, stop a live run and wait for its thread before uvicorn exits."""
    yield
    if LIVE.pipeline is not None:
        LIVE.pipeline.stop = True
    t = LIVE.thread
    if t is not None and t.is_alive():
        await asyncio.to_thread(t.join, 10.0)


app = FastAPI(title="Bag counting", lifespan=lifespan)
OUT.mkdir(exist_ok=True)
app.mount("/out", StaticFiles(directory=OUT), name="out")


# -- recorded runs -----------------------------------------------------------
def _clip_meta() -> dict:
    if not CLIP_META.exists():
        return {}
    return {k: v for k, v in json.loads(CLIP_META.read_text()).items() if not k.startswith("_")}


def _truth_for(source: str) -> tuple[Path, float] | None:
    """Ground-truth CSV and its window length for a clip, by file name."""
    stem = Path(source).stem
    for p in sorted(TRUTH.glob(f"{stem}_first*s.csv")):
        m = re.search(r"first(\d+)s", p.name)
        if m:
            return p, float(m.group(1))
    return None


def _read_events(run: Path) -> list[dict]:
    p = run / "events.csv"
    if not p.exists():
        return []
    thumbs = {int(f.name[:4]): f.name for f in (run / "events").glob("*.jpg")}
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f):
            r = {k: (float(v) if k in ("time_s", "x", "y") else int(v) if v.lstrip("-").isdigit() else v)
                 for k, v in r.items()}
            r["thumb"] = thumbs.get(r["idx"])
            rows.append(r)
    return rows


def _run_info(run: Path, with_events: bool = False) -> dict | None:
    sp = run / "summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())
    stem = Path(s["source"]).stem
    wall_start = s.get("wall_start") or _clip_meta().get(stem, {}).get("wall_start")
    info = {
        "name": run.name,
        "source": s["source"],
        "clip": stem,
        "phase": _clip_meta().get(stem, {}).get("phase", ""),
        "frames": s["frames"],
        "fps": s["fps"],
        "duration_s": s["frames"] / s["fps"] if s["fps"] else 0,
        "loaded": s["loaded"],
        "unloaded": s["unloaded"],
        "on_truck": s["on_truck"],
        "video": (run / "annotated.mp4").exists(),
        "counting_line": s.get("counting_line"),
        "wall_start": wall_start,
        "modified": sp.stat().st_mtime,
    }
    events = _read_events(run)
    info["n_events"] = len(events)
    info["clock_start"] = events[0]["clock"] if events and events[0].get("clock") else None
    t = _truth_for(s["source"])
    if t:
        tp, window = t
        hi = min(window, info["duration_s"])
        r = evaluate(run / "events.csv", tp, tolerance_s=5.0, window=(0, hi))
        info["truth"] = {
            "file": tp.name, "window_s": hi, "true_loaded": r.true_loaded, "true_unloaded": r.true_unloaded,
            "pred_loaded": r.pred_loaded, "pred_unloaded": r.pred_unloaded,
            "tp": r.tp, "fp": r.fp, "fn": r.fn, "precision": r.precision, "recall": r.recall,
            "missed": r.unmatched_true, "false": r.unmatched_pred,
        }
    if with_events:
        info["events"] = events
    return info


@app.get("/api/runs")
def list_runs() -> list[dict]:
    runs = [i for d in OUT.iterdir() if d.is_dir() and (i := _run_info(d))]
    # Runs with video first, then newest first.
    return sorted(runs, key=lambda r: (not r["video"], -r["modified"]))


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict:
    info = _run_info(OUT / name, with_events=True)
    if info is None:
        raise HTTPException(404, "no such run")
    return info


@app.get("/api/sources")
def list_sources() -> list[dict]:
    meta = _clip_meta()
    out = []
    for p in sorted(CLIPS.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        m = meta.get(p.stem, {})
        out.append({"path": f"clips/{p.name}", "name": p.stem, "duration_s": n / fps if fps else 0,
                    "fps": fps, "wall_start": m.get("wall_start"), "phase": m.get("phase", "")})
    return out


@app.get("/api/weights")
def list_weights() -> list[str]:
    default = Config().detector.weights
    patterns = ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "*.pt", "*.onnx")
    found = sorted({
        str(p.relative_to(ROOT))
        for base in (ROOT / "runs" / "detect", ROOT / "exports")
        if base.is_dir()
        for pat in patterns
        for p in base.glob(f"*/{pat}")
    })
    return [default] + [w for w in found if w != default]


# -- live mode -----------------------------------------------------------------
class LiveState:
    """One live run at a time. The pipeline thread pushes JPEG frames and
    event rows here; HTTP handlers read them."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_cv = threading.Condition(self.lock)
        self.reset()

    def reset(self) -> None:
        self.jpeg: bytes | None = None
        self.frame_idx = -1
        self.t = 0.0
        self.counts = {"loaded": 0, "unloaded": 0, "on_truck": 0}
        self.events: list[dict] = []
        self.pipeline: Pipeline | None = None
        self.thread: threading.Thread | None = None
        self.run_name: str | None = None
        self.source: str | None = None
        self.started: float | None = None
        self.finished: float | None = None
        self.error: str | None = None
        self.proc_fps = 0.0
        self._fps_mark = (0, 0.0)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def status(self) -> dict:
        with self.lock:
            return {
                "running": self.running, "run": self.run_name, "source": self.source,
                "frame": self.frame_idx, "t": self.t, "proc_fps": self.proc_fps, **self.counts,
                "n_events": len(self.events), "error": self.error,
                "started": self.started, "finished": self.finished,
            }

    def on_frame(self, fi: int, t: float, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        now = time.time()
        with self.frame_cv:
            if ok:
                self.jpeg = buf.tobytes()
            self.frame_idx, self.t = fi, t
            c = self.pipeline.counter
            self.counts = {"loaded": c.loaded, "unloaded": c.unloaded, "on_truck": c.on_truck}
            m_fi, m_t = self._fps_mark
            if now - m_t >= 2.0:
                if m_t:
                    self.proc_fps = (fi - m_fi) / (now - m_t)
                self._fps_mark = (fi, now)
            self.frame_cv.notify_all()

    def on_event(self, row: dict) -> None:
        with self.lock:
            self.events.append(row)


LIVE = LiveState()


class LiveStart(BaseModel):
    source: str
    weights: str | None = None
    conf: float | None = None
    tracker_max_age: int | None = None
    line: str | None = None
    no_line: bool = False
    wall_start: str | None = None
    max_frames: int = 0
    start: float = 0.0
    duration: float = 0.0
    save_video: bool = False


@app.post("/api/live/start")
def live_start(req: LiveStart) -> dict:
    if LIVE.running:
        raise HTTPException(409, "a live run is already in progress")
    if req.start < 0 or req.duration < 0:
        raise HTTPException(400, "start and duration must be non-negative")
    if req.max_frames and req.duration:
        raise HTTPException(400, "use either max_frames or duration, not both")
    if req.tracker_max_age is not None and req.tracker_max_age < 0:
        raise HTTPException(400, "tracker_max_age must be non-negative")
    if req.no_line and req.line:
        raise HTTPException(400, "use either line or no_line, not both")
    src = req.source
    if not (src.startswith("rtsp://") or src.startswith("http") or (ROOT / src).exists()):
        raise HTTPException(400, f"source not found: {src}")
    cfg = Config.load(None)
    if req.weights:
        cfg.detector.weights = req.weights
    if req.conf is not None:
        cfg.detector.conf = req.conf
    if req.tracker_max_age is not None:
        cfg.detector.tracker_max_age = req.tracker_max_age
    if req.line:
        try:
            cfg.zones.cross_line = parse_line(req.line)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
    if req.no_line:
        cfg.counter.require_line_crossing = False
    ws = req.wall_start or _clip_meta().get(Path(src).stem, {}).get("wall_start")
    name = f"live_{datetime.now():%Y%m%d_%H%M%S}"
    with LIVE.lock:
        LIVE.reset()
        LIVE.run_name, LIVE.source, LIVE.started = name, src, time.time()
    try:
        p = Pipeline(cfg, OUT / name, save_video=req.save_video,
                     wall_start=datetime.fromisoformat(ws) if ws else None,
                     on_frame=LIVE.on_frame, on_event=LIVE.on_event)
    except Exception as e:  # bad weights path etc.
        LIVE.error = str(e)
        raise HTTPException(400, str(e))
    LIVE.pipeline = p

    def work() -> None:
        try:
            p.run(src, max_frames=req.max_frames, start_s=req.start, duration_s=req.duration)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            LIVE.error = str(e)
        finally:
            LIVE.finished = time.time()
            with LIVE.frame_cv:
                LIVE.frame_cv.notify_all()

    LIVE.thread = threading.Thread(target=work, name="bagcount-live", daemon=True)
    LIVE.thread.start()
    return {"run": name}


@app.post("/api/live/stop")
def live_stop() -> dict:
    if LIVE.pipeline is not None:
        LIVE.pipeline.stop = True
    t = LIVE.thread
    if t is not None and t.is_alive():
        t.join(timeout=10.0)  # wait for capture/writer/CSV cleanup
    return {"stopping": LIVE.running}


@app.get("/api/live/status")
def live_status() -> dict:
    return LIVE.status()


@app.get("/api/live/stream")
async def live_stream() -> StreamingResponse:
    """MJPEG stream of the latest annotated frame."""
    boundary = b"--frame"

    def next_frame(last: int) -> tuple[int, bytes | None]:
        with LIVE.frame_cv:
            LIVE.frame_cv.wait_for(lambda: LIVE.frame_idx != last or not LIVE.running, timeout=1.0)
            return LIVE.frame_idx, LIVE.jpeg

    async def gen():
        last = -2
        while True:
            fi, jpeg = await asyncio.to_thread(next_frame, last)
            if jpeg is not None and fi != last:
                last = fi
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                    str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            elif not LIVE.running:
                await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/live/events")
async def live_events() -> StreamingResponse:
    """Server-sent events: a `status` line every half second and an `event`
    line for every counted bag."""
    async def gen():
        sent = 0
        while True:
            with LIVE.lock:
                new = LIVE.events[sent:]
                sent = len(LIVE.events)
            for row in new:
                yield f"event: bag\ndata: {json.dumps(row)}\n\n"
            yield f"event: status\ndata: {json.dumps(LIVE.status())}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- page ------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.get("/static/{name}")
def static(name: str) -> FileResponse:
    p = STATIC / name
    if not p.exists() or ".." in name:
        raise HTTPException(404)
    return FileResponse(p)


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Bag counting demo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
