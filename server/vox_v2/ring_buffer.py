"""60s in-memory ring buffer for instant look-back (2026-11-22).

All FT_VISION_UP frames land here first: last ~60s / max 180 frames, kept in
/dev/shm so it never touches disk or the VLM tagging queue. A stable
latest.jpg (or .webp) is kept for realtime navigation consumers.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

DEFAULT_ROOT = os.environ.get("VOX_RING_DIR", "/dev/shm/vox-ring")
KEEP_S = float(os.environ.get("VOX_RING_KEEP_S", "60"))
MAX_FRAMES = int(os.environ.get("VOX_RING_MAX", "180"))


def _sharpness(raw: bytes) -> float:
    """Laplacian variance; falls back to gradient energy. Higher = crisper."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            g = im.convert("L")
            if g.width > 480:
                ratio = 480.0 / g.width
                g = g.resize((480, max(1, int(g.height * ratio))))
            a = np.asarray(g, dtype=np.float32)
        if a.size < 16:
            return 0.0
        lap = (
            -4.0 * a[1:-1, 1:-1]
            + a[:-2, 1:-1]
            + a[2:, 1:-1]
            + a[1:-1, :-2]
            + a[1:-1, 2:]
        )
        return float(lap.var())
    except Exception:
        return 0.0


class RingBuffer:
    def __init__(self, root: str = DEFAULT_ROOT, keep_s: float = KEEP_S,
                 max_frames: int = MAX_FRAMES) -> None:
        self.root = Path(root)
        self.keep_s = keep_s
        self.max_frames = max_frames

    def _cdir(self, client_id: str) -> Path:
        d = self.root / client_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, client_id: str, raw: bytes, meta: dict | None = None,
            ts: float | None = None) -> str | None:
        if not raw or len(raw) < 1024:
            return None
        meta = meta or {}
        now = ts if ts is not None else time.time()
        fmt = str(meta.get("fmt", "jpeg")).lower()
        ext = "webp" if "webp" in fmt else "jpg"
        d = self._cdir(client_id)
        snap = d / ("f_%d_%03d.%s" % (int(now), int((now % 1) * 1000), ext))
        try:
            snap.write_bytes(raw)
            latest = d / ("latest." + ext)
            tmp = d / (".latest." + ext + ".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, latest)
            # 另一种格式的旧 latest 清掉，避免 /frame/last 读到过期格式
            other = d / ("latest." + ("jpg" if ext == "webp" else "webp"))
            if other.exists():
                try:
                    other.unlink()
                except OSError:
                    pass
        except Exception:
            return None
        sharp = meta.get("sharpness")
        try:
            sharp = float(sharp) if sharp is not None else None
        except (TypeError, ValueError):
            sharp = None
        self._prune(d, now, sharp_path=(snap, sharp))
        return str(snap)

    def _prune(self, d: Path, now: float, sharp_path=None) -> None:
        try:
            snaps = sorted(d.glob("f_*.*"), key=lambda p: p.stat().st_mtime)
        except Exception:
            return
        drop = []
        for p in snaps:
            try:
                if now - p.stat().st_mtime > self.keep_s:
                    drop.append(p)
            except OSError:
                continue
        if len(snaps) - len(drop) > self.max_frames:
            drop.extend(snaps[: max(0, len(snaps) - len(drop) - self.max_frames)])
        for p in drop:
            try:
                p.unlink()
            except OSError:
                pass

    def recent(self, client_id: str, seconds_ago_start: float = 15.0,
               seconds_ago_end: float = 0.0, count: int = 3) -> list[dict]:
        """Return the sharpest frames inside [now-start, now-end]."""
        d = self.root / client_id
        if not d.exists():
            return []
        now = time.time()
        lo = now - max(seconds_ago_start, seconds_ago_end)
        hi = now - min(seconds_ago_start, seconds_ago_end)
        rows = []
        for p in d.glob("f_*.*"):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if lo <= mt <= hi:
                rows.append((mt, p))
        if not rows:
            return []
        rows.sort()
        scored = []
        for mt, p in rows:
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            scored.append({"ts": mt, "path": str(p), "bytes": raw,
                           "sharpness": _sharpness(raw)})
        scored.sort(key=lambda r: r["sharpness"], reverse=True)
        # 同分/相近画面里优先时间靠近窗口末端（更贴近"刚刚"）
        picked = scored[: max(1, min(int(count), 3))]
        picked.sort(key=lambda r: r["ts"])
        return picked

    def sharpest_between(self, client_id: str, start_ts: float,
                         end_ts: float) -> dict | None:
        d = self.root / client_id
        if not d.exists():
            return None
        best = None
        for p in d.glob("f_*.*"):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if not (start_ts <= mt <= end_ts):
                continue
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            s = _sharpness(raw)
            if best is None or s > best["sharpness"]:
                best = {"ts": mt, "path": str(p), "bytes": raw, "sharpness": s}
        return best

    def latest_path(self, client_id: str) -> str | None:
        d = self.root / client_id
        for ext in ("jpg", "webp"):
            p = d / ("latest." + ext)
            if p.exists():
                return str(p)
        return None
