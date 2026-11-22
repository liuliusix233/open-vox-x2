"""Unified vision frame store with strict freshness TTL and zero 404 split-brain."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass
class FrameRecord:
    path: str
    mime: str
    ts: float
    size_bytes: int
    meta: dict[str, Any] = field(default_factory=dict)
    caption: str = ""


class VisionStore:
    """Single source of truth for incoming 0x03 frames and GET /frame/last."""

    def __init__(self, frame_dir: str, ttl_s: int = 300) -> None:
        self.frame_dir = Path(frame_dir)
        self.ttl_s = ttl_s
        self._latest: dict[str, FrameRecord] = {}
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, client_id: str, raw_bytes: bytes, meta: dict[str, Any]) -> FrameRecord:
        now = time.time()
        fmt = str(meta.get("fmt", "webp")).lower()
        ext = "webp" if "webp" in fmt else "jpg"
        mime = "image/webp" if ext == "webp" else "image/jpeg"

        cdir = self.frame_dir / client_id
        cdir.mkdir(parents=True, exist_ok=True)

        # Atomic write for latest.<ext> + timestamped snapshot
        ts_name = f"frame_{int(now * 1000)}.{ext}"
        snap_path = cdir / ts_name
        latest_path = cdir / f"latest.{ext}"
        tmp_path = cdir / f".latest.{ext}.tmp"

        snap_path.write_bytes(raw_bytes)
        tmp_path.write_bytes(raw_bytes)
        os.replace(tmp_path, latest_path)

        rec = FrameRecord(
            path=str(latest_path),
            mime=mime,
            ts=now,
            size_bytes=len(raw_bytes),
            meta=dict(meta),
        )
        self._latest[client_id] = rec
        self._prune_old(cdir, keep=40)
        return rec

    def _prune_old(self, cdir: Path, keep: int = 40) -> None:
        snaps = sorted(cdir.glob("frame_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in snaps[keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    def get_latest(self, client_id: str) -> FrameRecord | None:
        rec = self._latest.get(client_id)
        if rec and Path(rec.path).exists():
            return rec
        # Fallback to disk after restart so /frame/last never spuriously 404s
        cdir = self.frame_dir / client_id
        if not cdir.exists():
            return None
        for ext, mime in (("webp", "image/webp"), ("jpg", "image/jpeg")):
            p = cdir / f"latest.{ext}"
            if p.exists():
                st = p.stat()
                rec = FrameRecord(path=str(p), mime=mime, ts=st.st_mtime, size_bytes=st.st_size)
                self._latest[client_id] = rec
                return rec
        return None

    def get_latest_bytes(self, client_id: str) -> tuple[bytes, str] | None:
        rec = self.get_latest(client_id)
        if not rec:
            return None
        p = Path(rec.path)
        if not p.exists():
            return None
        return p.read_bytes(), rec.mime

    def get_fresh_image_b64(self, client_id: str) -> tuple[str, str] | None:
        rec = self.get_latest(client_id)
        if not rec or (time.time() - rec.ts) > self.ttl_s:
            return None
        p = Path(rec.path)
        if not p.exists():
            return None
        return rec.mime, base64.b64encode(p.read_bytes()).decode("ascii")

    def format_live_view_header(self, client_id: str) -> str:
        rec = self.get_latest(client_id)
        if not rec:
            return "[LIVE_VIEW | 当前无眼镜画面]"
        age = max(0, int(time.time() - rec.ts))
        ts_str = datetime.fromtimestamp(rec.ts, tz=TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M")
        if age > self.ttl_s:
            mins = age // 60
            return f"[LIVE_VIEW | 最近画面已过期（{ts_str}，约 {mins} 分钟前），严禁当作当前实时画面]"
        w = rec.meta.get("w", "?")
        h = rec.meta.get("h", "?")
        extra = f"；摘要：{rec.caption}" if rec.caption else "；如需查看细节请调用 describe_live_view 工具"
        return f"[LIVE_VIEW | 当前实时画面（{age} 秒前，{w}x{h}）| 严禁当作历史记录{extra}]"
