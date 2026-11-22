"""vox_v2/photo_lib.py — Self-maintaining photo library, keyframe archiver, and idle VLM indexer.

Replaces legacy frame_tagger.py + vision_ingest.py + port 8768 archiver.
Writes directly to vox_unified.db (vision_frames + memory_units) so search_memory
and inspect_past_frames see every new photo immediately with zero external DBs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("vox_v2.photo_lib")

DEFAULT_DB = os.getenv(
    "VOX_DB",
    os.path.expanduser("~/.openclaw/workspace/jarvis/voice_server/vox_unified.db"),
)
DEFAULT_ARCHIVE_ROOT = os.getenv(
    "VOX_FRAME_ARCHIVE",
    os.path.expanduser("~/.openclaw/workspace/jarvis/memory/voice-frames"),
)
DEFAULT_VISION_URL = os.getenv(
    "VOX_LLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
DEFAULT_VISION_MODEL = os.getenv("VOX_LLM_MODEL", "qwen2.5-7b-instruct")

TAG_PROMPT = (
    "用一句中文描述这张画面（60 字以内），再给 10 个检索关键词。"
    "严格输出两行，不要解释、不要引号：\n"
    "D: <画面描述>\n"
    "T: <词1,词2,...>"
)

# Dummy no-op tool to avoid Halogen :8731 502 on tools=None multimodal calls
_NOOP_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "do not call",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _gray_sig(raw_bytes: bytes) -> list[int] | None:
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw_bytes)).convert("L").resize((32, 24))
        return list(im.getdata())
    except Exception:
        return None


def _mad(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 999.0
    return sum(abs(x - y) for x, y in zip(a, b)) / float(len(a))


def _simhash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "big", signed=False)
    return val - (1 << 64) if val >= (1 << 63) else val


def parse_tag_reply(raw: str) -> tuple[str, str]:
    clean = (raw or "").replace("「", "").replace("」", "").replace("“", "").replace("”", "").strip()
    if not clean:
        return "", ""
    m_d = re.search(r"^D[:：]\s*(.+)", clean, re.M)
    m_t = re.search(r"^T[:：]\s*(.+)", clean, re.M)
    if m_d:
        return m_d.group(1).strip(), (m_t.group(1).strip() if m_t else "")
    return clean.splitlines()[0][:100].strip(), ""


def _server_busy() -> bool:
    """Ask 8766 /health: session THINKING/SPEAKING or spoke <30s ago -> yield."""
    try:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8766/health", timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return bool(data.get("busy") or data.get("recent_speech"))
    except Exception:
        return False


class PhotoLibrary:
    """Unified photo archive + background VLM tagger writing directly to vox_unified.db."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        archive_root: str = DEFAULT_ARCHIVE_ROOT,
        vision_url: str = DEFAULT_VISION_URL,
        vision_model: str = DEFAULT_VISION_MODEL,
        min_archive_interval_s: float = 15.0,
        mad_threshold: float = 12.0,
        force_archive_interval_s: float = 90.0,
        is_busy_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.db_path = db_path
        self.archive_root = Path(archive_root)
        self.vision_url = vision_url
        self.vision_model = vision_model
        self.min_archive_interval_s = min_archive_interval_s
        self.mad_threshold = mad_threshold
        self.force_archive_interval_s = force_archive_interval_s
        self.is_busy_fn = is_busy_fn or (lambda: False)
        self._last_sig: dict[str, tuple[float, list[int] | None]] = {}
        self._last_archive_ts: dict[str, float] = {}
        self.ring = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def maybe_archive_frame(
        self,
        client_id: str,
        raw_bytes: bytes,
        meta: dict[str, Any] | None = None,
        ts: float | None = None,
        force: bool = False,
    ) -> str | None:
        """关键帧归档（2026-11-22 节流版）。

        * vad 帧：3s 防抖，直接落盘
        * 其它帧：硬冷却 12s；到点时从环形池过去 12s 里挑最清晰的一张
        * 静止去重：dt < 120s 且灰度差 < 18 直接跳过
        """
        if not raw_bytes or len(raw_bytes) < 1024:
            return None
        now = ts if ts is not None else time.time()
        meta = meta or {}
        reason = str(meta.get("reason", "")).lower()
        last_any = self._last_archive_ts.get(client_id, 0.0)
        gap = now - last_any
        cool = 3.0 if reason == "vad" else 12.0
        if not force and gap < cool:
            return None

        payload = raw_bytes
        stamp = now
        if not force and reason != "vad" and getattr(self, "ring", None) is not None:
            best = self.ring.sharpest_between(client_id, now - 12.0, now)
            if best:
                payload = best["bytes"]
                stamp = best["ts"]
                meta = dict(meta)
                meta["fmt"] = "jpg" if best["path"].endswith(".jpg") else "webp"

        sig = _gray_sig(payload)
        prev_ts, prev_sig = self._last_sig.get(client_id, (0.0, None))
        diff = _mad(prev_sig, sig)
        if not force and prev_sig is not None and (stamp - prev_ts) < 120.0 and diff < 18.0:
            return None

        day = time.strftime("%Y-%m-%d", time.localtime(stamp))
        hms = time.strftime("%H%M%S", time.localtime(stamp))
        ms = int((stamp % 1) * 1000)
        fmt = str(meta.get("fmt", "webp")).lower()
        ext = "webp" if "webp" in fmt else "jpg"

        out_dir = self.archive_root / day / client_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("%s_%03d.%s" % (hms, ms, ext))
        out_path.write_bytes(payload)
        self._last_sig[client_id] = (stamp, sig)
        # 冷却用「落盘时刻」而不是帧时间戳：否则从环形池挑到 11 秒前的帧时，
        # 下一次调用的 gap 会被算成 11 秒，冷却被绕过（实测出现 0.4s 连发）。
        self._last_archive_ts[client_id] = time.time()

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO vision_frames(
                    frame_path, client_id, ts_start, ts_end, caption, tags,
                    sheet_path, model, lat, lon, acc, ingested_at
                ) VALUES (?, ?, ?, ?, '', '', '', 'pending', ?, ?, ?, ?)
                """,
                (
                    str(out_path),
                    client_id,
                    stamp,
                    stamp,
                    meta.get("lat"),
                    meta.get("lon"),
                    meta.get("acc"),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return str(out_path)

    def _call_vision_tagger(self, frame_path: str) -> tuple[str, str]:
        raw = Path(frame_path).read_bytes()
        mime = "image/webp" if frame_path.lower().endswith(".webp") else "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        body: dict[str, Any] = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TAG_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            "tools": _NOOP_TOOL,
            "tool_choice": "none",
            "max_tokens": 110,
            "temperature": 0.2,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self.vision_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return parse_tag_reply(content)

    def _upsert_vision_index(
        self,
        conn: sqlite3.Connection,
        frame_path: str,
        client_id: str,
        ts_start: float,
        ts_end: float,
        caption: str,
        tags: str,
        model: str,
        lat: float | None = None,
        lon: float | None = None,
        acc: float | None = None,
    ) -> None:
        now = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO vision_frames(
                frame_path, client_id, ts_start, ts_end, caption, tags,
                sheet_path, model, lat, lon, acc, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)
            """,
            (frame_path, client_id, ts_start, ts_end, caption, tags, model, lat, lon, acc, now),
        )
        src_key = f"cap:{frame_path}"
        entities = "  ".join(t.strip() for t in re.split(r"[,，\s]+", tags) if t.strip())
        struct_json = json.dumps({"src": "photo_lib", "path": frame_path}, ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_units(
                src_key, client_id, scope, kind, start_ts, end_ts,
                search_text, caption_clean, ocr_verified, text_verbatim,
                entities_str, scene_tag, objects_json, has_text, tag_model,
                struct_json, best_frame_path, best_frame_score,
                lat, lon, acc, confidence, hit_count, status, simhash64
            ) VALUES (
                ?, ?, 'user_shared', 'vision_episode', ?, ?,
                ?, ?, '', '',
                ?, 'unknown', '[]', 0, ?,
                ?, ?, 1.0,
                ?, ?, ?, 0.9, 1, 'active', ?
            )
            """,
            (
                src_key,
                client_id,
                ts_start,
                ts_end,
                caption,
                caption,
                entities,
                model,
                struct_json,
                frame_path,
                lat,
                lon,
                acc,
                _simhash64(caption),
            ),
        )

    def tag_pending(
        self,
        limit: int = 20,
        rate_s: float = 1.0,
        tagger_fn: Callable[[str], tuple[str, str]] | None = None,
        respect_busy: bool = True,
    ) -> int:
        """Tags pending archived frames (newest first) while yielding to live voice turns."""
        tag_fn = tagger_fn or self._call_vision_tagger
        conn = self._connect()
        done = 0
        try:
            rows = conn.execute(
                """
                SELECT frame_path, client_id, ts_start, ts_end, lat, lon, acc
                FROM vision_frames
                WHERE (caption = '' OR caption IS NULL)
                  AND frame_path NOT LIKE '/dev/shm/%'
                ORDER BY ts_start DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for r in rows:
                if respect_busy and (self.is_busy_fn() or _server_busy()):
                    log.info("Voice active; pausing background tagging")
                    break
                fp = r["frame_path"]
                if not os.path.isfile(fp):
                    continue
                try:
                    cap, tags = tag_fn(fp)
                except Exception as exc:
                    log.warning("Tag failed for %s: %s", os.path.basename(fp), exc)
                    continue
                if not cap:
                    continue
                self._upsert_vision_index(
                    conn,
                    frame_path=fp,
                    client_id=r["client_id"],
                    ts_start=r["ts_start"],
                    ts_end=r["ts_end"],
                    caption=cap,
                    tags=tags,
                    model=self.vision_model,
                    lat=r["lat"],
                    lon=r["lon"],
                    acc=r["acc"],
                )
                conn.commit()
                done += 1
                if rate_s > 0:
                    time.sleep(rate_s)
        finally:
            conn.close()
        return done

    def sync_legacy_frame_caps(self, legacy_db_path: str) -> int:
        """One-time catch-up: imports tagged rows from frame_caps.db.caps into vox_unified.db."""
        if not os.path.isfile(legacy_db_path):
            return 0
        src = sqlite3.connect(f"file:{legacy_db_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        dst = self._connect()
        imported = 0
        try:
            existing = {
                r[0]
                for r in dst.execute(
                    "SELECT best_frame_path FROM memory_units WHERE best_frame_path != ''"
                ).fetchall()
            }
            rows = src.execute(
                """
                SELECT path, client, ts, COALESCE(ts_end, ts) AS ts_end,
                       cap, COALESCE(tags, '') AS tags, COALESCE(model, '') AS model,
                       lat, lon, acc
                FROM caps
                WHERE cap IS NOT NULL AND cap != ''
                ORDER BY ts DESC
                """
            ).fetchall()
            for r in rows:
                p = r["path"]
                if not p or p in existing:
                    continue
                self._upsert_vision_index(
                    dst,
                    frame_path=p,
                    client_id=r["client"] or "voxx2",
                    ts_start=float(r["ts"]),
                    ts_end=float(r["ts_end"]),
                    caption=r["cap"],
                    tags=r["tags"],
                    model=r["model"] or "legacy_caps",
                    lat=r["lat"],
                    lon=r["lon"],
                    acc=r["acc"],
                )
                imported += 1
            dst.commit()
        finally:
            src.close()
            dst.close()
        return imported


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="vox_v2 self-maintaining photo library")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--sync-legacy-caps", default="", help="Path to legacy frame_caps.db to import")
    ap.add_argument("--tag-pending", type=int, default=0, help="Tag up to N pending frames")
    ap.add_argument("--rate", type=float, default=1.0)
    args = ap.parse_args()

    lib = PhotoLibrary(db_path=args.db, archive_root=args.archive_root)
    if args.sync_legacy_caps:
        n = lib.sync_legacy_frame_caps(args.sync_legacy_caps)
        print(f"Synced {n} legacy rows from {args.sync_legacy_caps} into {args.db}")
    if args.tag_pending > 0:
        n = lib.tag_pending(limit=args.tag_pending, rate_s=args.rate)
        print(f"Tagged {n} pending frames into {args.db}")


if __name__ == "__main__":
    main()
