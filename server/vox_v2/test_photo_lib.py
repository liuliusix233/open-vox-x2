"""Automated smoke tests for vox_v2/photo_lib.py."""

from __future__ import annotations

import io
import sqlite3
import tempfile
import time
from pathlib import Path

from PIL import Image

try:
    from .photo_lib import PhotoLibrary
except ImportError:  # 直接跑脚本时（python3 test_photo_lib.py）
    from photo_lib import PhotoLibrary


def _make_img(color: tuple[int, int, int], size: tuple[int, int] = (320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _init_unified_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE vision_frames (
            frame_path TEXT PRIMARY KEY, client_id TEXT NOT NULL,
            ts_start REAL NOT NULL, ts_end REAL NOT NULL,
            caption TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',
            sheet_path TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
            lat REAL, lon REAL, acc REAL, ingested_at REAL NOT NULL
        );
        CREATE TABLE memory_units (
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT, src_key TEXT UNIQUE,
            client_id TEXT NOT NULL, scope TEXT NOT NULL, kind TEXT NOT NULL,
            start_ts REAL NOT NULL, end_ts REAL NOT NULL,
            search_text TEXT NOT NULL, caption_clean TEXT NOT NULL,
            ocr_verified TEXT NOT NULL, text_verbatim TEXT NOT NULL,
            entities_str TEXT NOT NULL, scene_tag TEXT NOT NULL,
            objects_json TEXT NOT NULL, has_text INTEGER NOT NULL,
            tag_model TEXT NOT NULL, struct_json TEXT NOT NULL,
            best_frame_path TEXT NOT NULL, best_frame_score REAL NOT NULL,
            lat REAL, lon REAL, acc REAL, confidence REAL NOT NULL,
            hit_count INTEGER NOT NULL, status TEXT NOT NULL, simhash64 INTEGER NOT NULL
        );
        """
    )
    conn.close()


def run_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_photolib_test_") as tmp:
        tdir = Path(tmp)
        db = str(tdir / "vox_unified.db")
        legacy = str(tdir / "frame_caps.db")
        arch = str(tdir / "voice-frames")
        _init_unified_db(db)

        # 1. Legacy frame_caps.db catch-up sync
        lconn = sqlite3.connect(legacy)
        lconn.execute(
            "CREATE TABLE caps(path TEXT PRIMARY KEY, client TEXT, ts REAL, cap TEXT, tags TEXT, model TEXT, at REAL, ts_end REAL, sheet TEXT, lat REAL, lon REAL, acc REAL)"
        )
        lconn.execute(
            "INSERT INTO caps VALUES ('/tmp/old.jpg','voxx2',1795336779.0,'双屏桌面配MIDI键盘','桌面,MIDI','qwen3.8',1795336800.0,1795336790.0,'',NULL,NULL,NULL)"
        )
        lconn.commit()
        lconn.close()

        lib = PhotoLibrary(db_path=db, archive_root=arch, min_archive_interval_s=10.0)
        synced = lib.sync_legacy_frame_caps(legacy)
        assert synced == 1, f"Expected 1 synced row, got {synced}"
        assert lib.sync_legacy_frame_caps(legacy) == 0, "Idempotency check failed"

        # 2. Keyframe archive + dedup
        img_dark = _make_img((20, 20, 40))
        img_bright = _make_img((220, 200, 180))
        now = time.time()
        p1 = lib.maybe_archive_frame("voxx2", img_dark, {"fmt": "jpg"}, ts=now)
        assert p1 is not None and Path(p1).is_file()
        # Same frame 2 seconds later should be deduplicated
        p2 = lib.maybe_archive_frame("voxx2", img_dark, {"fmt": "jpg"}, ts=now + 2)
        assert p2 is None, "Duplicate frame should have been skipped"
        # 2026-11-22 节流契约：非 vad 帧有 12s 硬冷却，场景再大也不越级
        p3 = lib.maybe_archive_frame("voxx2", img_bright, {"fmt": "jpg"}, ts=now + 2)
        assert p3 is None, "12s hard cooldown must block non-vad frames"
        # 冷却过去后才允许归档新画面
        p3b = lib.maybe_archive_frame("voxx2", img_bright, {"fmt": "jpg"}, ts=now + 15)
        assert p3b is not None and Path(p3b).is_file()

        # 3. Background tagging writes BOTH vision_frames AND memory_units
        def mock_tagger(fp: str) -> tuple[str, str]:
            return ("测试场景：机械键盘与双屏显示器", "机械键盘,双屏,桌面")

        tagged = lib.tag_pending(limit=10, rate_s=0, tagger_fn=mock_tagger, respect_busy=False)
        assert tagged == 2, f"Expected 2 tagged frames, got {tagged}"

        conn = sqlite3.connect(db)
        mu_count = conn.execute("SELECT COUNT(*) FROM memory_units").fetchone()[0]
        vf_tagged = conn.execute("SELECT COUNT(*) FROM vision_frames WHERE caption != ''").fetchone()[0]
        conn.close()
        assert mu_count == 3, f"Expected 3 memory_units (1 legacy + 2 new), got {mu_count}"
        assert vf_tagged == 3, f"Expected 3 tagged vision_frames, got {vf_tagged}"
        print("ALL photo_lib SMOKE TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
