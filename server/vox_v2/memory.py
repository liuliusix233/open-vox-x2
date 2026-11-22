"""Single-database (vox_unified.db) memory store with strict client_id isolation."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

TZ_SHANGHAI = timezone(timedelta(hours=8))

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sys_meta (
    key   TEXT PRIMARY KEY,
    val   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL DEFAULT 'voxx2',
    ts          REAL NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system','ambient')),
    text        TEXT NOT NULL,
    epoch_id    INTEGER DEFAULT 0,
    audio_dur_s REAL DEFAULT 0.0,
    spk_sim     REAL,
    is_poisoned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turns_client_ts ON turns(client_id, ts DESC);

CREATE TABLE IF NOT EXISTS vision_frames (
    frame_path  TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL DEFAULT 'voxx2',
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    caption     TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '',
    sheet_path  TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    lat         REAL,
    lon         REAL,
    acc         REAL,
    ingested_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vf_client_ts ON vision_frames(client_id, ts_start DESC);

CREATE TABLE IF NOT EXISTS memory_units (
    unit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_key          TEXT UNIQUE,
    client_id        TEXT NOT NULL DEFAULT 'voxx2',
    scope            TEXT NOT NULL DEFAULT 'user_shared',
    kind             TEXT NOT NULL CHECK(kind IN ('vision_episode','dialog_turn','semantic_fact','event_anchor')),
    start_ts         REAL NOT NULL,
    end_ts           REAL NOT NULL,
    search_text      TEXT NOT NULL,
    caption_clean    TEXT NOT NULL DEFAULT '',
    ocr_verified     TEXT NOT NULL DEFAULT '',
    text_verbatim    TEXT NOT NULL DEFAULT '',
    entities_str     TEXT NOT NULL DEFAULT '',
    scene_tag        TEXT NOT NULL DEFAULT 'unknown',
    objects_json     TEXT NOT NULL DEFAULT '[]',
    has_text         INTEGER NOT NULL DEFAULT 0,
    tag_model        TEXT NOT NULL DEFAULT '',
    struct_json      TEXT NOT NULL DEFAULT '{}',
    best_frame_path  TEXT NOT NULL DEFAULT '',
    best_frame_score REAL NOT NULL DEFAULT 0.0,
    lat              REAL,
    lon              REAL,
    acc              REAL,
    confidence       REAL NOT NULL DEFAULT 0.5,
    hit_count        INTEGER NOT NULL DEFAULT 1,
    status           TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','retracted')),
    simhash64        INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB
);
CREATE INDEX IF NOT EXISTS idx_mu_time ON memory_units(client_id, start_ts DESC) WHERE status='active';

CREATE TABLE IF NOT EXISTS known_places (
    place_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id     TEXT NOT NULL DEFAULT 'voxx2',
    name          TEXT NOT NULL,
    lat_gcj       REAL NOT NULL,
    lon_gcj       REAL NOT NULL,
    radius_m      REAL NOT NULL DEFAULT 100,
    visit_count   INTEGER NOT NULL DEFAULT 1,
    last_visit_ts REAL NOT NULL,
    UNIQUE(client_id, name)
);
CREATE INDEX IF NOT EXISTS idx_places_client ON known_places(client_id, last_visit_ts DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    toks,
    content='',
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS semantic_facts (
    fact_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id        TEXT NOT NULL DEFAULT 'voxx2',
    subject          TEXT NOT NULL DEFAULT 'user',
    predicate        TEXT NOT NULL,
    cardinality      TEXT NOT NULL DEFAULT 'multi' CHECK(cardinality IN ('single','multi')),
    object_text      TEXT NOT NULL,
    norm_object      TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','retracted')),
    source           TEXT NOT NULL DEFAULT 'user_explicit',
    evidence_turn_id INTEGER,
    confidence       REAL NOT NULL DEFAULT 0.8,
    hit_count        INTEGER NOT NULL DEFAULT 1,
    first_event_ts   REAL NOT NULL,
    last_verified_ts REAL NOT NULL,
    event_ts         REAL,
    expires_at       REAL,
    superseded_by    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sf_active ON semantic_facts(client_id, status, predicate);

CREATE TABLE IF NOT EXISTS directives (
    dir_id      TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL DEFAULT 'voxx2',
    rule_text   TEXT NOT NULL,
    strength    REAL NOT NULL DEFAULT 1.0,
    created_ts  REAL NOT NULL,
    updated_ts  REAL NOT NULL,
    expires_at  REAL,
    hits        INTEGER NOT NULL DEFAULT 1,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS open_loops (
    loop_id       TEXT PRIMARY KEY,
    client_id     TEXT NOT NULL DEFAULT 'voxx2',
    kind          TEXT NOT NULL CHECK(kind IN ('follow_up','self_share','curiosity','watch')),
    topic         TEXT NOT NULL,
    goal          TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'watching' CHECK(status IN ('watching','pending_share','closed','dismissed','expired')),
    due_after     REAL NOT NULL DEFAULT 0,
    expires_at    REAL NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0,
    max_mentions  INTEGER NOT NULL DEFAULT 2,
    created_ts    REAL NOT NULL,
    updated_ts    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    client_id  TEXT NOT NULL DEFAULT 'voxx2',
    title      TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '{}',
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
"""


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=TZ_SHANGHAI).strftime("%m-%d %H:%M")


class MemoryStore:
    """Thread-safe single-source SQLite store for vox_unified.db."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            # Ensure event_ts exists on migrated semantic_facts tables
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()}
            if "event_ts" not in cols:
                self._conn.execute("ALTER TABLE semantic_facts ADD COLUMN event_ts REAL")
            self._conn.execute(
                "INSERT OR IGNORE INTO sys_meta(key, val) VALUES ('epoch_high_water', '10000')"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def next_epoch(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT val FROM sys_meta WHERE key='epoch_high_water'"
            ).fetchone()
            cur = int(row["val"]) if row else 10000
            nxt = cur + 1
            self._conn.execute(
                "INSERT OR REPLACE INTO sys_meta(key, val) VALUES ('epoch_high_water', ?)",
                (str(nxt),),
            )
            self._conn.commit()
            return nxt

    def record_turn(
        self,
        client_id: str,
        role: str,
        text: str,
        epoch_id: int = 0,
        audio_dur_s: float = 0.0,
        spk_sim: float = 0.0,
        is_poisoned: int = 0,
    ) -> int:
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO turns(client_id, ts, role, text, epoch_id, audio_dur_s, spk_sim, is_poisoned)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (client_id, now, role, text.strip(), epoch_id, audio_dur_s, spk_sim, is_poisoned),
                )
            except Exception:
                cur = self._conn.execute(
                    """
                    INSERT INTO turns(client_id, ts, role, text, epoch_id, audio_dur_s)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (client_id, now, role, text.strip(), epoch_id, audio_dur_s),
                )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_recent_turns(self, client_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, text, ts FROM turns
                WHERE client_id = ? AND is_poisoned = 0 AND role IN ('user', 'assistant')
                ORDER BY ts DESC LIMIT ?
                """,
                (client_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_known_facts(self, client_id: str, limit: int = 12) -> list[str]:
        with self._lock:
            facts = self._conn.execute(
                """
                SELECT predicate, object_text FROM semantic_facts
                WHERE client_id = ? AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY confidence DESC, last_verified_ts DESC
                LIMIT ?
                """,
                (client_id, time.time(), limit),
            ).fetchall()
            dirs = self._conn.execute(
                """
                SELECT rule_text FROM directives
                WHERE client_id = ? AND is_active = 1
                ORDER BY strength DESC, updated_ts DESC LIMIT 6
                """,
                (client_id,),
            ).fetchall()
        out = [f"{r['predicate']}: {r['object_text']}" for r in facts]
        out.extend(f"偏好/规则: {r['rule_text']}" for r in dirs)
        return out

    def save_fact(
        self,
        client_id: str,
        predicate: str,
        object_text: str,
        event_ts: float | None = None,
        cardinality: str = "single",
        source: str = "user_explicit",
        confidence: float = 0.95,
    ) -> int:
        now = time.time()
        norm = object_text.strip().lower()
        with self._lock:
            if cardinality == "single":
                self._conn.execute(
                    """
                    UPDATE semantic_facts SET status='superseded'
                    WHERE client_id=? AND predicate=? AND status='active'
                    """,
                    (client_id, predicate),
                )
            cur = self._conn.execute(
                """
                INSERT INTO semantic_facts(
                    client_id, subject, predicate, cardinality, object_text, norm_object,
                    status, source, confidence, first_event_ts, last_verified_ts, event_ts
                ) VALUES (?, 'user', ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    predicate,
                    cardinality,
                    object_text.strip(),
                    norm,
                    source,
                    confidence,
                    now,
                    now,
                    event_ts if event_ts is not None else now,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_reminder(self, client_id: str, title: str, due_str: str) -> str:
        now = time.time()
        job_id = f"rem-{uuid.uuid4().hex[:8]}"
        facts = json.dumps({"due_str": due_str}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs(job_id, client_id, title, status, facts_json, created_ts, updated_ts)
                VALUES (?, ?, ?, 'SCHEDULED', ?, ?, ?)
                """,
                (job_id, client_id, title.strip(), facts, now, now),
            )
            self._conn.commit()
        return job_id

    def save_vision_frame(
        self,
        client_id: str,
        frame_path: str,
        ts: float,
        caption: str = "",
        lat: float | None = None,
        lon: float | None = None,
        acc: float | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vision_frames(
                    frame_path, client_id, ts_start, ts_end, caption, lat, lon, acc, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (frame_path, client_id, ts, ts, caption, lat, lon, acc, time.time()),
            )
            self._conn.commit()

    def _vis_sql(self, client_id, query, start_ts, end_ts, direction, limit,
                 near_lat=None, near_lon=None, near_radius_m=150.0, offset=0):
        sql = """
            SELECT kind, start_ts, end_ts, search_text, caption_clean, ocr_verified, best_frame_path
            FROM memory_units
            WHERE client_id = ? AND status = 'active'
              AND kind IN ('vision_episode', 'event_anchor')
        """
        params = [client_id]
        if start_ts is not None:
            sql += " AND MIN(end_ts, start_ts + 600) >= ?"  # 老数据有 end_ts 跨天的脏行，封顶 10 分钟
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND start_ts <= ?"
            params.append(end_ts)
        if query:
            sql += " AND (search_text LIKE ? OR caption_clean LIKE ? OR ocr_verified LIKE ?)"
            like = "%" + query + "%"
            params.extend([like, like, like])
        if near_lat is not None and near_lon is not None:
            # 先做包围盒粗筛（走 SQLite 索引外的简单比较），距离精度交给上层再核
            import math as _m
            dlat = near_radius_m / 111000.0
            dlon = near_radius_m / (111000.0 * max(0.2, _m.cos(_m.radians(near_lat))))
            sql += " AND lat IS NOT NULL AND lon IS NOT NULL AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
            params.extend([near_lat - dlat, near_lat + dlat,
                           near_lon - dlon, near_lon + dlon])
        sql += " ORDER BY start_ts " + direction + " LIMIT ?"
        params.append(limit)
        if offset:
            sql += " OFFSET ?"
            params.append(offset)
        return sql, params

    # ── 人话地点记忆 ─────────────────────────────────────────

    def save_place(self, client_id: str, name: str, lat: float, lon: float,
                   radius_m: float = 100.0) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "reason": "name 不能为空"}
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT place_id FROM known_places WHERE client_id=? AND name=?",
                (client_id, name),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE known_places SET lat_gcj=?, lon_gcj=?, radius_m=?, "
                    "visit_count=visit_count+1, last_visit_ts=? WHERE place_id=?",
                    (lat, lon, radius_m, now, row["place_id"]),
                )
                place_id = row["place_id"]
            else:
                cur = self._conn.execute(
                    "INSERT INTO known_places(client_id,name,lat_gcj,lon_gcj,radius_m,visit_count,last_visit_ts)"
                    " VALUES (?,?,?,?,?,1,?)",
                    (client_id, name, lat, lon, radius_m, now),
                )
                place_id = cur.lastrowid
            self._conn.commit()
        return {"ok": True, "place_id": place_id, "name": name}

    def match_place(self, client_id: str, lat: float, lon: float) -> dict[str, Any] | None:
        """返回半径内最近的已知地点；命中时累加 visit_count。"""
        from .geo import haversine_m
        with self._lock:
            rows = self._conn.execute(
                "SELECT place_id,name,lat_gcj,lon_gcj,radius_m FROM known_places WHERE client_id=?",
                (client_id,),
            ).fetchall()
        best = None
        best_d = 1e18
        for r in rows:
            d = haversine_m(lat, lon, float(r["lat_gcj"]), float(r["lon_gcj"]))
            if d <= float(r["radius_m"]) and d < best_d:
                best, best_d = r, d
        if best is None:
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE known_places SET visit_count=visit_count+1, last_visit_ts=? WHERE place_id=?",
                (time.time(), best["place_id"]),
            )
            self._conn.commit()
        return {"name": best["name"], "distance_m": round(best_d, 1),
                "place_id": best["place_id"]}

    def list_places(self, client_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name,lat_gcj,lon_gcj,radius_m,visit_count,last_visit_ts "
                "FROM known_places WHERE client_id=? ORDER BY last_visit_ts DESC LIMIT ?",
                (client_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_place(self, client_id: str, name: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT name,lat_gcj,lon_gcj,radius_m FROM known_places WHERE client_id=? AND name=?",
                (client_id, name),
            ).fetchone()
        return dict(r) if r else None

    def latest_vision_caption(self, client_id: str) -> str:
        """最近一条非空视觉摘要（给主动心跳当视野用）。"""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(NULLIF(caption_clean, ''), search_text) AS cap
                FROM memory_units
                WHERE client_id = ? AND kind = 'vision_episode' AND status = 'active'
                ORDER BY start_ts DESC LIMIT 1
                """,
                (client_id,),
            ).fetchone()
        return str(row["cap"] or "") if row else ""

    def search_memory(
        self,
        client_id: str,
        query: str = "",
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int = 16,
        order: str = "desc",
        kind: str = "all",
        near_lat: float | None = None,
        near_lon: float | None = None,
        near_radius_m: float = 150.0,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Strict client-scoped search across memory_units, semantic_facts, and user turns.

        2026-11-22 结构升级（机主指令）：
          * order: asc/desc —— 问"最早/几点到"必须能正序看
          * kind:  all/vision/dialog —— 只看视觉转场或只看聊天
          * 视觉帧与对话分开限额：聊天记录不再把 vision_episode 挤掉
          * 返回 total_matched / earliest_time / latest_time —— 分页截断不等于"没有更早记录"
        """
        direction = "ASC" if str(order).lower() == "asc" else "DESC"
        kind = str(kind or "all").lower()
        if kind not in ("all", "vision", "dialog"):
            kind = "all"
        want_vision = kind in ("all", "vision")
        want_dialog = kind in ("all", "dialog")
        q = (query or "").strip()
        limit = max(1, min(int(limit or 16), 64))
        offset = max(0, min(int(offset or 0), 2000))

        hits: list[dict[str, Any]] = []
        frame_paths: list[str] = []

        with self._lock:
            total_matched = 0
            earliest = None
            latest = None

            # ── 视觉单元（含 vision_episode / event_anchor / dialog_turn 单元）
            if want_vision:
                sql, params = self._vis_sql(
                    client_id, q, start_ts, end_ts, direction, limit,
                    near_lat=near_lat, near_lon=near_lon, near_radius_m=near_radius_m,
                    offset=offset,
                )
                rows = self._conn.execute(sql, params).fetchall()
                for r in rows:
                    body = r["caption_clean"] or r["search_text"]
                    if r["ocr_verified"]:
                        body = body + " [OCR: " + r["ocr_verified"] + "]"
                    fp = r["best_frame_path"] or ""
                    hits.append(
                        {
                            "kind": r["kind"],
                            "time": _fmt_ts(r["start_ts"]),
                            "text": body,
                            "frame_path": fp,
                        }
                    )
                    if fp and len(frame_paths) < 4 and os.path.isfile(fp) and fp not in frame_paths:
                        frame_paths.append(fp)

                # 日级 event_anchor 单独补一轮：它们只有 6 条/天，容易被 episode 挤出 limit，
                # 而且问「某天/某段时间在干嘛」时正是最该被看到的那层摘要。
                a_sql = (
                    "SELECT start_ts, search_text FROM memory_units "
                    "WHERE client_id=? AND kind='event_anchor' AND status='active'"
                )
                a_params: list[Any] = [client_id]
                if start_ts is not None:
                    a_sql += " AND end_ts >= ?"
                    a_params.append(start_ts)
                if end_ts is not None:
                    a_sql += " AND start_ts <= ?"
                    a_params.append(end_ts)
                if q:
                    a_sql += " AND search_text LIKE ?"
                    a_params.append("%" + q + "%")
                a_sql += " ORDER BY start_ts " + direction + " LIMIT 6"
                for r in self._conn.execute(a_sql, a_params).fetchall():
                    hits.append(
                        {"kind": "event_anchor", "time": _fmt_ts(r["start_ts"]),
                         "text": r["search_text"]}
                    )

                stat = self._vis_stats(client_id, q, start_ts, end_ts)
                total_matched += stat[0]
                if stat[1] is not None:
                    earliest = stat[1] if earliest is None else min(earliest, stat[1])
                    latest = stat[2] if latest is None else max(latest, stat[2])

            # ── 长期事实（只有 kind=all 才带，避免污染纯视觉/纯对话检索）
            sf_sql = """
                SELECT predicate, object_text, COALESCE(event_ts, first_event_ts) AS ets
                FROM semantic_facts
                WHERE client_id = ? AND status = 'active'
            """
            sf_params: list[Any] = [client_id]
            if start_ts is not None:
                sf_sql += " AND COALESCE(event_ts, first_event_ts) >= ?"
                sf_params.append(start_ts)
            if end_ts is not None:
                sf_sql += " AND COALESCE(event_ts, first_event_ts) <= ?"
                sf_params.append(end_ts)
            if q:
                sf_sql += " AND (predicate LIKE ? OR object_text LIKE ?)"
                like = "%" + q + "%"
                sf_params.extend([like, like])
            sf_sql += " ORDER BY ets " + direction + " LIMIT ?"
            sf_params.append(max(2, limit // 2))
            _sf_rows = self._conn.execute(sf_sql, sf_params).fetchall() if kind == "all" else []
            for r in _sf_rows:
                hits.append(
                    {
                        "kind": "fact",
                        "time": _fmt_ts(r["ets"]),
                        "text": r["predicate"] + ": " + r["object_text"],
                    }
                )

            # ── 聊天记录：独立配额，最多占一半名额
            if want_dialog:
                t_sql = """
                    SELECT ts, text FROM turns
                    WHERE client_id = ? AND role = 'user' AND is_poisoned = 0
                """
                t_params: list[Any] = [client_id]
                if start_ts is not None:
                    t_sql += " AND ts >= ?"
                    t_params.append(start_ts)
                if end_ts is not None:
                    t_sql += " AND ts <= ?"
                    t_params.append(end_ts)
                if q:
                    t_sql += " AND text LIKE ?"
                    t_params.append("%" + q + "%")
                t_sql += " ORDER BY ts " + direction + " LIMIT ?"
                # 纯对话检索（kind=dialog）用满 limit；kind=all 才和视觉各占一半
                t_params.append(limit if kind == "dialog" else max(4, limit // 2))
                if offset:
                    t_sql += " OFFSET ?"
                    t_params.append(offset)
                for r in self._conn.execute(t_sql, t_params).fetchall():
                    hits.append(
                        {"kind": "user_turn", "time": _fmt_ts(r["ts"]), "text": r["text"]}
                    )

                st = self._turn_stats(client_id, q, start_ts, end_ts)
                total_matched += st[0]
                if st[1] is not None:
                    earliest = st[1] if earliest is None else min(earliest, st[1])
                    latest = st[2] if latest is None else max(latest, st[2])

        # 视觉与对话各自有序，合并后仍按时间正/倒序对齐，方便模型读"转场"
        hits.sort(key=lambda h: h["time"], reverse=(direction == "DESC"))
        # 保留 frame_path（brain 用它配证据图标签），只做去重收集
        for h in hits:
            fp = h.get("frame_path") or ""
            if fp and len(frame_paths) < 4 and fp not in frame_paths and os.path.isfile(fp):
                frame_paths.append(fp)

        out: dict[str, Any] = {
            "found": len(hits) > 0,
            "count": len(hits),
            "total_matched": total_matched,
            "returned": len(hits),
            "order": direction.lower(),
            "kind": kind,
            "hits": hits,
        }
        if earliest is not None:
            out["earliest_time"] = _fmt_ts(earliest)
            out["latest_time"] = _fmt_ts(latest)
        if total_matched > len(hits):
            out["truncated_note"] = (
                "还有 " + str(total_matched - len(hits)) + " 条命中未返回；"
                "最早到 " + str(out.get("earliest_time")) + "，别把返回的第一条当成最早记录。"
            )
        if frame_paths:
            out["frame_paths"] = frame_paths
            out["frame_note"] = "这些是当时真实拍到的原图；文字不足以说清细节时交给 inspect_past_frames。"
        return out

    def _vis_stats(self, client_id, query, start_ts, end_ts):
        sql = "SELECT COUNT(*), MIN(start_ts), MAX(start_ts) FROM memory_units WHERE client_id = ? AND status = 'active' AND kind IN ('vision_episode', 'event_anchor')"
        params = [client_id]
        if start_ts is not None:
            sql += " AND MIN(end_ts, start_ts + 600) >= ?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND start_ts <= ?"
            params.append(end_ts)
        if query:
            sql += " AND (search_text LIKE ? OR caption_clean LIKE ? OR ocr_verified LIKE ?)"
            like = "%" + query + "%"
            params.extend([like, like, like])
        r = self._conn.execute(sql, params).fetchone()
        return (r[0] or 0, r[1], r[2])

    def _turn_stats(self, client_id, query, start_ts, end_ts):
        sql = "SELECT COUNT(*), MIN(ts), MAX(ts) FROM turns WHERE client_id = ? AND role = 'user' AND is_poisoned = 0"
        params = [client_id]
        if start_ts is not None:
            sql += " AND ts >= ?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND ts <= ?"
            params.append(end_ts)
        if query:
            sql += " AND text LIKE ?"
            params.append("%" + query + "%")
        r = self._conn.execute(sql, params).fetchone()
        return (r[0] or 0, r[1], r[2])

    def browse_vision_timeline(
        self,
        client_id: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        max_points: int = 24,
    ) -> dict[str, Any]:
        """视觉转场时间线：只看 vision_episode，正序，均匀采样 + 去重。

        2026-11-22：回答"几点到/几点下车"必须能按时间正着走一遍画面，
        而不是拿关键词在聊天记录里撞运气。
        """
        max_points = max(2, min(int(max_points or 24), 48))
        sql = """
            SELECT start_ts, COALESCE(NULLIF(caption_clean, ''), search_text) AS body, best_frame_path
            FROM memory_units
            WHERE client_id = ? AND status = 'active' AND kind = 'vision_episode'
        """
        params: list[Any] = [client_id]
        # 时间线按 start_ts 严格切窗：跨天单元的 end_ts 会把前一天拉进来
        if start_ts is not None:
            sql += " AND start_ts >= ?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND start_ts <= ?"
            params.append(end_ts)
        sql += " ORDER BY start_ts ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        total = len(rows)
        if total == 0:
            return {"found": False, "count": 0, "total_in_window": 0, "points": []}

        def _bigrams(t: str) -> set:
            t = ''.join(ch for ch in (t or '') if '\u4e00' <= ch <= '\u9fff')
            return {t[i:i + 2] for i in range(len(t) - 1)}

        def _sim(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / float(len(a | b))

        # 2026-11-22：先按场景变化挑（转场点），不够再用均匀采样补齐。
        # 纯均匀采样会把「穿外套→照镜子→进电梯→走上小区路」这 3 分钟直接跳过。
        scenes = []
        last = None
        for r in rows:
            bg = _bigrams(r['body'])
            if last is None or _sim(last, bg) < 0.35:
                scenes.append(r)
                last = bg
        if len(scenes) > max_points:
            step = len(scenes) / float(max_points)
            scenes = [scenes[int(i * step)] for i in range(max_points)]
        picked = list(scenes)
        if len(picked) < max_points and total > len(picked):
            have = {r['start_ts'] for r in picked}
            need = max_points - len(picked)
            step = max(1, total // (need + 1))
            for i in range(0, total, step):
                r = rows[i]
                if r['start_ts'] in have:
                    continue
                picked.append(r)
                have.add(r['start_ts'])
                if len(picked) >= max_points:
                    break
        picked.sort(key=lambda r: r['start_ts'])
        if rows and (not picked or picked[-1]['start_ts'] != rows[-1]['start_ts']):
            picked.append(rows[-1])

        points = []
        for r in picked[:max_points]:
            points.append(
                {
                    "time": _fmt_ts(r["start_ts"]),
                    "caption": r["body"],
                    "best_frame_path": r["best_frame_path"] or "",
                }
            )
        return {
            "found": True,
            "count": len(points),
            "total_in_window": total,
            "sampled": total > len(points),
            "points": points,
            "note": "按时间正序；要确认哪一帧是到馆瞬间，把 best_frame_path 交给 inspect_past_frames 看原图。",
        }

