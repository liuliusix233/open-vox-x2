"""夜间「做梦蒸馏 v2」——单库 vox_unified.db 的日级记忆压缩。

三个动作，全部只读已落库数据，不脑补：
  1) 视觉轨迹 → 日级 event_anchor（3–6 条，src_key 幂等，重跑不重复）
  2) 仅从 [owner] 且 is_poisoned=0 且 spk_sim>=阈值 的语音里抽持久事实 → semantic_facts
  3) 清掉 >N 天、无 OCR、无锚点引用、非场景跃迁的冗余静态原图，最后 WAL checkpoint

用法：
    python3 -m vox_v2.dream_consolidator --date 2026-11-22 --dry-run
    python3 -m vox_v2.dream_consolidator            # 默认昨天，真写
    python3 -m vox_v2.dream_consolidator --keep-days 7 --min-sim 0.65
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger("vox_v2.dream")

TZ = timezone(timedelta(hours=8))
ANCHOR_BUCKET_S = 1800.0        # 30 分钟一档
MAX_ANCHORS = 6
MAX_FACT_UTTERANCES = 60
OWNER_PREFIX = "[owner]"


def day_window(date_str: str | None) -> tuple[str, float, float]:
    """Return (date_str, start_ts, end_ts) in Asia/Shanghai for that calendar day."""
    if date_str:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=TZ)
    else:
        d = (datetime.now(TZ) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = d.timestamp()
    return d.strftime("%Y-%m-%d"), start, start + 86400.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _head_caption(group: list[sqlite3.Row]) -> str:
    for r in group:
        c = (r["caption_clean"] or r["search_text"] or "").strip()
        if c:
            return c
    return ""


def _scene_similar(a: str, b: str) -> bool:
    """两条画面描述是否属于同一场景（字符二元组 Jaccard > 0.45）。"""
    def grams(s: str) -> set[str]:
        t = "".join(ch for ch in s if not ch.isspace())
        return {t[i:i + 2] for i in range(max(0, len(t) - 1))}

    x, y = grams(a), grams(b)
    if not x or not y:
        return False
    return len(x & y) / float(len(x | y)) > 0.45


class DreamConsolidator:
    """Daily consolidation. All writes are idempotent; dry_run touches nothing."""

    def __init__(
        self,
        db_path: str,
        *,
        keep_days: int = 7,
        min_sim: float = 0.65,
        health_url: str = "http://127.0.0.1:8766/health",
        llm_caller: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None,
        frame_roots: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.keep_days = max(1, int(keep_days))
        self.min_sim = float(min_sim)
        self.health_url = health_url
        self._llm_caller = llm_caller
        self.frame_roots = frame_roots or []

    # ------------------------------------------------------------------ db
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # --------------------------------------------------------- health gate
    async def is_quiet(self) -> tuple[bool, str]:
        """True = 可以跑（眼镜没在活跃对话、最近 120 秒没人说话）。"""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=3.0)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(self.health_url) as resp:
                    if resp.status != 200:
                        return True, f"health {resp.status} -> 按空闲处理"
                    data = await resp.json()
        except Exception as exc:
            return True, f"health 不可达({type(exc).__name__}) -> 按空闲处理"
        if data.get("busy"):
            return False, "session_busy"
        last = float(data.get("last_user_speech") or 0.0)
        if last and (time.time() - last) < 120.0:
            return False, "recent_speech"
        return True, "quiet"

    # ------------------------------------------------------- 1) anchors
    def build_anchors(self, conn: sqlite3.Connection, client_id: str,
                      day_start: float, day_end: float, date_str: str,
                      dry_run: bool) -> dict[str, Any]:
        rows = conn.execute(
            """
            SELECT start_ts, end_ts, search_text, caption_clean, best_frame_path,
                   lat, lon, ocr_verified
            FROM memory_units
            WHERE client_id=? AND kind='vision_episode' AND status='active'
              AND start_ts>=? AND start_ts<?
              AND (TRIM(caption_clean)<>'' OR TRIM(search_text)<>'')
            ORDER BY start_ts ASC
            """,
            (client_id, day_start, day_end),
        ).fetchall()
        if not rows:
            return {"anchors": 0, "episodes": 0, "written": 0, "note": "no_vision"}

        # ① 30 分钟一档
        buckets: dict[int, list[sqlite3.Row]] = {}
        for r in rows:
            buckets.setdefault(int((r["start_ts"] - day_start) // ANCHOR_BUCKET_S), []).append(r)

        # ② 相邻且画面相似的桶合并成一段（17:09–19:29 桌面工作 → 1 条）
        merged: list[list[sqlite3.Row]] = []
        for _, grp in sorted(buckets.items()):
            if merged:
                gap = grp[0]["start_ts"] - merged[-1][-1]["end_ts"]
                # 同场景合并；或时间相邻且有一侧是低频段（STANDBY/出门 15s 一帧）也合并，
                # 让"上车→停车场→高速"这类稀疏但连续的移动场景收成一条
                if _scene_similar(_head_caption(merged[-1]), _head_caption(grp)) or (
                    gap <= ANCHOR_BUCKET_S and len(merged[-1]) < 25 and len(grp) < 25
                ):
                    merged[-1] = merged[-1] + grp
                    continue
            merged.append(list(grp))

        # ③ 超过 MAX_ANCHORS 时用「最远点采样」铺满全天：
        #    先放帧数最多的一段，之后每次挑"离已选点最远"的一段（同距比帧数）。
        #    这样桌面上 LIVE 1s/帧 的 800 帧不会把出门 STANDBY 15s/帧 的车程挤掉。
        groups = merged
        if len(groups) > MAX_ANCHORS:
            starts = [g[0]["start_ts"] for g in groups]
            seed = max(range(len(groups)), key=lambda i: len(groups[i]))
            picked = [seed]
            while len(picked) < MAX_ANCHORS:
                best_i, best_key = None, None
                for i in range(len(groups)):
                    if i in picked:
                        continue
                    dist = min(abs(starts[i] - starts[j]) for j in picked)
                    key = (dist, len(groups[i]))
                    if best_key is None or key > best_key:
                        best_i, best_key = i, key
                if best_i is None:
                    break
                picked.append(best_i)
            groups = [groups[i] for i in sorted(picked)]

        anchors = []
        for g in groups:
            t0, t1 = g[0]["start_ts"], g[-1]["end_ts"]
            caps: list[str] = []
            seen: set[str] = set()
            for r in g:
                c = (r["caption_clean"] or r["search_text"] or "").strip()
                key = c[:12]
                if c and key not in seen:
                    seen.add(key)
                    caps.append(c)
                if len(caps) >= 3:
                    break
            lat = next((r["lat"] for r in g if r["lat"] is not None), None)
            lon = next((r["lon"] for r in g if r["lon"] is not None), None)
            start_local = datetime.fromtimestamp(t0, TZ).strftime("%H:%M")
            end_local = datetime.fromtimestamp(t1, TZ).strftime("%H:%M")
            span = start_local if start_local == end_local else f"{start_local}–{end_local}"
            # 精炼：只用最代表性的一条画面描述（≤64字）+ 帧数，不堆三段长句
            head = _clip(caps[0], 64) if caps else "画面记录"
            # 跨场景的时段（如"坐在桌前→上车"）把结尾画面也带一句，避免只记住开头
            tail_cap = _head_caption(g[-3:]) if len(g) > 3 else ""
            spans_long = (t1 - t0) > 2400.0     # 只有跨 40 分钟以上的时段才带结尾画面
            if tail_cap and spans_long and not _scene_similar(head, tail_cap):
                head = head + " → " + _clip(tail_cap, 32)
            extra = f"（该时段共 {len(g)} 帧）" if len(g) > 1 else ""
            anchors.append({
                "start_ts": t0,
                "end_ts": max(t1, t0 + 1.0),
                "search_text": _clip(f"事件：{span} {head}{extra}", 140),
                "caption": _clip("；".join(caps[:2]), 160),
                "frames": len(g),
                "lat": lat,
                "lon": lon,
                "frame_path": next((r["best_frame_path"] for r in g if r["best_frame_path"]), ""),
            })

        written = 0
        for idx, a in enumerate(anchors):
            src = f"dream:anchor:{date_str}:{idx}"
            exists = conn.execute(
                "SELECT unit_id FROM memory_units WHERE src_key=?", (src,)
            ).fetchone()
            if dry_run:
                continue
            if exists:
                conn.execute(
                    """
                    UPDATE memory_units
                       SET start_ts=?, end_ts=?, search_text=?, caption_clean=?,
                           best_frame_path=?, lat=?, lon=?, hit_count=hit_count+1
                     WHERE src_key=?
                    """,
                    (a["start_ts"], a["end_ts"], a["search_text"], a["caption"],
                     a["frame_path"], a["lat"], a["lon"], src),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO memory_units
                        (src_key, client_id, scope, kind, start_ts, end_ts, search_text,
                         caption_clean, ocr_verified, text_verbatim, entities_str, scene_tag,
                         objects_json, has_text, tag_model, struct_json, best_frame_path,
                         best_frame_score, lat, lon, acc, confidence, hit_count, status, simhash64)
                    VALUES (?, ?, 'user_shared', 'event_anchor', ?, ?, ?, ?, '', '', ?, 'dream',
                            '[]', 0, 'dream', ?, ?, 0.0, ?, ?, NULL, 0.7, 1, 'active', 0)
                    """,
                    (
                        src, client_id, a["start_ts"], a["end_ts"], a["search_text"],
                        a["caption"], "dream_day:" + date_str, "{}",
                        a["frame_path"], a["lat"], a["lon"],
                    ),
                )
            written += 1
        return {"anchors": len(anchors), "episodes": len(rows), "written": written,
                "preview": [a["search_text"] for a in anchors]}

    # --------------------------------------------------------- 2) facts
    def owner_utterances(self, conn: sqlite3.Connection, client_id: str,
                         day_start: float, day_end: float) -> list[sqlite3.Row]:
        rows = conn.execute(
            """
            SELECT turn_id, ts, text, spk_sim FROM turns
            WHERE client_id=? AND role='user' AND is_poisoned=0
              AND text LIKE ? AND spk_sim>=?
              AND ts>=? AND ts<?
            ORDER BY ts ASC
            """,
            (client_id, OWNER_PREFIX + "%", self.min_sim, day_start, day_end),
        ).fetchall()
        out = []
        for r in rows:
            t = (r["text"] or "").replace(OWNER_PREFIX, "", 1).strip()
            # 结构化前置过滤：只留第一人称陈述，砍掉对画面的旁白式复述
            if len(t) < 4 or not any(k in t for k in ("我", "俺", "咱")):
                continue
            out.append({**dict(r), "clean": t})
        return out

    async def _default_llm(self, messages: list[dict[str, Any]]) -> str:
        import aiohttp

        from .config import VoxConfig

        cfg = VoxConfig.from_env()
        payload: dict[str, Any] = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": 0.0,
            "reasoning_effort": "medium",
            "max_tokens": 800,
        }
        headers = {
            "Authorization": "Bearer " + cfg.llm_api_key,
            "Content-Type": "application/json",
        }
        # 直连局域网推理机，绕开 HTTP_PROXY
        async with aiohttp.ClientSession(trust_env=False) as s:
            async with s.post(cfg.llm_url, json=payload, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=90)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return str(data["choices"][0]["message"].get("content") or "")

    async def extract_facts(self, conn: sqlite3.Connection, client_id: str,
                            day_start: float, day_end: float, date_str: str,
                            dry_run: bool) -> dict[str, Any]:
        utts = self.owner_utterances(conn, client_id, day_start, day_end)
        if not utts:
            return {"facts": 0, "inserted": 0, "updated": 0, "utterances": 0}
        utts = utts[:MAX_FACT_UTTERANCES]
        lines = "\n".join(
            f"- {datetime.fromtimestamp(u['ts'], TZ).strftime('%H:%M')} {_clip(u['clean'], 80)}"
            for u in utts
        )
        prompt = (
            "下面是一个人戴智能眼镜时、当天亲口说过的话（已确认是他本人）。\n"
            "请抽取**值得长期记住、以后能直接用来服务他**的个人档案级事实。\n\n"
            "✅ 只收这几类：身份/职业、常住地与常去城市、家人与称呼、长期健康与饮食禁忌、"
            "每天在用的工具/设备/软件偏好、固定作息与通勤、长期目标。\n"
            "❌ 一律不要：眼前物品的描写（鼠标、眼镜、纸箱、杯子…）、一次性的欲望或情绪"
            "（想喝柠檬水、今天好累）、对别人的评价、玩笑、猜测、任何需要推断才能得出的结论。\n"
            "判定标准：如果这条事实半年后仍然成立、并且能改变你下次的回答，才值得写。\n\n"
            "规则：\n"
            "1. 最多 4 条；宁可空数组，也不要凑数。\n"
            "2. confidence < 0.75 的不要输出。\n"
            "3. 只输出 JSON 数组，形如 "
            '{"predicate":"常住城市","object":"上海","cardinality":"single","confidence":0.92}\n'
            "4. predicate 用 2–8 个汉字名词短语；object 不超过 30 字。\n"
            "5. 不要任何解释文字，不要 Markdown。\n\n"
            f"日期：{date_str}\n原话：\n{lines}\n"
        )
        messages = [
            {"role": "system", "content": "你是一个严格的事实抽取器，只输出 JSON 数组。"},
            {"role": "user", "content": prompt},
        ]
        caller = self._llm_caller or self._default_llm
        raw = ""
        try:
            raw = await caller(messages)
        except Exception as exc:
            log.warning("dream facts llm failed: %s", exc)
            return {"facts": 0, "inserted": 0, "updated": 0, "utterances": len(utts),
                    "error": str(exc)[:120]}
        m = re.search(r"\[.*\]", raw, re.S)
        items: list[dict[str, Any]] = []
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    items = [x for x in parsed if isinstance(x, dict)]
            except Exception as exc:
                log.warning("dream facts json parse failed: %s", exc)
        inserted = updated = 0
        preview: list[str] = []
        for it in items[:4]:
            pred = str(it.get("predicate") or "").strip()
            obj = str(it.get("object") or "").strip()
            if not (2 <= len(pred) <= 12) or not (2 <= len(obj) <= 30):
                continue
            try:
                conf = float(it.get("confidence") or 0.7)
            except Exception:
                conf = 0.7
            if conf < 0.75:
                continue
            card = str(it.get("cardinality") or "single")
            norm = _norm(obj)
            row = conn.execute(
                """
                SELECT fact_id FROM semantic_facts
                WHERE client_id=? AND predicate=? AND norm_object=? AND status='active'
                """,
                (client_id, pred, norm),
            ).fetchone()
            preview.append(f"{pred} = {obj}")
            if dry_run:
                continue
            now = time.time()
            if row:
                conn.execute(
                    "UPDATE semantic_facts SET last_verified_ts=?, hit_count=hit_count+1, "
                    "confidence=MAX(confidence, ?) WHERE fact_id=?",
                    (now, conf, row["fact_id"]),
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO semantic_facts
                        (client_id, subject, predicate, cardinality, object_text, norm_object,
                         status, source, evidence_turn_id, confidence, hit_count,
                         first_event_ts, last_verified_ts, expires_at, superseded_by, event_ts)
                    VALUES (?, 'user', ?, ?, ?, ?, 'active', 'dream', NULL, ?, 1, ?, ?, NULL, NULL, ?)
                    """,
                    (client_id, pred, card, obj, norm, conf, day_start, now, day_start),
                )
                inserted += 1
        return {"facts": len(items), "inserted": inserted, "updated": updated,
                "utterances": len(utts), "preview": preview}

    # ------------------------------------------------------- 3) cleanup
    def cleanup_frames(self, conn: sqlite3.Connection, day_end: float,
                       dry_run: bool) -> dict[str, Any]:
        cutoff = day_end - self.keep_days * 86400.0
        rows = conn.execute(
            """
            SELECT f.frame_path, f.ts_start, f.tags FROM vision_frames f
            WHERE f.ts_start < ?
              AND TRIM(COALESCE(f.caption,''))=''
              AND COALESCE(f.tags,'') NOT LIKE '%scene%'
              AND NOT EXISTS (
                  SELECT 1 FROM memory_units m
                  WHERE m.best_frame_path = f.frame_path AND m.status='active'
              )
            LIMIT 2000
            """,
            (cutoff,),
        ).fetchall()
        freed = 0
        removed_files = 0
        for r in rows:
            p = r["frame_path"] or ""
            if not dry_run:
                try:
                    if p and os.path.isfile(p):
                        freed += os.path.getsize(p)
                        os.unlink(p)
                        removed_files += 1
                except OSError:
                    pass
                conn.execute("DELETE FROM vision_frames WHERE frame_path=?", (p,))
        return {"candidates": len(rows), "removed_files": removed_files,
                "freed_mb": round(freed / 1048576.0, 2), "keep_days": self.keep_days}

    # ------------------------------------------------------------- driver
    async def run(self, date_str: str | None = None, *, dry_run: bool = False,
                  client_id: str = "voxx2", force: bool = False,
                  skip_facts: bool = False) -> dict[str, Any]:
        date_str, day_start, day_end = day_window(date_str)
        report: dict[str, Any] = {"date": date_str, "dry_run": dry_run,
                                  "min_sim": self.min_sim, "keep_days": self.keep_days}
        if not dry_run or force:
            ok, why = await self.is_quiet()
            report["health"] = why
            if not ok and not force:
                report["skipped"] = True
                return report
        conn = self._connect()
        try:
            report.update(self.build_anchors(conn, client_id, day_start, day_end, date_str, dry_run))
            if skip_facts:
                report.update({"facts": 0, "inserted": 0, "updated": 0, "utterances": 0,
                               "facts_skipped": True})
            else:
                report.update(await self.extract_facts(conn, client_id, day_start, day_end,
                                                       date_str, dry_run))
            report.update(self.cleanup_frames(conn, day_end, dry_run))
            if not dry_run:
                conn.commit()
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as exc:
                    log.warning("wal_checkpoint failed: %s", exc)
        finally:
            conn.close()
        return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vox nightly dream consolidator v2")
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday, Asia/Shanghai)")
    ap.add_argument("--dry-run", action="store_true", help="只统计与预览，不写库")
    ap.add_argument("--keep-days", type=int, default=7)
    ap.add_argument("--min-sim", type=float, default=0.65)
    ap.add_argument("--db", default=os.environ.get(
        "VOX_DB", os.path.expanduser("~/.openclaw/workspace/jarvis/voice_server/vox_unified.db")))
    ap.add_argument("--health-url", default="http://127.0.0.1:8766/health")
    ap.add_argument("--force", action="store_true", help="忽略 /health 繁忙判定")
    ap.add_argument("--skip-facts", action="store_true", help="只写日级锚点，不做事实抽取")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dc = DreamConsolidator(
        args.db, keep_days=args.keep_days, min_sim=args.min_sim, health_url=args.health_url
    )
    report = asyncio.run(dc.run(args.date, dry_run=args.dry_run, force=args.force,
                               skip_facts=args.skip_facts))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
