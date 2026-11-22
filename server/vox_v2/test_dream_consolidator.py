"""冒烟 #10：夜间做梦蒸馏 v2 的硬过滤 / 幂等 / 永不写 todo。"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone

from .dream_consolidator import DreamConsolidator, day_window
from .memory import MemoryStore
from .todo_goals import TodoGoalStore

TZ = timezone(timedelta(hours=8))


def _ts(date_str: str, hh: int, mm: int = 0) -> float:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hh, minute=mm, tzinfo=TZ).timestamp()


def _seed_vision(conn: sqlite3.Connection, client_id: str, base: float) -> None:
    caps = [
        "客厅里白色沙发和茶几，窗外是小区绿化",
        "餐桌上的外卖盒与两杯饮料，室内灯光偏暖",
        "车内仪表盘与方向盘，前方是高架路",
        "地下车库的水泥柱与车位编号",
        "办公室桌面，双显示器与机械键盘",
        "楼下便利店货架，饮料冷柜",
    ]
    for i, cap in enumerate(caps):
        conn.execute(
            """
            INSERT INTO memory_units
                (src_key, client_id, scope, kind, start_ts, end_ts, search_text,
                 caption_clean, ocr_verified, text_verbatim, entities_str, scene_tag,
                 objects_json, has_text, tag_model, struct_json, best_frame_path,
                 best_frame_score, lat, lon, acc, confidence, hit_count, status, simhash64)
            VALUES (?, ?, 'user_shared', 'vision_episode', ?, ?, ?, ?, '', '', '', 'unknown',
                    '[]', 0, 'test', '{}', ?, 0.5, 31.2, 121.4, 10.0, 0.8, 1, 'active', 0)
            """,
            (f"test:ep:{i}", client_id, base + i * 1800, base + i * 1800 + 60,
             cap, cap, f"/tmp/test_frame_{i}.jpg"),
        )


async def run_all_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_dream_test_") as tmp:
        db_path = os.path.join(tmp, "vox.db")
        mem = MemoryStore(db_path)
        TodoGoalStore(db_path)          # 建 todos 表，用来验证"永不写 todo"
        client = "voxx2"
        date_str = "2026-11-22"
        _, day_start, day_end = day_window(date_str)

        # —— 语音：只有一条应当被采用 ——
        mem.record_turn(client, "user", "[owner] 我常住上海，平时用 vim 写代码", spk_sim=0.91)
        mem.record_turn(client, "user", "[owner] 这条声纹不够像，别采信", spk_sim=0.51)
        mem.record_turn(client, "user", "[他人] 我住在北京，我喜欢喝可乐", spk_sim=0.95)
        mem.record_turn(client, "user", "[owner] 这是我被标脏的话", spk_sim=0.95, is_poisoned=1)
        # 把时间挪到当天（record_turn 用的是 now）
        conn = mem._conn
        conn.execute("UPDATE turns SET ts=?", (_ts(date_str, 10, 0),))
        conn.commit()

        _seed_vision(conn, client, _ts(date_str, 9, 0))
        # 一条旧的无用帧（应被清理）和一条被引用的帧（应保留）
        old_ts = day_end - 30 * 86400
        conn.execute(
            "INSERT INTO vision_frames(frame_path, client_id, ts_start, ts_end, caption, tags, "
            "sheet_path, model, lat, lon, acc, ingested_at) VALUES (?,?,?,?,'','', '', 'test', NULL, NULL, NULL, ?)",
            ("/tmp/dream_orphan.jpg", client, old_ts, old_ts, old_ts),
        )
        conn.execute(
            "INSERT INTO vision_frames(frame_path, client_id, ts_start, ts_end, caption, tags, "
            "sheet_path, model, lat, lon, acc, ingested_at) VALUES (?,?,?,?,'已引用','', '', 'test', NULL, NULL, NULL, ?)",
            ("/tmp/test_frame_0.jpg", client, old_ts, old_ts, old_ts),
        )
        conn.commit()

        seen_prompts: list[str] = []

        async def fake_llm(messages):
            seen_prompts.append("\n".join(m.get("content", "") for m in messages))
            return json.dumps([
                {"predicate": "常住城市", "object": "上海", "cardinality": "single", "confidence": 0.92},
                {"predicate": "常用编辑器", "object": "vim", "cardinality": "single", "confidence": 0.8},
            ], ensure_ascii=False)

        dc = DreamConsolidator(db_path=db_path, keep_days=7, min_sim=0.65,
                               health_url="http://127.0.0.1:9/health", llm_caller=fake_llm)

        todos_before = conn.execute("SELECT COUNT(*) FROM todos").fetchone()[0]
        r1 = await dc.run(date_str, dry_run=False, client_id=client)
        anchors_1 = conn.execute(
            "SELECT COUNT(*) FROM memory_units WHERE src_key LIKE 'dream:anchor:%'").fetchone()[0]
        assert 1 <= anchors_1 <= 6, f"anchor 数量应在 1–6，实际 {anchors_1}"
        assert anchors_1 == r1["anchors"], "报告数量与落库数量不一致"
        assert r1["inserted"] == 2, f"应写入 2 条事实，实际 {r1['inserted']}"

        # 硬过滤：抽取 prompt 里绝不能出现他人 / 低声纹 / 脏数据
        joined = seen_prompts[0]
        assert "北京" not in joined and "可乐" not in joined, "他人语音泄漏进了事实抽取"
        assert "声纹不够像" not in joined, "spk_sim<0.65 的语音被采信"
        assert "标脏的话" not in joined, "is_poisoned=1 的语音被采信"

        # 幂等：再跑一次，锚点不增加，事实走 update 不新增
        r2 = await dc.run(date_str, dry_run=False, client_id=client)
        anchors_2 = conn.execute(
            "SELECT COUNT(*) FROM memory_units WHERE src_key LIKE 'dream:anchor:%'").fetchone()[0]
        assert anchors_2 == anchors_1, f"重跑产生了重复锚点：{anchors_1} -> {anchors_2}"
        assert r2["inserted"] == 0 and r2["updated"] == 2, "重跑应只更新已有事实"

        # 永不写 todo
        todos_after = conn.execute("SELECT COUNT(*) FROM todos").fetchone()[0]
        assert todos_after == todos_before, "蒸馏过程写入了 todo"

        # 存储自洁：旧的无用帧删除，被引用的保留
        orphan = conn.execute(
            "SELECT COUNT(*) FROM vision_frames WHERE frame_path='/tmp/dream_orphan.jpg'").fetchone()[0]
        kept = conn.execute(
            "SELECT COUNT(*) FROM vision_frames WHERE frame_path='/tmp/test_frame_0.jpg'").fetchone()[0]
        assert orphan == 0, "无 OCR / 无引用的旧帧未被清理"
        assert kept == 1, "被锚点引用的关键帧被误删"

        # dry-run 不写库
        before = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
        await dc.run(date_str, dry_run=True, client_id=client)
        after = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
        assert before == after, "dry-run 写入了数据"

        mem.close()
    print("ALL DREAM CONSOLIDATOR SMOKE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
