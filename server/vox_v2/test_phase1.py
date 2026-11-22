"""End-to-end smoke tests for Phase 1 (hormones_v2 + todo_goals + agency_loop)."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time

try:
    from .agency_loop_v2 import AgencyLoopV2
    from .hormones_v2 import BASE, HormonesV2
    from .todo_goals import TodoGoalStore
except ImportError:
    from agency_loop_v2 import AgencyLoopV2
    from hormones_v2 import BASE, HormonesV2
    from todo_goals import TodoGoalStore


def test_hormones() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_p1_h_") as tmp:
        snap = os.path.join(tmp, "horm.json")
        logp = os.path.join(tmp, "causal.jsonl")
        h = HormonesV2(snapshot_path=snap, causal_log_path=logp, enabled=True)

        t0 = 1_800_000_000.0
        assert h.dominant_delta(now=t0) == ""
        assert h.dynamic_tail_directive(now=t0) == ""
        assert h.proactive_interval_s(now=t0) == 95.0

        # User replies shortly after proactive speech -> DA & OXT jump
        h.on_proactive_spoken(now=t0)
        h.on_user_reply(text_len=20, now=t0 + 5.0)
        assert h.dominant_delta(now=t0 + 5.0) == "DA"
        assert "兴致高" in h.dynamic_tail_directive(now=t0 + 5.0)
        assert h.tts_style(now=t0 + 5.0)["speed"] == 1.06

        # Repeated ignored proactive speech -> HT rises, eventually mutes
        for i in range(3):
            ts = t0 + 100.0 + i * 50.0
            h.on_proactive_spoken(now=ts)
            assert h.on_proactive_ignored(now=ts + 36.0)
        assert h.proactive_interval_s(now=t0 + 240.0) is None

        # Causal log output
        rec = h.log_causal_turn("dialog", "你刚在看什么？", user_replied_in_35s=True)
        assert rec["has_question"] is True
        assert os.path.isfile(logp)
        assert os.path.isfile(snap)


def test_todo_goals() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_p1_tg_") as tmp:
        db = os.path.join(tmp, "vox_unified.db")
        tg = TodoGoalStore(db)
        try:
            r1 = tg.exec_tool("voxx2", "manage_todo", {"action": "add", "title": "买5号电池", "priority": "high"})
            assert r1["ok"]
            r2 = tg.exec_tool("other_client", "manage_todo", {"action": "add", "title": "别人的事"})
            assert r2["ok"]

            # Client isolation
            listed = tg.list_todos("voxx2")
            assert len(listed) == 1 and listed[0]["title"] == "买5号电池"

            # Active goal
            g1 = tg.exec_tool(
                "voxx2",
                "manage_goal",
                {"action": "set", "title": "去图书馆", "destination": "Shanghai Library", "status_note": "准备出门"},
            )
            assert g1["ok"]
            ctx = tg.format_context_block("voxx2")
            assert "Shanghai Library" in ctx and "买5号电池" in ctx

            # Complete todo and goal
            assert tg.exec_tool("voxx2", "manage_todo", {"action": "done", "title": "电池"})["ok"]
            assert tg.exec_tool("voxx2", "manage_goal", {"action": "complete", "title": "图书馆"})["ok"]
            assert tg.format_context_block("voxx2") == ""
        finally:
            tg.close()


async def test_agency_loop() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_p1_ag_") as tmp:
        db = os.path.join(tmp, "vox_unified.db")
        h = HormonesV2(snapshot_path=os.path.join(tmp, "h.json"), causal_log_path=os.path.join(tmp, "c.jsonl"))
        tg = TodoGoalStore(db)
        spoken: list[tuple[str, float]] = []

        tg.set_or_update_goal("voxx2", "去图书馆", destination="Shanghai Library", status_note="穿外套")

        async def fake_llm(messages, tools):
            assert "[ACTIVE_GOALS]" in messages[1]["content"]
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "proactive_speak",
                            "arguments": json.dumps(
                                {"text": "看你已经在穿外套了，是准备去图书馆吗？", "reason": "goal_confirm"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

        async def fake_speak(text: str, speed: float) -> None:
            spoken.append((text, speed))

        async def fake_exec(name, args):
            return {"ok": True}

        loop = AgencyLoopV2(
            client_id="voxx2",
            system_prompt="你是 Vox。",
            unified_tools=[],
            hormones=h,
            todo_goals=tg,
            llm_caller=fake_llm,
            tool_executor=fake_exec,
            speak_cb=fake_speak,
            state_provider=lambda: {
                "busy": False,
                "last_user_ts": time.time() - 120.0,
                "mode": "LIVE",
                "power_profile": "AC",
                "heartbeat_mult": 1.0,
                "latest_caption": "男子在镜子前穿深色外套",
                "recent_turns": [{"role": "user", "text": "待会儿想去图书馆"}],
            },
            enabled=True,
        )
        res = await loop.step_once(force=True)
        assert res["action"] == "proactive_speak"
        assert len(spoken) == 1 and "图书馆" in spoken[0][0]
        assert h.unacked_proactive == 1
        tg.close()


def main() -> None:
    test_hormones()
    test_todo_goals()
    asyncio.run(test_agency_loop())
    print("ALL PHASE 1 AGENCY SMOKE TESTS PASSED!")


if __name__ == "__main__":
    main()
