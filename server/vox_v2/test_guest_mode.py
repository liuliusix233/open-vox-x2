"""GUEST 模式冒烟：状态机 + 工具硬拦截 + 上下文零泄露 + owner回归。"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vox_v2.app import evaluate_guest_state, GUEST_EXIT_SIM
from vox_v2.brain import GUEST_BLOCK_MSG, VoxBrain
from vox_v2.config import VoxConfig
from vox_v2.memory import MemoryStore
from vox_v2.vision import VisionStore


def test_state_machine() -> None:
    base = 800.0
    now = 1000.0
    # 非owner近场、owner 20s 没说话 -> 进访客
    g, last = evaluate_guest_state(speaker="bystander", spk_sim=0.2, raw_rms=600,
                                   owner_near_rms=base, guest_mode=False,
                                   now=now, last_owner_near_ts=0.0)
    assert g is True, "应进入访客模式"
    # 非owner但声音太远（远处路人）-> 不进
    g2, _ = evaluate_guest_state(speaker="bystander", spk_sim=0.2, raw_rms=200,
                                 owner_near_rms=base, guest_mode=False,
                                 now=now, last_owner_near_ts=0.0)
    assert g2 is False, "远场路人不该触发"
    # 机主刚说过话（20s 内）-> 不因他人进入访客模式
    g3, _ = evaluate_guest_state(speaker="bystander", spk_sim=0.2, raw_rms=900,
                                 owner_near_rms=base, guest_mode=False,
                                 now=now, last_owner_near_ts=now - 5)
    assert g3 is False, "owner刚在近场说话，不应被bystander顶掉"
    # owner高置信度回归 -> 立刻解除
    g4, last4 = evaluate_guest_state(speaker="owner", spk_sim=0.9, raw_rms=700,
                                     owner_near_rms=base, guest_mode=True,
                                     now=now + 30, last_owner_near_ts=0.0)
    assert g4 is False and last4 == now + 30, "owner回归应立即解锁"
    # owner低置信度 -> 不改状态
    g5, _ = evaluate_guest_state(speaker="owner", spk_sim=0.5, raw_rms=700,
                                 owner_near_rms=base, guest_mode=True,
                                 now=now + 31, last_owner_near_ts=0.0)
    assert g5 is True, "低置信度owner不应解锁"
    print("STATE MACHINE OK")


async def test_tools_and_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = VoxConfig(db_path=os.path.join(tmp, "t.db"), frame_dir=tmp)
        mem = MemoryStore(cfg.db_path)
        vis = VisionStore(tmp, ttl_s=60)
        mem.record_turn("c1", "user", "owner历史：昨天下午三点到了图书馆", epoch_id=1)
        mem.record_turn("c1", "assistant", "是的，你昨天三点到图书馆", epoch_id=1)

        captured = {}

        async def mock_llm(messages, tools=None, tool_choice=None):
            captured["messages"] = [dict(m) for m in messages]
            return {"role": "assistant", "content": "你好，我是 Vox。"}

        b = VoxBrain(cfg, mem, vis, llm_caller=mock_llm)
        since = time.time()
        res = await b.run_turn("c1", "我昨天几点到图书馆", guest_mode=True,
                               guest_since_ts=since)
        assert res.reply_text, "访客也应正常回复"

        blob = json.dumps(captured["messages"], ensure_ascii=False)
        assert "[GUEST_MODE]" in blob, "缺少访客提示"
        assert "[KNOWN_FACTS" not in blob, "访客上下文泄露了 KNOWN_FACTS"
        assert "昨天下午三点到了图书馆" not in blob, "访客上下文泄露了owner历史"
        assert "你昨天三点到图书馆" not in blob, "访客上下文泄露了owner历史回复"

        # 工具硬拦截
        blocked = ["search_memory", "browse_vision_timeline", "inspect_past_frames",
                   "save_user_fact", "manage_todo", "list_ha_entities",
                   "call_home_assistant", "run_custom_tool"]
        for name in blocked:
            out = await b._exec_tool("c1", name, {"query": "图书馆"}, [], guest=True)
            assert GUEST_BLOCK_MSG in out, "%s 未被拦截: %s" % (name, out)

        # control_glasses：只放亮度/音量
        ok = await b._exec_tool("c1", "control_glasses", {"action": "set_brightness", "value": "30"}, [], guest=True)
        assert GUEST_BLOCK_MSG not in ok, "set_brightness 不该被拦"
        bad = await b._exec_tool("c1", "control_glasses", {"action": "open_app", "value": "com.x"}, [], guest=True)
        assert GUEST_BLOCK_MSG in bad, "open_app 必须被拦"

        # 白名单工具可用
        allow = await b._exec_tool("c1", "get_weather", {}, [], guest=True)
        assert GUEST_BLOCK_MSG not in allow, "get_weather 不该被拦"

        # 访客对话不得进入owner可读上下文
        recent = mem.get_recent_turns("c1", limit=20)
        assert all("我昨天几点到图书馆" not in t["text"] for t in recent), "访客话术污染了owner记忆"

        # owner回归后恢复全部能力
        GUEST = False
        back = await b._exec_tool("c1", "list_ha_entities", {"domain": "light"}, [], guest=False)
        assert GUEST_BLOCK_MSG not in back, "owner模式仍被拦截"
        print("TOOLS + CONTEXT OK")


async def main() -> None:
    test_state_machine()
    await test_tools_and_context()
    print("ALL GUEST MODE SMOKE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
