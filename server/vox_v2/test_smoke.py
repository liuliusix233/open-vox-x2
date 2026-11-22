"""End-to-end smoke test for vox_v2 (DB isolation, TTL, wire framing, and tool-calling loop)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .audio import FT_CONTROL, pack_control, unpack_frame
from .brain import VoxBrain
from .config import VoxConfig
from .memory import MemoryStore
from .vision import VisionStore


async def run_smoke_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_v2_test_") as tmp:
        tmp_path = Path(tmp)
        db_file = str(tmp_path / "vox_unified.db")
        frame_dir = str(tmp_path / "frames")

        # 1. Binary wire framing
        pkt = pack_control(10042, "hud_patch", lines=["test"])
        ftype, epoch, payload = unpack_frame(pkt)
        assert ftype == FT_CONTROL and epoch == 10042
        obj = json.loads(payload.decode("utf-8"))
        assert obj["type"] == "HUD_PATCH", "CONTROL type must be forced uppercase"

        # 2. MemoryStore: epoch persistence, strict client_id isolation, event_ts window
        mem = MemoryStore(db_file)
        e1 = mem.next_epoch()
        e2 = mem.next_epoch()
        assert e2 == e1 + 1 and e1 >= 10001

        ts_breakfast = 1763506800.0  # 2025-11-19 07:00 CST (example timestamp)
        mem.save_fact("voxx2", "meal:breakfast", "麦当劳小汉堡配麦咖啡", event_ts=ts_breakfast)
        mem.save_fact("voxx2", "fact:pet", "两只猫：一只灰猫，一只白棕长毛猫")

        hit = mem.search_memory("voxx2", start_ts=ts_breakfast - 3600, end_ts=ts_breakfast + 3600)
        assert hit["found"] and "麦当劳" in hit["hits"][0]["text"]

        isolated = mem.search_memory("other_client", start_ts=ts_breakfast - 3600, end_ts=ts_breakfast + 3600)
        assert not isolated["found"], "Cross-client leak detected!"

        # 3. VisionStore: single source of truth + TTL expiration
        vis = VisionStore(frame_dir, ttl_s=300)
        assert vis.get_latest_bytes("voxx2") is None
        rec = vis.ingest("voxx2", b"RIFFFAKEWEBP", {"fmt": "webp", "w": 1280, "h": 960})
        raw, mime = vis.get_latest_bytes("voxx2") or (b"", "")
        assert raw == b"RIFFFAKEWEBP" and mime == "image/webp"
        assert "当前实时画面" in vis.format_live_view_header("voxx2")
        rec.ts = time.time() - 600  # Simulate 10 min stale frame
        assert "已过期" in vis.format_live_view_header("voxx2")
        assert vis.get_fresh_image_b64("voxx2") is None

        # 4. Brain: zero regex router + native tool calling + uppercase HUD_PATCH
        call_count = 0

        async def mock_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            if last_user.startswith("北京时间"):
                # 2026-11-22 KV 缓存契约：system 必须保持纯静态，动态上下文挂在最后一条 user 上
                assert "[CURRENT_TIME |" not in messages[0]["content"], "system 被污染"
                assert "[CURRENT_TIME |" in messages[-1]["content"], "动态上下文没拼到 user"
                return {"role": "assistant", "content": "现在是北京时间测试时间。"}
            if "吃了什么" in str(last_user) and messages[-1]["role"] == "user":
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_memory",
                                "arguments": json.dumps({"query": "麦当劳"}),
                            },
                        }
                    ],
                }
            return {"role": "assistant", "content": "记录显示你吃了麦当劳小汉堡配麦咖啡。"}

        cfg = VoxConfig(db_path=db_file, frame_dir=frame_dir)
        brain = VoxBrain(cfg, mem, vis, llm_caller=mock_llm)

        r1 = await brain.run_turn("voxx2", "北京时间", epoch_id=e2)
        assert "北京时间" in r1.reply_text and not r1.tools_called

        r2 = await brain.run_turn("voxx2", "11月19号早上我吃了什么", epoch_id=mem.next_epoch())
        assert "search_memory" in r2.tools_called
        assert len(r2.hud_patches) == 1 and r2.hud_patches[0]["type"] == "HUD_PATCH"
        assert "麦当劳" in r2.reply_text

        mem.close()
        print("ALL vox_v2 SMOKE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_smoke_tests())
