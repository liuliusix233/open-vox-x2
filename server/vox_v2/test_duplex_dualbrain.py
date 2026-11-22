"""Automated smoke suite #9 for MOSS-Live-style Dual-Brain Full-Duplex Coordinator."""

from __future__ import annotations

import asyncio
import time

from .duplex_coordinator import BgTask, DuplexCoordinator, is_backchannel


async def run_all_tests() -> None:
    # 1. Fast non-blocking dispatch + concurrent foreground chat while worker runs
    coord = DuplexCoordinator(silence_gate_s=0.15, poll_interval_s=0.05)
    worker_started = asyncio.Event()
    worker_release = asyncio.Event()

    async def slow_ha_worker(task: BgTask):
        worker_started.set()
        await worker_release.wait()
        return (
            "游戏房窗帘已经帮你关好了。",
            [{"type": "HUD_PATCH", "title": "Home Assistant", "lines": ["游戏房窗帘: 已关闭"]}],
            [],
        )

    t0 = time.monotonic()
    rec = coord.dispatch("voxx2", "关闭游戏房窗帘", slow_ha_worker)
    dispatch_ms = (time.monotonic() - t0) * 1000.0
    assert dispatch_ms < 50.0, f"dispatch blocked foreground: {dispatch_ms:.1f}ms"

    await asyncio.wait_for(worker_started.wait(), timeout=1.0)
    assert rec.status == "running"

    # Foreground sees [BG_TASKS] while worker is still running
    ctx_running = coord.format_context_block("voxx2")
    assert "[BG_TASKS]" in ctx_running and "进行中" in ctx_running and "关闭游戏房窗帘" in ctx_running
    # Guest mode must NEVER see owner's tasks
    assert coord.format_context_block("voxx2", guest_mode=True) == ""

    # 2. Silence-gate interjection: holds while busy, fires ONCE when silence gate opens
    busy_flag = True
    last_act = time.time()
    interjected: list[BgTask] = []
    interjected_ev = asyncio.Event()

    async def on_interject(task: BgTask) -> None:
        interjected.append(task)
        interjected_ev.set()

    coord.start_watcher(
        "voxx2",
        is_busy_fn=lambda: busy_flag,
        last_activity_fn=lambda: last_act,
        guest_mode_fn=lambda: False,
        on_interject=on_interject,
    )

    # Let worker finish while user is still busy talking
    worker_release.set()
    await asyncio.sleep(0.12)
    assert rec.status == "done"
    assert len(interjected) == 0, "must NOT interject while user is busy"

    # User stops speaking, wait past silence_gate_s -> watcher fires ONCE
    busy_flag = False
    last_act = time.time() - 0.25
    await asyncio.wait_for(interjected_ev.wait(), timeout=1.0)
    assert len(interjected) == 1 and interjected[0].task_id == rec.task_id
    assert interjected[0].result_text == "游戏房窗帘已经帮你关好了。"
    assert len(interjected[0].hud_patches) == 1

    # Ensure zero duplicate interjections on subsequent poll ticks
    await asyncio.sleep(0.15)
    assert len(interjected) == 1, "interjection fired more than once!"

    # 3. Intent superseding (user changes mind mid-flight -> cancelled, never speaks)
    cancel_started = asyncio.Event()

    async def cancellable_worker(task: BgTask):
        cancel_started.set()
        await asyncio.sleep(5.0)
        return ("应该永远不会被播报", [], [])

    rec2 = coord.dispatch("voxx2", "查明早去机场路线", cancellable_worker)
    await asyncio.wait_for(cancel_started.wait(), timeout=1.0)
    cancelled_ids = coord.cancel("voxx2", rec2.task_id, reason="user_changed_mind")
    assert cancelled_ids == [rec2.task_id]
    assert rec2.status == "superseded" and rec2.announced is True
    await asyncio.sleep(0.15)
    assert len(interjected) == 1, "superseded task must never interject"

    # 4. Foreground consumption prevents duplicate watcher interjection
    coord.stop_watcher("voxx2")

    async def fast_worker(task: BgTask):
        return ("Shanghai今天晴，18度。", [], [])

    rec3 = coord.dispatch("voxx2", "查Shanghai天气", fast_worker)
    await asyncio.sleep(0.08)
    assert rec3.status == "done" and not rec3.announced
    ctx_done = coord.format_context_block("voxx2")
    assert "刚完成未播报" in ctx_done and "Shanghai今天晴" in ctx_done
    marked = coord.mark_unannounced_consumed("voxx2")
    assert rec3.task_id in marked
    assert coord.pop_ready_interjection("voxx2", is_busy=False, last_activity_ts=0.0) is None

    # 5. Guest Mode hard block on dispatch
    guest_blocked = False
    try:
        coord.dispatch("voxx2", "关家里所有灯", fast_worker, guest_mode=True)
    except PermissionError:
        guest_blocked = True
    assert guest_blocked, "Guest mode must hard-block private background dispatch"

    # 6. Worker failure graceful handling
    async def failing_worker(task: BgTask):
        raise RuntimeError("HA timeout")

    rec4 = coord.dispatch("voxx2", "开空调", failing_worker)
    await asyncio.sleep(0.08)
    assert rec4.status == "failed" and "HA timeout" in rec4.error
    ready_err = coord.pop_ready_interjection("voxx2", is_busy=False, last_activity_ts=0.0)
    assert ready_err is not None and ready_err.task_id == rec4.task_id

    # 7. Backchannel vs genuine barge-in
    for bc in ("嗯", "嗯嗯", "哦", "哈哈", "好的。", "对的！", "ok"):
        assert is_backchannel(bc), f"expected backchannel: {bc!r}"
    for real in ("等一下", "停", "帮我关灯", "换一个笑话", "不对，不是那个"):
        assert not is_backchannel(real), f"false backchannel: {real!r}"

    await coord.shutdown()
    print("ALL DUPLEX DUAL-BRAIN SMOKE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
