"""MOSS-Live-inspired Dual-Brain Full-Duplex Coordinator for vox_v2.

Architecture (Foreground Fast Brain + Background Worker + Coordination Layer):
  - Foreground (Halogen reasoning_effort="none", ~7ms TTFT): stays non-blocking,
    handles live conversation, immediate acknowledgments, and dispatches heavy
    multi-step work (HA control, memory/vision search, weather/nav, ToolForge)
    to background via ``dispatch_bg_task``.
  - Background Worker (Halogen reasoning_effort="medium", separate FAEX1 slot):
    executes the full multi-step tool loop asynchronously and writes concise
    factual results + HUD/NAV patches back into the coordinator.
  - Coordination Layer (DuplexCoordinator):
      1. Tracks per-client in-flight & completed tasks with intent superseding
         (``cancel`` marks tasks ``superseded`` so stale results never speak).
      2. Exposes a compact ``[BG_TASKS]`` block for the dynamic user tail
         (preserving FAEX1 99.9% KV prefix cache on ``messages[0]``).
      3. Runs a Silence-Gate Interjection Watcher per active session: waits
         until the wearer is idle (not speaking, not thinking, >= silence_gate_s
         since last voice activity) before naturally interjecting results.
      4. Enforces hard Guest Mode isolation (never leaks owner tasks to guests).
      5. Provides a lightweight backchannel guard (``is_backchannel``) so short
         affirmations ("嗯", "哈哈", "哦") during playback don't falsely barge-in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("vox_v2.duplex")

# Short vocal acknowledgments / backchannels that should NOT trigger hard barge-in
# when Vox is actively speaking. Keep strict and short (<= 4 CJK chars after stripping).
_BACKCHANNEL_SET = frozenset({
    "嗯", "嗯嗯", "嗯哼", "哦", "哦哦", "噢", "啊", "好", "好的", "行", "对",
    "对对", "对的", "是", "是的", "哈哈", "哈哈哈", "呵呵", "明白", "知道了",
    "了解", "收到", "可以", "不错", "厉害", "牛", "ok", "okay", "yeah", "yep",
    "uh-huh", "aha", "hmm",
})
_PUNCT_RE = re.compile(r"[\s，,。\.！!？\?、~～…—\-]+")


def is_backchannel(text: str) -> bool:
    """Return True if *text* is purely a brief backchannel acknowledgment."""
    cleaned = _PUNCT_RE.sub("", (text or "").strip()).lower()
    if not cleaned:
        return True
    return cleaned in _BACKCHANNEL_SET


@dataclass
class BgTask:
    """Lifecycle record for a background worker task."""

    task_id: str
    client_id: str
    goal: str
    need_vision: bool = False
    status: str = "queued"  # queued | running | done | failed | superseded
    created_ts: float = field(default_factory=time.time)
    started_ts: float = 0.0
    finished_ts: float = 0.0
    result_text: str = ""
    error: str = ""
    hud_patches: list[dict[str, Any]] = field(default_factory=list)
    nav_patches: list[dict[str, Any]] = field(default_factory=list)
    announced: bool = False
    _async_task: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)

    def is_terminal(self) -> bool:
        return self.status in ("done", "failed", "superseded")


WorkerResult = tuple[str, list[dict[str, Any]], list[dict[str, Any]]]
WorkerFactory = Callable[[BgTask], Awaitable[WorkerResult]]
InterjectCallback = Callable[[BgTask], Awaitable[None]]


class DuplexCoordinator:
    """Coordinates foreground fast brain, background workers, and silence-gate interjections."""

    def __init__(
        self,
        *,
        silence_gate_s: float | None = None,
        poll_interval_s: float = 0.35,
        max_concurrent_per_client: int = 2,
        max_history_per_client: int = 8,
        task_timeout_s: float = 45.0,
    ) -> None:
        self.silence_gate_s = float(
            silence_gate_s
            if silence_gate_s is not None
            else os.getenv("VOX_INTERJECT_SILENCE_S", "1.8")
        )
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.max_concurrent_per_client = max(1, int(max_concurrent_per_client))
        self.max_history_per_client = max(4, int(max_history_per_client))
        self.task_timeout_s = max(5.0, float(task_timeout_s))

        self._tasks: dict[str, list[BgTask]] = {}
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _client_tasks(self, client_id: str) -> list[BgTask]:
        return self._tasks.setdefault(client_id, [])

    def _client_sem(self, client_id: str) -> asyncio.Semaphore:
        sem = self._semaphores.get(client_id)
        if sem is None:
            sem = asyncio.Semaphore(self.max_concurrent_per_client)
            self._semaphores[client_id] = sem
        return sem

    def _prune(self, client_id: str) -> None:
        items = self._client_tasks(client_id)
        if len(items) <= self.max_history_per_client:
            return
        # Always preserve running/queued or unannounced finished tasks.
        keep: list[BgTask] = []
        done_announced: list[BgTask] = []
        for t in items:
            if (not t.is_terminal()) or (t.status in ("done", "failed") and not t.announced):
                keep.append(t)
            else:
                done_announced.append(t)
        slots = max(0, self.max_history_per_client - len(keep))
        self._tasks[client_id] = done_announced[-slots:] + keep if slots > 0 else keep

    def dispatch(
        self,
        client_id: str,
        goal: str,
        worker_factory: WorkerFactory,
        *,
        need_vision: bool = False,
        guest_mode: bool = False,
    ) -> BgTask:
        """Queue and start a background task without blocking the foreground turn."""
        if guest_mode:
            raise PermissionError("GUEST_MODE_BLOCKED: 访客模式下禁止派发后台隐私任务")

        goal_clean = (goal or "").strip()
        if not goal_clean:
            raise ValueError("goal cannot be empty")

        task_id = f"bg_{uuid.uuid4().hex[:6]}"
        rec = BgTask(
            task_id=task_id,
            client_id=client_id,
            goal=goal_clean,
            need_vision=bool(need_vision),
            status="queued",
        )
        self._client_tasks(client_id).append(rec)
        self._prune(client_id)

        async def _runner() -> None:
            sem = self._client_sem(client_id)
            async with sem:
                if rec.status == "superseded":
                    return
                rec.status = "running"
                rec.started_ts = time.time()
                log.info("BG_START id=%s client=%s goal=%r", rec.task_id, client_id, rec.goal[:60])
                try:
                    reply, hud_patches, nav_patches = await asyncio.wait_for(
                        worker_factory(rec), timeout=self.task_timeout_s
                    )
                    if rec.status == "superseded":
                        log.info("BG_DISCARD_SUPERSEDED id=%s", rec.task_id)
                        return
                    rec.result_text = (reply or "").strip()
                    rec.hud_patches = list(hud_patches or [])
                    rec.nav_patches = list(nav_patches or [])
                    rec.status = "done"
                    rec.finished_ts = time.time()
                    log.info(
                        "BG_DONE id=%s elapsed=%.2fs result=%r",
                        rec.task_id,
                        rec.finished_ts - rec.started_ts,
                        rec.result_text[:60],
                    )
                except asyncio.CancelledError:
                    if rec.status != "superseded":
                        rec.status = "superseded"
                    rec.finished_ts = time.time()
                    log.info("BG_CANCELLED id=%s", rec.task_id)
                    raise
                except Exception as exc:
                    if rec.status == "superseded":
                        return
                    rec.status = "failed"
                    rec.error = str(exc)[:160]
                    rec.result_text = f"刚才那个后台任务没跑通：{rec.error[:40]}"
                    rec.finished_ts = time.time()
                    log.warning("BG_FAIL id=%s err=%s", rec.task_id, rec.error)

        rec._async_task = asyncio.create_task(_runner(), name=f"vox-{task_id}")
        return rec

    def cancel(self, client_id: str, task_id: str = "", *, reason: str = "user_cancelled") -> list[str]:
        """Mark matching in-flight or unannounced tasks as superseded and cancel them."""
        cancelled: list[str] = []
        target = (task_id or "").strip()
        for rec in self._client_tasks(client_id):
            if target and rec.task_id != target:
                continue
            if rec.status in ("queued", "running") or (rec.status in ("done", "failed") and not rec.announced):
                rec.status = "superseded"
                rec.announced = True
                rec.error = reason
                rec.finished_ts = time.time()
                if rec._async_task is not None and not rec._async_task.done():
                    rec._async_task.cancel()
                cancelled.append(rec.task_id)
        if cancelled:
            log.info("BG_SUPERSEDED client=%s ids=%s reason=%s", client_id, cancelled, reason)
        return cancelled

    def list_tasks(self, client_id: str, *, include_announced: bool = False) -> list[BgTask]:
        out: list[BgTask] = []
        for rec in self._client_tasks(client_id):
            if include_announced or (not rec.is_terminal()) or (not rec.announced):
                out.append(rec)
        return out

    def format_context_block(self, client_id: str, *, guest_mode: bool = False) -> str:
        """Build a concise [BG_TASKS] dynamic tail for the final user message."""
        if guest_mode:
            return ""
        lines: list[str] = []
        for rec in self._client_tasks(client_id):
            if rec.status in ("queued", "running"):
                lines.append(f"- [进行中 #{rec.task_id}] {rec.goal}")
            elif rec.status == "done" and not rec.announced and rec.result_text:
                lines.append(f"- [刚完成未播报 #{rec.task_id}] {rec.goal} -> {rec.result_text}")
            elif rec.status == "failed" and not rec.announced:
                lines.append(f"- [刚失败未播报 #{rec.task_id}] {rec.goal} -> {rec.error or '执行失败'}")
        if not lines:
            return ""
        return (
            "[BG_TASKS]\n"
            + "\n".join(lines)
            + "\n(若上面有「刚完成未播报」的结果，可在本次回复里顺口带一句并无需重复派发；"
            "若用户改口说不要做了，请调用 cancel_bg_task。)"
        )

    def mark_unannounced_consumed(self, client_id: str, task_ids: list[str] | None = None) -> list[str]:
        """Mark finished tasks as announced once the foreground reply has reported them."""
        marked: list[str] = []
        allowed = set(task_ids) if task_ids is not None else None
        for rec in self._client_tasks(client_id):
            if rec.status in ("done", "failed") and not rec.announced:
                if allowed is None or rec.task_id in allowed:
                    rec.announced = True
                    marked.append(rec.task_id)
        return marked

    def pop_ready_interjection(
        self,
        client_id: str,
        *,
        is_busy: bool,
        last_activity_ts: float,
        now: float | None = None,
        guest_mode: bool = False,
    ) -> BgTask | None:
        """Return the oldest completed unannounced task if the silence gate is open."""
        if is_busy or guest_mode:
            return None
        ts = time.time() if now is None else float(now)
        if last_activity_ts > 0 and (ts - last_activity_ts) < self.silence_gate_s:
            return None
        for rec in self._client_tasks(client_id):
            if rec.status in ("done", "failed") and not rec.announced:
                rec.announced = True
                return rec
        return None

    def start_watcher(
        self,
        client_id: str,
        *,
        is_busy_fn: Callable[[], bool],
        last_activity_fn: Callable[[], float],
        guest_mode_fn: Callable[[], bool],
        on_interject: InterjectCallback,
    ) -> asyncio.Task[None]:
        """Start a lightweight background watcher that interjects when the silence gate opens."""
        self.stop_watcher(client_id)

        async def _loop() -> None:
            while True:
                await asyncio.sleep(self.poll_interval_s)
                try:
                    ready = self.pop_ready_interjection(
                        client_id,
                        is_busy=bool(is_busy_fn()),
                        last_activity_ts=float(last_activity_fn()),
                        guest_mode=bool(guest_mode_fn()),
                    )
                    if ready is not None:
                        await on_interject(ready)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("BG_WATCHER err client=%s %s", client_id, exc)

        task = asyncio.create_task(_loop(), name=f"vox-duplex-watch-{client_id}")
        self._watchers[client_id] = task
        return task

    def stop_watcher(self, client_id: str) -> None:
        task = self._watchers.pop(client_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def shutdown(self, client_id: str | None = None) -> None:
        """Stop watchers and cancel in-flight tasks on disconnect or server shutdown."""
        clients = [client_id] if client_id is not None else list(self._tasks.keys())
        for cid in clients:
            self.stop_watcher(cid)
            self.cancel(cid, reason="session_shutdown")
