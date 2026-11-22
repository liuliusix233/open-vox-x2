"""agency_loop_v2.py — 3-step silent-investigation proactive Agent loop.

Upgrades Phase 1 AgencyLoop:
  1. Reuses brain.py's exact static SYSTEM_PROMPT & unified TOOLS_SCHEMA so FAEX1
     maintains 99.9% KV prefix cache (zero 504 stalls).
  2. Supports up to 3 tool steps per heartbeat: the model can silently inspect
     get_weather / search_memory / inspect_recent_frames / inspect_past_frames
     first, then decide between do_nothing, proactive_speak, or start_navigation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger("vox_v2.agency")

# ---------------------------------------------------------------------------
# 2026-11-22 三道硬锁（进程级，跨实例共享）
#   1) 单例互斥：同一 client_id 全局只允许一个活跃 AgencyLoopV2
#   2) 冷却 + 原文去重：两次 [主动] 至少间隔 90s（到期 todo 例外）；
#      10 分钟内与最近 3 条 [主动] 字面重合度 > 60% 直接丢弃
#   3) 单次即止：proactive_speak 执行完立刻 return，不进下一轮
# ---------------------------------------------------------------------------
_ACTIVE: dict[str, "AgencyLoopV2"] = {}
_SPOKEN_TEXTS: deque[str] = deque(maxlen=3)
_SPOKEN_TS: float = 0.0
COOLDOWN_S = float(os.environ.get("VOX_PROACTIVE_COOLDOWN_S", "90"))
DEDUP_WINDOW_S = float(os.environ.get("VOX_PROACTIVE_DEDUP_S", "600"))
DEDUP_RATIO = 0.6

# 模型偶尔会把「[主动] xxx」当成正文吐出来，这个标记只属于主动播报通道
_MARKER_RE = re.compile(r"^\s*[\[【]\s*(?:主动|AGENCY|心跳|后台)[^\]】]*[\]】]\s*")
_META_RE = re.compile(r"^\s*[（(][^）)]{0,40}[）)]\s*$")


def strip_proactive_marker(text: str) -> str:
    """去掉正文里混进来的 [主动] 之类通道标记。"""
    out = _MARKER_RE.sub("", text or "")
    return out.strip()


def _bigrams(s: str) -> set[str]:
    s = "".join(ch for ch in (s or "") if not ch.isspace())
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def text_similar(a: str, b: str) -> float:
    """字面重合度（字符二元组 Jaccard）。"""
    x, y = _bigrams(a), _bigrams(b)
    if not x or not y:
        return 0.0
    return len(x & y) / float(len(x | y))


def proactive_allowed(text: str, *, due: bool = False, now: float | None = None) -> tuple[bool, str]:
    """冷却 + 原文去重闸门。返回 (是否放行, 原因)。"""
    ts = time.time() if now is None else float(now)
    if not (text or "").strip():
        return False, "empty"
    if not due and _SPOKEN_TS > 0 and (ts - _SPOKEN_TS) < COOLDOWN_S:
        return False, "cooldown_%.0fs" % (COOLDOWN_S - (ts - _SPOKEN_TS))
    if _SPOKEN_TS > 0 and (ts - _SPOKEN_TS) <= DEDUP_WINDOW_S:
        for prev in _SPOKEN_TEXTS:
            if text_similar(text, prev) > DEDUP_RATIO:
                return False, "duplicate"
    return True, "ok"


def note_proactive_spoken(text: str, *, now: float | None = None) -> None:
    global _SPOKEN_TS
    _SPOKEN_TS = time.time() if now is None else float(now)
    t = (text or "").strip()
    if t and (not _SPOKEN_TEXTS or _SPOKEN_TEXTS[-1] != t):
        _SPOKEN_TEXTS.append(t)
TZ_SHANGHAI = timezone(timedelta(hours=8))

AGENCY_CONTROL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "description": "后台心跳观察拍专用：查证后认为当前无需打扰用户时调用，保持安静。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "简短静默原因"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proactive_speak",
            "description": "后台心跳观察拍专用：确认有真正值得说的信息（关心穿衣/到期提醒/目标确认/新奇细节）时主动播报 1–2 句自然口语。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要播报给用户的自然口语内容，限1–2句"},
                    "reason": {"type": "string", "description": "简短触发原因"},
                },
                "required": ["text", "reason"],
            },
        },
    },
]

LlmCaller = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
SpeakCallback = Callable[[str, float], Awaitable[None]]
StateProvider = Callable[[], dict[str, Any]]


class AgencyLoopV2:
    """3-step silent-investigation proactive heartbeat sharing the main KV cache."""

    def __init__(
        self,
        client_id: str,
        system_prompt: str,
        unified_tools: list[dict[str, Any]],
        hormones: Any,
        todo_goals: Any,
        llm_caller: LlmCaller,
        tool_executor: ToolExecutor,
        speak_cb: SpeakCallback,
        state_provider: StateProvider,
        min_user_idle_s: float = 25.0,
        max_steps: int = 3,
        enabled: bool | None = None,
    ) -> None:
        self.client_id = client_id
        self.system_prompt = system_prompt
        self.unified_tools = unified_tools
        self.hormones = hormones
        self.todo_goals = todo_goals
        self.llm_caller = llm_caller
        self.tool_executor = tool_executor
        self.speak_cb = speak_cb
        self.state_provider = state_provider
        self.min_user_idle_s = min_user_idle_s
        self.max_steps = max_steps
        self.enabled = (os.environ.get("VOX_AGENCY_ENABLED", "1") != "0") if enabled is None else enabled
        self._task: asyncio.Task[None] | None = None
        self._last_eval_ts: float = 0.0
        self._last_caption_seen: str = ""
        self._pending_causal_text: str = ""
        self._pending_causal_ts: float = 0.0

    # ---- 硬锁 1：单例互斥 ----
    @classmethod
    async def start_singleton(cls, **kwargs: Any) -> "AgencyLoopV2":
        """同一 client_id 全局只保留一个实例：先把旧实例彻底停掉，再起新的。"""
        cid = str(kwargs.get("client_id") or "")
        old = _ACTIVE.get(cid)
        if old is not None:
            log.info("AGENCY_REPLACE client=%s 停掉旧实例", cid)
            await old.stop()
        loop = cls(**kwargs)
        _ACTIVE[cid] = loop
        loop.start()
        log.info("AGENCY_SINGLETON client=%s running=%s active_clients=%d",
                 cid, loop.is_running(), len([k for k, v in _ACTIVE.items() if v.is_running()]))
        return loop

    @classmethod
    async def stop_client(cls, client_id: str) -> None:
        """WS 断开 / 会话销毁时调用，确保连注册表一起停干净。"""
        cid = str(client_id or "")
        loop = _ACTIVE.pop(cid, None)
        if loop is not None:
            await loop.stop()
            log.info("AGENCY_STOPPED client=%s", cid)

    @classmethod
    def active_count(cls, client_id: str | None = None) -> int:
        if client_id is None:
            return sum(1 for v in _ACTIVE.values() if v.is_running())
        v = _ACTIVE.get(str(client_id))
        return 1 if (v is not None and v.is_running()) else 0

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run(), name=f"vox-agency-{self.client_id}")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if _ACTIVE.get(self.client_id) is self:
            _ACTIVE.pop(self.client_id, None)

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(5.0)
                await self.step_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("agency_v2 tick failed: %s", exc)

    async def step_once(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        st = self.state_provider()
        if st.get("busy"):
            return {"action": "skip", "reason": "session_busy"}

        # 2026-11-22 三模式门控：SLEEP_LISTEN 完全不评估；STANDBY 只放行到期 todo
        mode = str(st.get("mode") or "LIVE").upper()
        if mode in ("SLEEP_LISTEN", "SLEEP"):
            return {"action": "skip", "reason": "sleep"}
        if st.get("guest_mode"):
            return {"action": "skip", "reason": "guest_mode"}

        # 不插电（BAT）时按客户端给的倍率拉长节拍
        hb_mult = float(st.get("heartbeat_mult") or 1.0)
        power = str(st.get("power_profile") or "AC").upper()

        last_user_ts = float(st.get("last_user_ts") or getattr(self.hormones, "last_user_ts", 0.0) or 0.0)
        idle_s = (now - last_user_ts) if last_user_ts > 0 else 999.0
        self.hormones.on_idle_tick(idle_s, now=now)

        # Settle causal feedback 35s after previous proactive speech
        if self._pending_causal_ts > 0 and (now - self._pending_causal_ts) >= 35.0:
            replied = last_user_ts >= self._pending_causal_ts
            if not replied:
                self.hormones.on_proactive_ignored(now=now)
            self.hormones.log_causal_turn(
                trigger="proactive",
                reply_text=self._pending_causal_text,
                user_replied_in_35s=replied,
            )
            self._pending_causal_ts = 0.0
            self._pending_causal_text = ""

        due_todos = self.todo_goals.get_due_todos(self.client_id, within_min=15.0)
        if mode == "STANDBY" and not due_todos:
            return {"action": "skip", "reason": "standby_quiet"}
        if due_todos:
            urgent = any(t.get("priority") == "high" for t in due_todos)
            self.hormones.on_todo_due(urgent=urgent, now=now)

        latest_cap = str(st.get("latest_caption") or "").strip()
        if latest_cap and latest_cap != self._last_caption_seen:
            if self._last_caption_seen:
                self.hormones.on_scene_shift(0.55, now=now)
            self._last_caption_seen = latest_cap

        if not force:
            if idle_s < self.min_user_idle_s:
                return {"action": "skip", "reason": "user_recent"}
            interval = self.hormones.proactive_interval_s(now=now)
            if interval is None and not due_todos:
                return {"action": "skip", "reason": "hormone_muted"}
            effective_interval = (30.0 if due_todos else (interval or 95.0))
            if power == "BAT":
                effective_interval *= max(1.0, hb_mult)
            if (now - self._last_eval_ts) < effective_interval:
                return {"action": "skip", "reason": "cooldown"}

        self._last_eval_ts = now
        return await self._investigate_and_decide(st, due_todos, now)

    async def _investigate_and_decide(
        self, st: dict[str, Any], due_todos: list[dict[str, Any]], now: float
    ) -> dict[str, Any]:
        bj = datetime.fromtimestamp(now, tz=TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M %A")
        tail_parts = [
            "[AGENCY_HEARTBEAT | 后台静默观察拍]",
            "你可以先静默调用只读工具（如 get_weather、search_memory、inspect_recent_frames）核实细节；"
            "若确认无需打扰请调 do_nothing；若真有价值请调 proactive_speak（限1–2句）或直接 start_navigation / manage_goal。",
            f"[CURRENT_TIME] {bj}",
        ]
        directive = self.hormones.dynamic_tail_directive(is_proactive=True, now=now)
        if directive:
            tail_parts.append(directive)
        if st.get("latest_caption"):
            tail_parts.append(f"[LATEST_SCENE] {st['latest_caption']}")
        tg_block = self.todo_goals.format_context_block(self.client_id)
        if tg_block:
            tail_parts.append(tg_block)
        if due_todos:
            tail_parts.append("[DUE_SOON] " + "；".join(f"{t['title']}({t['due']})" for t in due_todos))
        recent = st.get("recent_turns") or []
        if recent:
            tail_parts.append(
                "[RECENT_DIALOG]\n" + "\n".join(f"{r['role']}: {str(r['text'])[:80]}" for r in recent[-4:])
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(tail_parts)},
        ]

        tools_trace: list[str] = []
        for _step in range(self.max_steps):
            if self.state_provider().get("busy"):
                return {"action": "skip", "reason": "preempted_by_user", "tools": tools_trace}
            msg = await self.llm_caller(messages, self.unified_tools)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return {"action": "do_nothing", "reason": "no_tool_call", "tools": tools_trace}

            messages.append(msg)
            terminal_result: dict[str, Any] | None = None

            for tc in tool_calls:
                tc_id = tc.get("id") or f"ag_{_step}"
                fn = tc.get("function") or {}
                name = str(fn.get("name", ""))
                raw_args = fn.get("arguments", "{}")
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                tools_trace.append(name)

                if name == "do_nothing":
                    terminal_result = {
                        "action": "do_nothing",
                        "reason": str(args.get("reason", "")),
                        "tools": tools_trace,
                    }
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": '{"ok":true}'})
                elif name == "proactive_speak":
                    text = strip_proactive_marker(str(args.get("text", "")).strip())
                    if text and not self.state_provider().get("busy"):
                        # 硬锁 2：冷却 + 原文去重（到期 todo 可越过冷却）
                        ok, why = proactive_allowed(text, due=bool(due_todos))
                        if not ok:
                            log.info("PROACTIVE_DROP reason=%s text=%r", why, text[:40])
                            terminal_result = {
                                "action": "proactive_dropped",
                                "reason": why,
                                "text": text,
                                "tools": tools_trace,
                            }
                        else:
                            style = self.hormones.tts_style(now=now)
                            self.hormones.on_proactive_spoken(now=now)
                            note_proactive_spoken(text)
                            self._pending_causal_text = text
                            self._pending_causal_ts = time.time()
                            await self.speak_cb(text, float(style["speed"]))
                            log.info("PROACTIVE_SPOKEN text=%r", text[:40])
                            terminal_result = {
                                "action": "proactive_speak",
                                "text": text,
                                "tts_speed": style["speed"],
                                "tools": tools_trace,
                            }
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": '{"ok":true}'})
                        # 硬锁 3：单次即止 —— 说过（或被丢弃）立刻收工，绝不进下一轮
                        return terminal_result
                else:
                    # Silent investigation or state update (get_weather / search_memory / start_navigation / etc.)
                    tool_out = await self.tool_executor(name, args)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_out})

            if terminal_result is not None:
                return terminal_result

        return {"action": "do_nothing", "reason": "max_steps_reached", "tools": tools_trace}
