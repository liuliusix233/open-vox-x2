"""hormones_v2.py — Clean 4-transmitter state machine with 3 real behavioral exits.

Exits:
  1. proactive_interval_s(): controls how often agency_loop wakes up (or None to stay silent).
  2. dynamic_tail_directive(): 1-line dynamic context hint (ONLY injected into current-turn
     user tail, NEVER into static system prompt, preserving 99.9% KV prefix cache).
  3. tts_style(): speech speed/mood hints for CosyVoice2 + causal JSONL audit logging.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any

HALF_LIFE_S = {"NE": 45.0, "DA": 100.0, "OXT": 240.0, "HT": 180.0}
CAPS = {"NE": 0.90, "DA": 0.90, "OXT": 0.85, "HT": 0.85}
BASE = {"NE": 0.10, "DA": 0.25, "OXT": 0.12, "HT": 0.20}

SNAPSHOT_PATH = os.environ.get("VOX_HORM_SNAPSHOT", "/tmp/vox_horm.json")
CAUSAL_LOG_PATH = os.environ.get("VOX_HORM_CAUSAL_LOG", "/tmp/vox_horm_causal.jsonl")
ENABLED = os.environ.get("VOX_HORMONES", "1") != "0"

_DIRECTIVES = {
    "DA": "[生理态:兴致高] 先完整回应用户正事，若合适可顺带追问一个你好奇的细节或轻快搭一句，限2句。",
    "HT": "[生理态:克制] 只给最核心结论，不反问、不展开，尽量控制在20字以内。",
    "OXT": "[生理态:关切] 语气熟稔温和，在答完正事之余顺带留意对方当下的状态。",
    "NE": "[生理态:警觉] 直接点出重点、风险或到期事项，短促干脆不绕弯。",
}

_PROACTIVE_DIRECTIVES = {
    "DA": "[主动心境:好奇] 发现有意思的新变化时，简短问一句或点评一个具体细节，限1–2句。",
    "HT": "[主动心境:克制] 除非有明确到期事项或重要变化，否则调用 do_nothing 保持安静。",
    "OXT": "[主动心境:关切] 关注用户当下状态或未完日常，轻声关心一句即可，别说教。",
    "NE": "[主动心境:警觉] 只提醒明确到期、异常或需要马上确认的事，一句话说清。",
}

_TTS_STYLE = {
    "DA": {"mood": "轻快好奇", "speed": 1.06},
    "HT": {"mood": "平淡简短", "speed": 0.95},
    "OXT": {"mood": "温和放慢", "speed": 0.96},
    "NE": {"mood": "紧凑干脆", "speed": 1.10},
    "": {"mood": "自然平稳", "speed": 1.00},
}


class HormonesV2:
    """Thread-safe 4-transmitter dynamics with refractory period & habituation."""

    def __init__(
        self,
        snapshot_path: str = SNAPSHOT_PATH,
        causal_log_path: str = CAUSAL_LOG_PATH,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = ENABLED if enabled is None else enabled
        self.snapshot_path = snapshot_path
        self.causal_log_path = causal_log_path
        self._lock = threading.RLock()
        self.v: dict[str, float] = dict(BASE)
        self._last_t: float = time.time()
        self._last_flush: float = 0.0
        self._hab: dict[str, float] = {}
        self._last_cat_ts: dict[str, float] = {}

        self.hits: int = 0
        self.last_user_ts: float = 0.0
        self.last_proactive_ts: float = 0.0
        self.unacked_proactive: int = 0
        self.ignored_streak: int = 0
        self._ignored_checked_for: float = 0.0
        self._flush(force=True)

    def _decay_locked(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        dt = max(0.0, min(600.0, now - self._last_t))
        self._last_t = now
        if dt <= 0.0:
            return
        for k, half in HALF_LIFE_S.items():
            self.v[k] = BASE[k] + (self.v[k] - BASE[k]) * (0.5 ** (dt / half))
            self.v[k] = max(0.02, min(CAPS[k], self.v[k]))

    def _apply_deltas_locked(
        self,
        deltas: dict[str, float],
        category: str = "",
        refractory_s: float = 0.0,
        habituate: bool = False,
        now: float | None = None,
    ) -> float:
        """Apply deltas with optional refractory period, habituation, and soft cap."""
        if not self.enabled:
            return 0.0
        now = time.time() if now is None else now
        self._decay_locked(now)

        gain = 1.0
        if category:
            last = self._last_cat_ts.get(category, 0.0)
            dt = now - last if last > 0 else 9999.0
            cnt = self._hab.get(category, 0.0)
            if habituate and dt > 180.0 and cnt > 0:
                cnt *= 0.5 ** ((dt - 180.0) / 120.0)
            refrac = 0.15 if (refractory_s > 0 and dt < refractory_s) else 1.0
            hab = math.exp(-0.55 * cnt) if habituate else 1.0
            gain = refrac * hab
            self._hab[category] = cnt + (0.25 if (refractory_s > 0 and dt < refractory_s) else 1.0)
            self._last_cat_ts[category] = now

        for k, dv in deltas.items():
            if k not in self.v:
                continue
            if dv > 0:
                headroom = max(0.0, 1.0 - self.v[k] / CAPS[k]) ** 1.2
                self.v[k] = min(CAPS[k], self.v[k] + dv * gain * headroom)
            else:
                self.v[k] = max(0.02, self.v[k] + dv * (0.5 + 0.5 * gain))

        self.hits += 1
        self._flush_locked(now)
        return gain

    def _flush(self, force: bool = False) -> None:
        with self._lock:
            self._flush_locked(time.time(), force=force)

    def _flush_locked(self, now: float, force: bool = False) -> None:
        if not force and (now - self._last_flush) < 3.0:
            return
        self._last_flush = now
        if not self.snapshot_path:
            return
        try:
            payload = {
                "ts": round(now, 1),
                "v": {k: round(x, 3) for k, x in self.v.items()},
                "dom_delta": self._dominant_delta_locked(),
                "interval_s": self._proactive_interval_locked(),
                "unacked": self.unacked_proactive,
                "ignored_streak": self.ignored_streak,
                "hits": self.hits,
            }
            tmp = self.snapshot_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self.snapshot_path)
        except Exception:
            pass

    # ---- Real-world event inputs ----

    def on_user_reply(self, text_len: int = 10, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            gap = now - self.last_proactive_ts if self.last_proactive_ts > 0 else 9999.0
            self.last_user_ts = now
            self.unacked_proactive = 0
            self.ignored_streak = 0
            if 0 < gap <= 35.0:
                # User engaged with proactive utterance -> strong positive reinforcement
                boost = 1.25 if text_len >= 12 else 1.0
                self._apply_deltas_locked(
                    {"DA": 0.34 * boost, "OXT": 0.14 * boost, "HT": -0.10},
                    now=now,
                )
            else:
                # Normal user-initiated turn
                self._apply_deltas_locked(
                    {"DA": 0.12, "OXT": 0.08, "HT": -0.05},
                    now=now,
                )

    def on_barge_in(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.last_user_ts = now
            self.unacked_proactive = 0
            self._apply_deltas_locked({"HT": 0.20, "DA": -0.15}, now=now)

    def on_proactive_spoken(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.last_proactive_ts = now
            self.unacked_proactive += 1
            deltas = {"DA": -0.10}
            if self.unacked_proactive >= 2:
                deltas["HT"] = 0.28
                deltas["DA"] = -0.20
            self._apply_deltas_locked(deltas, now=now)

    def on_proactive_ignored(self, now: float | None = None) -> bool:
        """Call ~35s after proactive speech if user never replied. Returns True once per utterance."""
        now = time.time() if now is None else now
        with self._lock:
            ts = self.last_proactive_ts
            if ts <= 0 or abs(self._ignored_checked_for - ts) < 0.5:
                return False
            if self.last_user_ts >= ts:
                self._ignored_checked_for = ts
                return False
            self._ignored_checked_for = ts
            self.ignored_streak += 1
            self._apply_deltas_locked({"HT": 0.18, "DA": -0.10}, now=now)
            return True

    def on_device_alert(self, now: float | None = None) -> float:
        """设备异常（低电/高温）→ 一次 NE 脉冲，让心跳自然提醒。"""
        return self._apply_deltas_locked(
            {"NE": 0.16}, category="device_alert", refractory_s=600.0, habituate=True, now=now
        )

    def on_scene_shift(self, magnitude: float = 0.6, now: float | None = None) -> float:
        """Significant visual scene transition (e.g. stepping outdoors, new room)."""
        mag = max(0.0, min(1.0, magnitude))
        with self._lock:
            return self._apply_deltas_locked(
                {"DA": 0.22 * mag, "NE": 0.14 * mag},
                category="scene_shift",
                refractory_s=20.0,
                habituate=True,
                now=now,
            )

    def on_todo_due(self, urgent: bool = False, now: float | None = None) -> float:
        with self._lock:
            deltas = {"NE": 0.34, "DA": 0.08} if urgent else {"NE": 0.18, "DA": 0.12}
            return self._apply_deltas_locked(
                deltas,
                category="todo_due",
                refractory_s=60.0,
                habituate=False,
                now=now,
            )

    def on_goal_completed(self, now: float | None = None) -> None:
        with self._lock:
            self._apply_deltas_locked({"DA": 0.18, "OXT": 0.10, "NE": -0.12}, now=now)

    def on_idle_tick(self, idle_s: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._decay_locked(now)
            if not self.enabled:
                return
            if idle_s >= 180.0:
                self.v["HT"] = min(CAPS["HT"], self.v["HT"] + 0.008)
                self.v["DA"] = max(0.02, self.v["DA"] - 0.005)
            self._flush_locked(now)

    # ---- Readouts & 3 hard exits ----

    def snap(self, now: float | None = None) -> dict[str, float]:
        with self._lock:
            self._decay_locked(now)
            return {k: round(x, 3) for k, x in self.v.items()}

    def _dominant_delta_locked(self) -> str:
        if not self.enabled:
            return ""
        deltas = {k: self.v[k] - BASE[k] for k in ("NE", "DA", "OXT", "HT")}
        top = max(deltas, key=deltas.get)
        return top if deltas[top] >= 0.12 else ""

    def dominant_delta(self, now: float | None = None) -> str:
        with self._lock:
            self._decay_locked(now)
            return self._dominant_delta_locked()

    def _proactive_interval_locked(self) -> float | None:
        """Exit 1: Heartbeat cadence in seconds (None = stay silent)."""
        if not self.enabled:
            return 90.0
        if self.ignored_streak >= 3 or self.v["HT"] >= 0.70:
            return None
        if self.v["NE"] >= 0.45:
            return 35.0
        if self.v["DA"] >= 0.45:
            return 60.0
        if self.v["HT"] >= 0.42:
            return 180.0
        return 95.0

    def proactive_interval_s(self, now: float | None = None) -> float | None:
        with self._lock:
            self._decay_locked(now)
            return self._proactive_interval_locked()

    def dynamic_tail_directive(self, is_proactive: bool = False, now: float | None = None) -> str:
        """Exit 2: 1-line dynamic tail constraint (never put in static system prompt)."""
        with self._lock:
            self._decay_locked(now)
            dom = self._dominant_delta_locked()
            if not dom:
                return ""
            return _PROACTIVE_DIRECTIVES[dom] if is_proactive else _DIRECTIVES[dom]

    def tts_style(self, now: float | None = None) -> dict[str, Any]:
        """Exit 3: TTS speed/mood mapping for CosyVoice2."""
        dom = self.dominant_delta(now)
        return dict(_TTS_STYLE.get(dom, _TTS_STYLE[""]))

    def log_causal_turn(
        self,
        trigger: str,
        reply_text: str,
        user_replied_in_35s: bool | None = None,
    ) -> dict[str, Any]:
        """Write 1 structured JSONL row to prove hormones measurably change behavior."""
        with self._lock:
            now = time.time()
            self._decay_locked(now)
            dom = self._dominant_delta_locked() or "NEUTRAL"
            style = _TTS_STYLE.get("" if dom == "NEUTRAL" else dom, _TTS_STYLE[""])
            rec = {
                "ts": round(now, 1),
                "trigger": trigger,
                "dom": dom,
                "v": {k: round(x, 3) for k, x in self.v.items()},
                "tts_speed": style["speed"],
                "reply_chars": len(reply_text.strip()),
                "has_question": ("？" in reply_text or "?" in reply_text),
                "user_replied_in_35s": user_replied_in_35s,
            }
        if self.causal_log_path:
            try:
                with open(self.causal_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return rec
