"""Minimal aiohttp server for vox_v2: /ws/v1/duplex, /health, /frame/last, /text."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .audio import (
    FT_AUDIO_DOWN,
    FT_AUDIO_UP,
    FT_CONTROL,
    FT_VISION_UP,
    AsrCallable,
    StreamingVad,
    TtsCallable,
    cosyvoice2_tts_stream,
    gpt_sovits_tts_stream,
    openai_tts_stream,
    default_http_asr,
    pack_control,
    pcm_rms,
    pack_frame,
    unpack_frame,
    unpack_vision_payload,
)
from .brain import VoxBrain, split_sentences
from .duplex_coordinator import BgTask, DuplexCoordinator, is_backchannel
from .config import VoxConfig
from .memory import MemoryStore
from .photo_lib import PhotoLibrary
from .ring_buffer import RingBuffer
from .agency_loop_v2 import (
    AgencyLoopV2,
    note_proactive_spoken,
    proactive_allowed,
    strip_proactive_marker,
)
from .vision import VisionStore

log = logging.getLogger("vox_v2")

# 2026-11-22：/health 的 busy 判定（打标让路用）
_ACTIVITY = {"busy": 0, "last_user_speech": 0.0}


GUEST_ENTER_RATIO = 0.65
GUEST_OWNER_QUIET_S = 20.0
GUEST_EXIT_RATIO = 0.55
GUEST_EXIT_SIM = 0.65
# 2026-11-22：口令必须是"动作 + 访客模式"，裸提"访客模式"不再算指令
# （实测bystander聊天说了句"什么模式？访客模式。"就把owner关进访客模式了）
GUEST_ENTER_RE = re.compile(r"(开启|打开|进入|切成|切到|切换到|启动|开一下)\s*访客\s*模式")
GUEST_EXIT_RE = re.compile(
    r"(退出|关闭|关掉|结束|取消)\s*访客\s*模式"
    r"|访客\s*模式\s*(退出|关闭|关掉|关了|结束|关|停)"
    r"|退出访客"
)
GUEST_BORROW_PHRASES = (
    "借别人戴一下", "借给别人戴", "让别人戴一下", "给别人戴一下", "给朋友戴一下",
)


def guest_intent(text: str) -> str:
    """识别访客模式口令，返回 'enter' / 'exit' / ''。

    进入必须是短句里的「动词+访客模式」，长句和裸词一律不算，避免bystander闲聊误触发。
    """
    t = (text or "").strip()
    if not t:
        return ""
    if GUEST_EXIT_RE.search(t):
        return "exit"
    if len(t) <= 14 and GUEST_ENTER_RE.search(t):
        return "enter"
    if len(t) <= 16 and any(ph in t for ph in GUEST_BORROW_PHRASES):
        return "enter"
    return ""

# 2026-11-22 声纹/音量自动切入访客：默认关（VOX_AUTO_GUEST=0），避免bystander说话大声误切。
# 关闭后只保留：手动口令进入 / 手动口令退出 / owner声纹解锁。
_AUTO_GUEST = os.environ.get("VOX_AUTO_GUEST", "0").strip().lower() not in (
    "0", "false", "no", "off",
)

# 唤醒词（SLEEP_LISTEN 模式生效）：VOX_WAKE_WORDS 逗号分隔，默认只认 "vox"
WAKE_WORDS = tuple(
    w.strip().lower()
    for w in (os.environ.get("VOX_WAKE_WORDS", "vox") or "vox").split(",")
    if w.strip()
)

# 2026-11-22 双脑：轻附和词（嗯/哈哈/哦）不触发打断，默认关闭
_BACKCHANNEL_GUARD = os.environ.get("VOX_BACKCHANNEL_GUARD", "0").strip().lower() in (
    "1", "true", "yes", "on",
)


def _has_wake_word(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in WAKE_WORDS)


def evaluate_guest_state(
    *, speaker: str, spk_sim: float, raw_rms: float, owner_near_rms: float,
    guest_mode: bool, now: float, last_owner_near_ts: float,
    owner_soft_ok: bool = False, text_len: int = 99,
) -> tuple[bool, float]:
    """返回 (新 guest_mode, 新的 last_owner_near_ts)。"""
    base = max(150.0, float(owner_near_rms or 800.0))
    is_owner_confident = (speaker == "owner" and spk_sim >= GUEST_EXIT_SIM
                          and raw_rms >= base * GUEST_EXIT_RATIO)
    if is_owner_confident:
        return False, now
    # 真实环境里owner sim 偶尔落在 0.5–0.65：连续两次软命中即解锁，避免被锁在外面
    if guest_mode and speaker == "owner" and spk_sim >= 0.50 and owner_soft_ok and raw_rms >= base * GUEST_EXIT_RATIO:
        return False, now
    if speaker == "owner" and spk_sim >= 0.55:
        return guest_mode, now
    near = raw_rms >= base * GUEST_ENTER_RATIO
    quiet_owner = (now - last_owner_near_ts) > GUEST_OWNER_QUIET_S
    # 太短的语气词（"呃呵""嗯"）和几乎没有声纹相似度的噪声，不当作"别人在戴"
    if ((not guest_mode) and speaker and speaker != "owner" and near and quiet_owner
            and text_len >= 4 and spk_sim >= 0.18):
        return True, last_owner_near_ts
    return guest_mode, last_owner_near_ts


class DuplexSession:
    """Per-connection 6-state full-duplex WebSocket session."""

    def __init__(
        self,
        ws: web.WebSocketResponse,
        client_id: str,
        cfg: VoxConfig,
        memory: MemoryStore,
        vision: VisionStore,
        brain: VoxBrain,
        asr_fn: AsrCallable,
        tts_fn: TtsCallable,
        http: Any = None,
        photo_lib: PhotoLibrary | None = None,
        ring: RingBuffer | None = None,
        coordinator: DuplexCoordinator | None = None,
    ) -> None:
        self.http = http
        self.coordinator = coordinator
        self.photo_lib = photo_lib
        self.ring = ring
        self.ws = ws
        self.client_id = client_id
        self.cfg = cfg
        self.memory = memory
        self.vision = vision
        self.brain = brain
        self.asr_fn = asr_fn
        self.tts_fn = tts_fn
        self.vad = StreamingVad()
        self.epoch = self.memory.next_epoch()
        self.speaking = False
        self._turn_task: asyncio.Task[None] | None = None
        self._barge_frames = 0
        self.last_spoken_text = ""
        self._thinking = False
        self.last_user_ts = 0.0
        self.agency = None
        # 2026-11-22 三模式：LIVE / STANDBY / SLEEP_LISTEN（由客户端 MODE_CHANGE 控制帧驱动）
        self.mode = "LIVE"
        self._proactive_task: asyncio.Task[None] | None = None
        self.device_state: dict[str, Any] = {}
        self.guest_mode: bool = False
        self.guest_since_ts: float = 0.0
        self.owner_near_rms: float = float(os.environ.get("VOX_OWNER_NEAR_RMS", "800"))
        self.last_owner_near_ts: float = 0.0
        self._owner_soft_streak: int = 0
        self.last_gps: tuple[float, float, float] | None = None   # (lat_gcj, lon_gcj, acc)
        self.current_place: str = ""
        self.current_road: str = ""
        self._road_ts: float = 0.0
        self._place_checked_ts: float = 0.0
        self.power_profile: str = "AC"
        self.heartbeat_mult: float = 1.0
        self._last_dev_alert: float = 0.0
        # 双脑：轻附和不误打断（VOX_BACKCHANNEL_GUARD=1 才生效）
        self._deferred_barge: bool = False

    async def send_bytes(self, data: bytes) -> None:
        if not self.ws.closed:
            await self.ws.send_bytes(data)

    async def start(self) -> None:
        await self.send_bytes(
            pack_control(self.epoch, "SERVER_HELLO", client=self.client_id, version=__version__)
        )
        await self.send_bytes(pack_control(self.epoch, "SPEECH_STATE", state="IDLE"))

        # 2026-11-22 Phase1：主动心跳（受激素节律门控；空闲才动）
        try:
            # 硬锁 1：同一 client_id 全局只允许一个活跃实例（旧实例先停干净）
            self.agency = await AgencyLoopV2.start_singleton(
                client_id=self.client_id,
                system_prompt=self.brain.system_prompt,
                unified_tools=self.brain.tools_schema,
                tool_executor=self._agency_tool,
                hormones=self.brain.hormones,
                todo_goals=self.brain.todo_goals,
                llm_caller=self._agency_llm,
                speak_cb=self._proactive_speak,
                state_provider=self._agency_state,
                enabled=None,
            )
        except Exception as exc:
            log.warning("AGENCY start failed %s", exc)
        log.info("AGENCY_BOOT client=%s active=%d",
                 self.client_id, AgencyLoopV2.active_count(self.client_id))

        # 2026-11-22 双脑：静默窗口择机插话 watcher
        if self.coordinator is not None:
            self._start_bg_watcher()

    async def cancel_turn(self, reason: str = "barge_in") -> None:
        """2026-11-22：同时打断普通回合与主动播报，并复位 VAD/状态后立刻换 epoch。"""
        try:
            self.brain.hormones.on_barge_in()
        except Exception:
            pass
        for attr in ("_turn_task", "_proactive_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            setattr(self, attr, None)
        self.speaking = False
        self._thinking = False
        try:
            self.vad.reset()
        except Exception:
            pass
        self.epoch = self.memory.next_epoch()
        await self.send_bytes(pack_control(self.epoch, "BARGE_IN_CLEAR", reason=reason))
        await self.send_bytes(pack_control(self.epoch, "SPEECH_STATE", state="IDLE"))

    async def handle_text_turn(self, text: str, speaker: str = "", spk_sim: float = 0.0) -> None:
        self.last_user_ts = time.time()
        _ACTIVITY["last_user_speech"] = time.time()
        if self._turn_task and not self._turn_task.done():
            await self.cancel_turn("new_turn")
        self.epoch = self.memory.next_epoch()
        epoch = self.epoch
        self._turn_task = asyncio.create_task(
            self._run_turn(epoch, text, speaker=speaker, spk_sim=spk_sim)
        )

    def _update_gps(self, lat: float | None, lon: float | None,
                    acc: float | None = 0.0, force: bool = False) -> None:
        """收到有效坐标（acc<=60m）→ WGS84→GCJ02，落地围栏判定。"""
        try:
            if lat is None or lon is None:
                return
            # (0,0) 是客户端没有定位时的占位，不能当真
            if abs(float(lat)) < 0.1 and abs(float(lon)) < 0.1:
                return
            accv = float(acc or 0.0)
            if accv > 60.0:
                return
            from .geo import wgs84_to_gcj02
            glat, glon = wgs84_to_gcj02(float(lat), float(lon))
            self.last_gps = (glat, glon, accv)
            now = time.time()
            if force or (now - self._place_checked_ts) >= 30.0:
                self._place_checked_ts = now
                hit = self.memory.match_place(self.client_id, glat, glon)
                new_name = hit["name"] if hit else ""
                if new_name != self.current_place:
                    log.info("PLACE -> %r (dist=%s)", new_name, hit and hit["distance_m"])
                self.current_place = new_name
                # 精确路名：配了高德 Web 服务 Key 时后台补逆地理（60s 一次，失败静默）
                if (now - self._road_ts) >= 60.0:
                    self._road_ts = now
                    try:
                        asyncio.create_task(self._refresh_road(glat, glon))
                    except Exception:
                        pass
        except Exception as exc:
            log.debug("GPS update err %s", exc)

    async def _refresh_road(self, lat: float, lon: float) -> None:
        try:
            rg = await self.brain.amap.regeo(lat, lon)
            road = (rg or {}).get("road") or ""
            if road and road != self.current_road:
                self.current_road = road
                log.info("ROAD -> %r", road)
        except Exception as exc:
            log.debug("road refresh err %s", exc)

    def _set_guest(self, active: bool, reason: str) -> None:
        if self.guest_mode == active:
            return
        self.guest_mode = active
        if active:
            self.guest_since_ts = time.time()
            try:
                self.brain.social.update_situation("访客正在试戴眼镜")
            except Exception:
                pass
        log.info("GUEST_MODE -> %s (%s)", active, reason)
        try:
            asyncio.create_task(self.send_bytes(
                pack_control(self.epoch, "GUEST_MODE", active=active)
            ))
        except Exception:
            pass

    def _update_guest_state(self, speaker: str, spk_sim: float, pcm: bytes, text: str) -> None:
        # 手动口令优先（退出 > 进入）
        _intent = guest_intent(text)
        if _intent == "exit":
            self._set_guest(False, "voice_command_exit")
            return
        if _intent == "enter":
            self._set_guest(True, "voice_command")
            return
        try:
            raw_rms = pcm_rms(pcm)
        except Exception:
            raw_rms = 0.0
        if speaker == "owner" and spk_sim >= 0.50 and raw_rms >= self.owner_near_rms * GUEST_EXIT_RATIO:
            self._owner_soft_streak += 1
        else:
            self._owner_soft_streak = 0
        new_state, new_last = evaluate_guest_state(
            speaker=speaker, spk_sim=spk_sim, raw_rms=raw_rms,
            owner_near_rms=self.owner_near_rms, guest_mode=self.guest_mode,
            now=time.time(), last_owner_near_ts=self.last_owner_near_ts,
            owner_soft_ok=(self._owner_soft_streak >= 2),
            text_len=len(text or ""),
        )
        if not _AUTO_GUEST and new_state and not self.guest_mode:
            # 自动进入已关闭：声纹/音量不再能开局访客，只认「开启访客模式」口令
            log.info("AUTO_GUEST_OFF drop_enter speaker=%s sim=%.2f rms=%d",
                     speaker, spk_sim, raw_rms)
            new_state = False
        self.last_owner_near_ts = new_last
        if speaker == "owner" and spk_sim >= GUEST_EXIT_SIM and raw_rms > 0:
            self.owner_near_rms = min(6000.0, max(300.0,
                self.owner_near_rms * 0.7 + raw_rms * 0.3))
        self._set_guest(new_state, "auto_speaker=%s rms=%d base=%d" % (
            speaker, raw_rms, self.owner_near_rms))

    def _looks_like_echo(self, text: str) -> bool:
        """与最近一次播报的字二元组重合度 > 0.6 视为自己的回声。"""
        prev = (self.last_spoken_text or "").strip()
        cur = (text or "").strip()
        if len(prev) < 4 or len(cur) < 4:
            return False
        a = {prev[i:i + 2] for i in range(len(prev) - 1)}
        b = {cur[i:i + 2] for i in range(len(cur) - 1)}
        if not b:
            return False
        return len(a & b) / float(len(b)) > 0.6

    def _agency_state(self) -> dict:
        """心跳用的真实视野：最近视觉摘要 + 最近 4 条对话。"""
        try:
            caption = self.memory.latest_vision_caption(self.client_id)
        except Exception:
            caption = ""
        try:
            recent = self.memory.get_recent_turns(self.client_id, limit=4)
        except Exception:
            recent = []
        return {
            "busy": bool(
                self.speaking
                or self._thinking
                or _ACTIVITY["busy"] > 0
                or (time.time() - _ACTIVITY["last_user_speech"]) < 30
            ),
            "last_user_ts": self.last_user_ts,
            "mode": self.mode,
            "guest_mode": self.guest_mode,
            "power_profile": self.power_profile,
            "heartbeat_mult": self.heartbeat_mult,
            "latest_caption": caption,
            "recent_turns": recent,
        }

    async def _agency_tool(self, name: str, args: dict) -> dict:
        """心跳用：只读/状态工具，不触发导航启停这类副作用。"""
        if name == "get_weather":
            try:
                return await self.brain.amap.get_weather(self.client_id, city=str(args.get("city", "") or ""))
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
        if name in ("manage_todo", "manage_goal"):
            return self.brain.todo_goals.exec_tool(self.client_id, name, args)
        return {"ok": False, "reason": "tool_not_allowed_in_heartbeat"}

    async def _agency_llm(self, messages, tools):
        # 绝不与 /text 或语音回合争抢唯一的 FAEX1 槽位
        if _ACTIVITY["busy"] > 0 or self.speaking or self._thinking:
            raise RuntimeError("slot_busy")
        return await asyncio.wait_for(
            self.brain._call_llm(messages, tools=tools, timeout_s=20,
                                 reasoning_effort="medium"),
            timeout=25
        )

    def _maybe_device_alert(self) -> None:
        """未充电且电量 <=15%，或机身 >=43°C → 给一次 NE 脉冲（10 分钟内只触发一次）。"""
        ds = self.device_state or {}
        try:
            bat = int(ds.get("battery_pct", 100))
            chg = bool(ds.get("charging"))
            temp = float(ds.get("temp_c") or 0.0)
        except Exception:
            return
        if (bat <= 15 and not chg) or temp >= 43.0:
            now = time.time()
            if now - self._last_dev_alert > 600.0:
                self._last_dev_alert = now
                try:
                    self.brain.hormones.on_device_alert()
                except Exception:
                    pass
                log.info("DEVICE_ALERT bat=%s charging=%s temp=%s", bat, chg, temp)

    def _start_bg_watcher(self) -> None:
        if self.coordinator is None:
            return
        try:
            self.coordinator.start_watcher(
                self.client_id,
                is_busy_fn=lambda: bool(
                    self.speaking or self._thinking
                    or getattr(self.vad, "in_speech", False)
                ),
                last_activity_fn=lambda: float(self.last_user_ts),
                guest_mode_fn=lambda: bool(self.guest_mode),
                on_interject=self._on_bg_interject,
            )
            log.info("BG_WATCHER started client=%s", self.client_id)
        except Exception as exc:
            log.warning("BG_WATCHER start failed %s", exc)

    async def _on_bg_interject(self, task: BgTask) -> None:
        """后台结果落到眼镜：先推 HUD/NAV，再主动开口（走同一条 TTS 下行）。"""
        if self.ws.closed or self.mode != "LIVE":
            log.info("BG_INTERJECT_SKIP id=%s mode=%s closed=%s",
                     task.task_id, self.mode, self.ws.closed)
            return
        for patch in task.hud_patches:
            await self.send_bytes(
                pack_control(self.epoch, "HUD_PATCH", **{k: v for k, v in patch.items() if k != "type"})
            )
        for np in task.nav_patches:
            await self.send_bytes(
                pack_control(self.epoch, "NAV_PATCH", **{k: v for k, v in np.items() if k != "type"})
            )
        if task.result_text:
            log.info("BG_INTERJECT id=%s text=%r", task.task_id, task.result_text[:40])
            # 独立 task：不让 cancel_turn 顺着 _proactive_task 把 watcher 一起干掉
            await asyncio.create_task(self._proactive_speak(task.result_text))

    async def _proactive_speak(self, text: str, speed: float = 1.0) -> None:
        """主动开口：走同一条 TTS 下行，期间置 speaking 防回声自激。
        2026-11-22：登记 current_task，让 cancel_turn 能连主动播报一起打断。"""
        try:
            self._proactive_task = asyncio.current_task()
        except Exception:
            self._proactive_task = None
        text = strip_proactive_marker((text or "").strip())
        if not text:
            return
        # 统一闸门：所有主动播报（心跳 / 后台插话）都要过原文去重
        ok, why = proactive_allowed(text, due=True)
        if not ok:
            log.info("PROACTIVE_SKIP reason=%s text=%r", why, text[:40])
            return
        note_proactive_spoken(text)
        epoch = self.epoch
        self.speaking = True
        try:
            await self.send_bytes(
                pack_control(epoch, "TRANSCRIPT", role="assistant", text=text, final=True)
            )
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="SPEAKING"))
            _pb = 0
            async for pcm24k in self.tts_fn(text):
                if epoch != self.epoch:
                    break
                _pb += len(pcm24k)
                await self.send_bytes(pack_frame(FT_AUDIO_DOWN, epoch, pcm24k))
            log.info("PROACTIVE_SENT bytes=%d text=%r", _pb, text[:32])
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="IDLE"))
        except Exception as exc:
            log.warning("PROACTIVE_SPEAK err %s", exc)
        finally:
            self.speaking = False
            self._proactive_task = None

    async def _say_hint(self, text: str = "我翻一下当时的照片") -> None:
        """原图反刍前先出声垫一句，避免语音场景沉默十几秒。"""
        try:
            epoch = self.epoch
            self.speaking = True
            async for pcm24k in self.tts_fn(text):
                if epoch != self.epoch:
                    break
                await self.send_bytes(pack_frame(FT_AUDIO_DOWN, epoch, pcm24k))
        except Exception as exc:
            log.warning("HINT_TTS err %s", exc)
        finally:
            self.speaking = False


    async def _speak_sentences(self, epoch: int, sentences: list[str], t0: float | None = None) -> None:
        """逐句播报，下一句的 TTS 首包在本句播放时预取（VOX_TTS_PREFETCH=0 关闭）。"""
        prefetch = os.environ.get("VOX_TTS_PREFETCH", "0") != "0"
        _ttfb_logged = False

        async def _prime(sent: str):
            ag = self.tts_fn(sent)
            try:
                first = await ag.__anext__()
            except StopAsyncIteration:
                return None, None
            return ag, first

        _next = None
        for idx, sent in enumerate(sentences):
            if epoch != self.epoch:
                return
            if _next is not None:
                ag, first = _next
                _next = None
            else:
                ag, first = await _prime(sent)
            if ag is None:
                continue
            if prefetch and idx + 1 < len(sentences):
                _next = await _prime(sentences[idx + 1])
            if not _ttfb_logged and first is not None:
                _ttfb_logged = True
                log.info("TTS_TTFB_MS %.0f first_sent=%r bytes=%d",
                         (time.time() - (t0 or time.time())) * 1000.0, sent[:24], len(first))
            if first is not None:
                await self.send_bytes(pack_frame(FT_AUDIO_DOWN, epoch, first))
            async for pcm24k in ag:
                if epoch != self.epoch:
                    return
                await self.send_bytes(pack_frame(FT_AUDIO_DOWN, epoch, pcm24k))
    async def _run_turn(self, epoch: int, text: str, speaker: str = "", spk_sim: float = 0.0) -> None:
        _ACTIVITY["busy"] += 1
        try:
            await self.send_bytes(pack_control(epoch, "TRANSCRIPT", role="user", text=text, final=True))
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="THINKING"))

            self._thinking = True
            try:
                _social_block = self.brain.social.format_context_block()
            except Exception:
                _social_block = ""
            tts_queue: asyncio.Queue = asyncio.Queue()

            async def _tts_consumer() -> None:
                """唯一串行消费者：严格 FIFO，一句读完再读下一句（禁并发 TTS）。"""
                while True:
                    sent = await tts_queue.get()
                    if sent is None:
                        return
                    _bytes = 0
                    try:
                        async for pcm in self.tts_fn(sent):
                            if epoch != self.epoch:
                                return
                            _bytes += len(pcm)
                            await self.send_bytes(pack_frame(FT_AUDIO_DOWN, epoch, pcm))
                        log.info("TTS_SENT sent=%r bytes=%d", sent[:24], _bytes)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.warning("TTS_SENT err %s", exc)

            consumer = asyncio.create_task(_tts_consumer())

            async def _on_sentence(sent: str) -> None:
                await tts_queue.put(sent)

            try:
                res = await self.brain.run_turn(
                    self.client_id, text, epoch_id=epoch, on_slow_tool=self._say_hint,
                    speaker=speaker, spk_sim=spk_sim, social_block=_social_block,
                    guest_mode=self.guest_mode, guest_since_ts=self.guest_since_ts,
                    first_sentence_cb=_on_sentence,
                )
            except asyncio.CancelledError:
                consumer.cancel()
                raise
            self._thinking = False
            if epoch != self.epoch:
                return

            for patch in res.hud_patches:
                await self.send_bytes(pack_control(epoch, "HUD_PATCH", **{k: v for k, v in patch.items() if k != "type"}))

            if not (res.reply_text or "").strip():
                log.info("SILENT_TURN speaker=%r", speaker)
                await tts_queue.put(None)
                try:
                    await consumer
                except Exception:
                    pass
                await self.send_bytes(pack_control(epoch, "TURN_END"))
                await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="IDLE"))
                return

            self.last_spoken_text = res.reply_text or ""
            _t_transcript = time.time()
            await self.send_bytes(
                pack_control(epoch, "TRANSCRIPT", role="assistant", text=res.reply_text, final=True)
            )
            self.speaking = True
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="SPEAKING"))

            # 流式模式下所有句子已由 on_sentence 入队；这里只补尾部残余，绝不重播全文
            _spoken = (getattr(res, "pre_spoken", "") or "").strip()
            _reply = (res.reply_text or "")
            _rest = ""

            def _nows(s: str) -> str:
                return "".join(ch for ch in s if not ch.isspace())

            if _spoken:
                _cs, _cr = _nows(_spoken), _nows(_reply)
                if _cr.startswith(_cs):
                    # 在原文里定位"已播完"的位置，跳过空白差异
                    _cnt = 0
                    _pos = len(_reply)
                    for _i, _ch in enumerate(_reply):
                        if not _ch.isspace():
                            _cnt += 1
                        if _cnt >= len(_cs):
                            _pos = _i + 1
                            break
                    _rest = _reply[_pos:]
                    _rest = _rest.lstrip("，,。.、；;：: \n\t")
                else:
                    log.warning("SPOKEN_MISMATCH spoken=%r reply=%r", _cs[:40], _cr[:40])
                    _rest = ""
            else:
                _rest = _reply

            log.info("SPOKEN_CHECK spoken_len=%d rest_len=%d", len(_spoken), len(_rest.strip()))
            if _rest.strip():
                for _s in split_sentences(_rest):
                    await tts_queue.put(_s)

            await tts_queue.put(None)
            await consumer
            self.speaking = False
            await self.send_bytes(pack_control(epoch, "TURN_END"))
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="IDLE"))
        except asyncio.CancelledError:
            self.speaking = False
            raise
        except Exception as exc:
            log.exception("Turn failed: %s", exc)
            await self.send_bytes(pack_control(epoch, "ERROR", message=str(exc)))
            await self.send_bytes(pack_control(epoch, "SPEECH_STATE", state="IDLE"))
        finally:
            # 2026-11-22：取消（打断/新轮/断线）时也必须复位，
            # 否则 _thinking/busy 永久卡住，心跳从此再也不评估。
            self._thinking = False
            self.speaking = False
            _ACTIVITY["busy"] = max(0, _ACTIVITY["busy"] - 1)

    async def on_binary(self, raw: bytes) -> None:
        ftype, fepoch, payload = unpack_frame(raw)
        # Epoch iron rule: accept slightly stale uplink 0x01 smoothly; never disconnect on WiFi reorder
        if fepoch > self.epoch + 1000:
            await self.ws.close(code=1002, message=b"epoch ahead")
            return

        if ftype == FT_AUDIO_UP:
            if self.speaking and self.cfg.half_duplex_guard:
                return  # Optional half-duplex guard while hardware AEC is tuned
            ev, pcm = self.vad.feed(payload)
            if ev == "onset":
                if self.speaking:
                    # 2026-11-22 轻附和不误打断：先不砍，等 ASR 出来再判是不是「嗯/哈哈」
                    if _BACKCHANNEL_GUARD:
                        self._deferred_barge = True
                        log.info("BARGE_DEFER 等 ASR 判附和词")
                    else:
                        await self.cancel_turn("user_speech")
                await self.send_bytes(pack_control(self.epoch, "SPEECH_STATE", state="LISTENING"))
            elif ev == "utterance" and pcm:
                text = ""
                speaker, spk_sim = "", 0.0
                _t_asr = time.time()
                try:
                    # ASR 与声纹并行，消除串行阻塞
                    _asr_t = asyncio.create_task(self.asr_fn(pcm))
                    _spk_t = asyncio.create_task(
                        asyncio.to_thread(self.brain.speaker_id.identify_pcm, pcm))
                    text_new, _spk = await asyncio.gather(_asr_t, _spk_t)
                    text = (text_new or "").strip()
                    speaker, spk_sim = _spk
                    log.info("ASR_SPK %.0fms speaker=%s sim=%.2f text=%r",
                             (time.time() - _t_asr) * 1000, speaker, spk_sim, text[:24])
                    # 纯文本 /text 探针没有音频、speaker 为空：不参与多人静默判定
                    if speaker:
                        self.brain.social.note_utterance(speaker, text)
                except Exception as exc:
                    log.warning("ASR_SPK err %s", exc)
                if text and self._looks_like_echo(text):
                    log.info("ASR_ECHO_DROP %r", text[:24])
                    text = ""
                if self._deferred_barge:
                    self._deferred_barge = False
                    if text and is_backchannel(text):
                        log.info("BACKCHANNEL_IGNORE %r", text[:16])
                        if self.speaking:
                            # 附和词不打断，让当前回复顺滑播完
                            return
                        text = ""
                if text:
                    _ACTIVITY["last_user_speech"] = time.time()
                if text and self.mode == "SLEEP_LISTEN" and not _has_wake_word(text):
                    log.info("AMBIENT_DROP %r", text[:32])
                    try:
                        self.memory.record_turn(self.client_id, "ambient", text, epoch_id=self.epoch)
                    except Exception as exc:
                        log.warning("AMBIENT_RECORD err %s", exc)
                    await self.send_bytes(pack_control(self.epoch, "SPEECH_STATE", state="IDLE"))
                    return
                if text:
                    self._update_guest_state(speaker, spk_sim, pcm, text)
                    await self.handle_text_turn(text, speaker=speaker, spk_sim=spk_sim)
                else:
                    await self.send_bytes(pack_control(self.epoch, "SPEECH_STATE", state="IDLE"))
        elif ftype == FT_VISION_UP:
            meta, img_bytes = unpack_vision_payload(payload)
            if isinstance(meta, dict):
                self._update_gps(meta.get("lat"), meta.get("lon"), meta.get("acc"))
            if self.ring is not None:
                try:
                    self.ring.add(self.client_id, img_bytes, meta)
                except Exception as exc:
                    log.warning("RING_ADD err %s", exc)
            rec = self.vision.ingest(self.client_id, img_bytes, meta)
            self.memory.save_vision_frame(
                client_id=self.client_id,
                frame_path=rec.path,
                ts=rec.ts,
                lat=meta.get("lat"),
                lon=meta.get("lon"),
                acc=meta.get("acc"),
            )
            # 2026-11-22: 自维护图库接管 —— 关键帧去重落盘到 voice-frames/，
            # 由 photo_lib 的后台打标写回 vox_unified.db。不再转发给 :8768。
            if self.photo_lib is not None and not self.guest_mode:
                try:
                    self.photo_lib.maybe_archive_frame(
                        self.client_id, img_bytes, meta, ts=rec.ts
                    )
                except Exception as exc:
                    log.warning("PHOTO_ARCHIVE err %s", exc)

        elif ftype == FT_CONTROL:
            obj = json.loads(payload.decode("utf-8"))
            mtype = str(obj.get("type", "")).upper()
            if mtype == "CLIENT_HELLO":
                if obj.get("client"):
                    self.client_id = str(obj["client"])
            elif mtype == "CLIENT_INTERRUPT":
                await self.cancel_turn(str(obj.get("reason", "client_interrupt")))
            elif mtype == "TEXT_INPUT":
                txt = str(obj.get("text", "")).strip()
                # 退出必须先于进入判定（"关闭访客模式"里也含"访客模式"）
                _gi = guest_intent(txt)
                if _gi == "exit":
                    self._set_guest(False, "text_command_exit")
                elif _gi == "enter":
                    self._set_guest(True, "text_command")
                if txt and self.mode == "SLEEP_LISTEN" and not _has_wake_word(txt):
                    # 防社死（文本同源）：静默入库，不外放
                    log.info("AMBIENT_DROP text=%r", txt[:32])
                    try:
                        self.memory.record_turn(self.client_id, "ambient", txt, epoch_id=self.epoch)
                    except Exception as exc:
                        log.warning("AMBIENT_RECORD err %s", exc)
                elif txt:
                    await self.handle_text_turn(txt)
            elif mtype == "DEVICE_TELEMETRY":
                ds = {k: v for k, v in obj.items() if k not in ("type", "epoch_id")}
                self.device_state = ds
                self._update_gps(ds.get("lat"), ds.get("lon"), ds.get("acc"))
                log.debug("DEVICE_TELEMETRY %s", ds)
                self._maybe_device_alert()
            elif mtype == "PARKING_PIN":
                try:
                    self._update_gps(obj.get("lat"), obj.get("lon"), obj.get("acc"), force=True)
                    if self.last_gps:
                        _r = self.memory.save_place(self.client_id, "上次停车位",
                                                    self.last_gps[0], self.last_gps[1], 80.0)
                        log.info("PARKING_PIN saved place_id=%s", _r.get("place_id"))
                except Exception as exc:
                    log.warning("PARKING_PIN err %s", exc)
            elif mtype == "MODE_CHANGE":
                _m = str(obj.get("mode", "LIVE")).upper()
                if _m in ("LIVE", "STANDBY", "SLEEP_LISTEN", "SLEEP"):
                    self.mode = _m
                if obj.get("power_profile"):
                    self.power_profile = str(obj["power_profile"]).upper()
                if obj.get("heartbeat_mult") is not None:
                    try:
                        self.heartbeat_mult = float(obj["heartbeat_mult"])
                    except Exception:
                        pass
                log.info("MODE_CHANGE -> %s power=%s hb=%.2f",
                         self.mode, self.power_profile, self.heartbeat_mult)
                # 真休眠不插话：停 watcher；回 LIVE 再挂上
                if self.coordinator is not None:
                    if self.mode == "SLEEP":
                        self.coordinator.stop_watcher(self.client_id)
                        log.info("BG_WATCHER stopped (SLEEP) client=%s", self.client_id)
                    elif self.mode == "LIVE":
                        self._start_bg_watcher()
            elif mtype == "PING":
                await self.send_bytes(pack_control(self.epoch, "PONG", t_ms=obj.get("t_ms", 0)))




def build_tts_fn(cfg: VoxConfig, http: Any):
    """按 VOX_TTS_BACKEND 选 TTS 适配器：cosyvoice（本地）/ openai（任意兼容服务）。"""
    backend = (getattr(cfg, "tts_backend", "cosyvoice") or "cosyvoice").strip().lower()
    if backend == "openai":
        base = getattr(cfg, "tts_openai_url", "") or ""
        if not base:
            raise RuntimeError("VOX_TTS_BACKEND=openai 时必须设置 VOX_TTS_OPENAI_URL")
        _key = getattr(cfg, "tts_openai_key", "") or ""
        _model = getattr(cfg, "tts_openai_model", "tts-1") or "tts-1"
        _voice = getattr(cfg, "tts_openai_voice", "alloy") or "alloy"
        log.info("TTS_BACKEND openai url=%s model=%s voice=%s", base, _model, _voice)
        return lambda s: openai_tts_stream(http, base, _key, _model, _voice, s)
    engine = (getattr(cfg, "tts_engine", "") or "").lower()
    if engine in ("gsv", "gpt-sovits", "gpt_sovits") and getattr(cfg, "gsv_url", ""):
        log.info("TTS_BACKEND gsv ref=%s", cfg.gsv_ref_wav)
        return lambda s: gpt_sovits_tts_stream(http, cfg.gsv_url, cfg.gsv_ref_wav, cfg.gsv_prompt_text, s)
    log.info("TTS_BACKEND cosyvoice url=%s spk=%s", cfg.tts_url, cfg.tts_spk_id)
    return lambda s: cosyvoice2_tts_stream(http, cfg.tts_url, cfg.tts_spk_id, s)


def create_app(
    cfg: VoxConfig | None = None,
    asr_override: AsrCallable | None = None,
    tts_override: TtsCallable | None = None,
) -> web.Application:
    cfg = cfg or VoxConfig.from_env()
    memory = MemoryStore(cfg.db_path)
    vision = VisionStore(cfg.frame_dir, ttl_s=cfg.live_view_ttl_s)
    app = web.Application()
    active_sessions: set[DuplexSession] = set()

    # 2026-11-22 MOSS-Live 双脑总开关：默认关闭，VOX_DUPLEX_DUAL_BRAIN=1 才启用
    coordinator: DuplexCoordinator | None = None
    if os.environ.get("VOX_DUPLEX_DUAL_BRAIN", "0").strip().lower() in ("1", "true", "yes", "on"):
        coordinator = DuplexCoordinator()
        log.info("DUPLEX_DUAL_BRAIN enabled silence_gate=%.2fs poll=%.2fs",
                 coordinator.silence_gate_s, coordinator.poll_interval_s)
    app["coordinator"] = coordinator
    log.info("GUEST_AUTO %s", "on (声纹/音量自动切)" if _AUTO_GUEST else "off (仅手动口令进入/退出 + 机主声纹解锁)")

    def _device_state() -> dict:
        for _s in list(active_sessions):
            return dict(getattr(_s, "device_state", {}) or {})
        return {}

    def _device_cmd(action: str, value: str = "") -> dict:
        sent = 0
        for _s in list(active_sessions):
            try:
                asyncio.create_task(
                    _s.send_bytes(pack_control(_s.epoch, "DEVICE_CMD", action=action, value=value))
                )
                sent += 1
            except Exception as exc:
                log.warning("DEVICE_CMD send err %s", exc)
        return {"ok": sent > 0, "sent": sent, "action": action, "value": value}
    # 打标只在没人说话时跑（FAEX1 是同一块卡，抢了会拖慢回复）
    photo_lib = PhotoLibrary(
        db_path=cfg.db_path,
        vision_url=cfg.llm_url,
        vision_model=os.environ.get("VOX_VISION_MODEL", "qwen3.8-flash-next"),
        is_busy_fn=lambda: any(getattr(s, "speaking", False) for s in active_sessions),
    )
    app["photo_lib"] = photo_lib
    ring = RingBuffer()
    app["ring"] = ring
    app["activity"] = {"busy": 0, "last_user_speech": 0.0}
    photo_lib.ring = ring if photo_lib is not None else None

    async def on_startup(a: web.Application) -> None:
        import aiohttp

        a["http"] = aiohttp.ClientSession()
        a["brain"] = VoxBrain(
            cfg, memory, vision, http_session=a["http"], ring=a["ring"],
            coordinator=coordinator,
        )
        # 2026-11-22 本机遥测/控制接线
        a["brain"].device_state_fn = _device_state
        a["brain"].device_cmd_fn = _device_cmd

        def _location() -> dict:
            for _s in list(active_sessions):
                g = getattr(_s, "last_gps", None)
                if g:
                    return {"lat": g[0], "lon": g[1], "acc": g[2],
                            "place": getattr(_s, "current_place", ""),
                            "road": getattr(_s, "current_road", "")}
            return {}

        a["brain"].location_fn = _location



    async def on_shutdown(a: web.Application) -> None:
        """Close live WebSockets and stop agency/watchers so systemd stop is instant.

        aiohttp otherwise waits for every handler to finish; the glasses never hang up
        on their own, so each restart used to sit there until TimeoutStopSec + SIGKILL.
        """
        sessions = list(active_sessions)
        log.info("SHUTDOWN begin: %d session(s)", len(sessions))
        for s in sessions:
            try:
                await AgencyLoopV2.stop_client(s.client_id)
            except Exception:
                pass
            if coordinator is not None:
                try:
                    coordinator.stop_watcher(s.client_id)
                except Exception:
                    pass
            for attr in ("_turn_task", "_proactive_task"):
                t = getattr(s, attr, None)
                if t is not None and not t.done():
                    t.cancel()
            try:
                s.speaking = False
                s._thinking = False
            except Exception:
                pass
        for s in sessions:
            try:
                await asyncio.wait_for(
                    s.ws.close(code=1001, message=b"server_shutdown"), timeout=1.0
                )
            except Exception:
                pass
        if coordinator is not None:
            try:
                await coordinator.shutdown()
            except Exception:
                pass
        log.info("SHUTDOWN done")

    app.on_shutdown.append(on_shutdown)

    async def on_cleanup(a: web.Application) -> None:
        if coordinator is not None:
            try:
                await coordinator.shutdown()
            except Exception:
                pass
        await a["http"].close()
        memory.close()

    async def _warm_tts(a_: web.Application) -> None:
        """首音预热：服务启动先合成一句丢掉，避免第一个回合吃 5s 冷启动。"""
        async def _boot() -> None:
            # aiohttp 的 startup 信号执行顺序不保证，这里等主 startup 把 http 会话建好
            for _ in range(50):
                if a_.get("http") is not None:
                    break
                await asyncio.sleep(0.1)
            http = a_.get("http")
            if http is None:
                log.warning("TTS_WARMUP skipped: http session not ready")
                return
            try:
                _fn = build_tts_fn(cfg, http)
                async for _pcm in _fn("你好"):
                    break
                log.info("TTS_WARMUP done")
            except Exception as exc:
                log.warning("TTS_WARMUP err %s", exc)

        asyncio.create_task(_boot())

    app.on_startup.append(_warm_tts)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def health(req: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "version": __version__,
                "db": cfg.db_path,
                "llm_url": cfg.llm_url,
                "tts_url": cfg.tts_url,
                "active_ws": len(active_sessions),
                "half_duplex_guard": cfg.half_duplex_guard,
                # busy = 真的有回合在跑（打断后立刻回落）；
                # recent_speech = 30s 内说过话（打标器让路看这个）
                "busy": bool(_ACTIVITY["busy"] > 0) or any(
                    getattr(x, "speaking", False) for x in active_sessions
                ),
                "recent_speech": (time.time() - _ACTIVITY["last_user_speech"]) < 30,
                "mode": (next(iter(active_sessions)).mode if active_sessions else "LIVE"),
                "power_profile": (next(iter(active_sessions)).power_profile if active_sessions else "AC"),
                "guest_mode": (next(iter(active_sessions)).guest_mode if active_sessions else False),
                "device": (next(iter(active_sessions)).device_state if active_sessions else {}),
                "last_user_speech": _ACTIVITY["last_user_speech"],
            }
        )

    async def frame_last(req: web.Request) -> web.Response:
        client = req.query.get("client", cfg.default_client)
        res = vision.get_latest_bytes(client)
        if not res:
            return web.Response(status=404, text="no frame")
        data, mime = res
        return web.Response(body=data, content_type=mime)

    async def text_turn(req: web.Request) -> web.Response:
        body = await req.json()
        client = str(body.get("client", cfg.default_client))
        text = str(body.get("text", "")).strip()
        if not text:
            return web.json_response({"error": "empty text"}, status=400)
        epoch = memory.next_epoch()
        res = await app["brain"].run_turn(client, text, epoch_id=epoch)
        return web.json_response(
            {
                "epoch_id": epoch,
                "text": res.reply_text,
                "hud_patches": res.hud_patches,
                "tools_called": res.tools_called,
            }
        )

    async def duplex_ws(req: web.Request) -> web.WebSocketResponse:
        if cfg.auth_token and req.headers.get("X-Vox-Token") != cfg.auth_token:
            return web.Response(status=401, text="unauthorized")
        client = req.query.get("client", cfg.default_client)
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(req)

        asr_fn = asr_override or (lambda pcm: default_http_asr(app["http"], cfg.asr_url, pcm))
        if tts_override:
            tts_fn = tts_override
        else:
            tts_fn = build_tts_fn(cfg, app["http"])
        sess = DuplexSession(
            ws, client, cfg, memory, vision, app["brain"], asr_fn, tts_fn,
            http=app["http"], photo_lib=app["photo_lib"], ring=app["ring"],
            coordinator=coordinator,
        )
        active_sessions.add(sess)
        await sess.start()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await sess.on_binary(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    log.warning("WS error: %s", ws.exception())
        finally:
            try:
                # 硬锁 1：连注册表一起停干净（重复 stop 无害）
                await AgencyLoopV2.stop_client(sess.client_id)
                if sess.agency is not None:
                    await sess.agency.stop()
            except Exception:
                pass
            await sess.cancel_turn("ws_closed")
            if coordinator is not None:
                try:
                    await coordinator.shutdown(sess.client_id)
                except Exception:
                    pass
            active_sessions.discard(sess)
        return ws

    app.router.add_get("/health", health)
    app.router.add_get("/frame/last", frame_last)
    app.router.add_post("/text", text_turn)
    app.router.add_get("/ws/v1/duplex", duplex_ws)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = VoxConfig.from_env()
    web.run_app(create_app(cfg), host=cfg.host, port=cfg.port, shutdown_timeout=3.0)


if __name__ == "__main__":
    main()
