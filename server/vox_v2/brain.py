"""Zero-regex-router LLM brain: 100% of utterances go straight to FAEX1 with native Tool Calling."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time

import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable

from .amap_tools import AMAP_TOOLS_SCHEMA, AmapNavEngine
from .speaker_id import SocialTracker, SpeakerIdentifier
from .tool_forge import TOOLFORGE_TOOLS_SCHEMA, ToolForge
from .hormones_v2 import HormonesV2
from .todo_goals import TODO_GOAL_TOOLS_SCHEMA, TodoGoalStore
from .evidence_hud import build_evidence_hud_patch, merge_turn_hud_patches
from .duplex_coordinator import BgTask, DuplexCoordinator, is_backchannel  # noqa: F401
from .agency_loop_v2 import strip_proactive_marker
from .config import VoxConfig
from .memory import MemoryStore
from .vision import VisionStore

log = logging.getLogger("vox_v2.brain")

TZ_SHANGHAI = timezone(timedelta(hours=8))

# 长期归档帧的根目录（memory_units.best_frame_path 指向这里）。
# inspect_past_frames 只允许读这两个根下的文件，防止模型顺手读到别处。
FRAME_ARCHIVE_DIR = os.environ.get(
    "VOX_FRAME_ARCHIVE_DIR",
    os.path.expanduser("~/.local/share/open-vox-x2/frames"),
)
WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 对话回合里不允许出声的"通道标记/舞台说明"
_MARK_HEAD_RE = re.compile(r"^\s*[\[【]\s*(?:主动|AGENCY|心跳|后台)")
_META_ONLY_RE = re.compile(r"^\s*[（(][^）)]{0,60}[）)]\s*$")

GUEST_TOOL_ALLOW = {"describe_live_view", "get_weather", "do_nothing", "inspect_recent_frames"}
GUEST_CTRL_ALLOW = {"set_brightness", "set_volume"}
GUEST_BLOCK_MSG = "当前为访客模式，已保护机主隐私与设备权限"


DEFAULT_PERSONA = """你是 Vox，运行在用户 RayNeo X2 智能眼镜上的实时语音助手。
回答铁律：
1. 用自然、简短、口语化的中文直接回答（适合语音播报，不要用 Markdown 符号、列表星号或冗长客套）。
2. 当前精确北京时间已在头部 [CURRENT_TIME] 给出，问时间/日期/星期直接回答，绝不瞎猜。
3. [LIVE_VIEW] 仅代表眼镜此刻画面；如果标注“已过期”或“无画面”，严禁把它当成此刻眼前画面。
4. 涉及过去某天发生的事、吃过什么、做过什么，必须先调用 search_memory 工具查证；若工具返回 found=false，必须直接坦白“没找到那段时间的记录”，严禁编造。
5. 涉及看眼前物体/屏幕细节且头部摘要不足时，调用 describe_live_view 工具。
6. 涉及设提醒、记下个人长期事实、或控制智能家居时，直接调用对应工具，不要误当成历史查询。
7. 如果 search_memory 返回了 frame_paths（当时的原图），而文字记录不足以说清视觉细节（长相、颜色、屏幕上的字、桌上摆了什么东西），必须再调用 inspect_past_frames 把原图调出来看清楚再回答，不要凭摘要猜。

8. 回答「几点到 / 几点出发 / 几点下车 / 当天行程」这类问题时，不要只用地点关键词搜聊天记录：必须先用 browse_vision_timeline 正序看视觉转场时间线（或 search_memory 带 order=asc、kind=vision），必要时调 inspect_past_frames 看转场原图确认。若 total_matched 大于 returned、或 earliest_time 比你看到的第一条更早，严禁声称「没有更早的记录」。
9. 用户问「某段时间/一路上/刚才都聊了什么、说了什么、听到什么」时，必须调用 search_memory 且 kind="dialog"、query 留空，只填时间窗（start_iso/end_iso 或 seconds_ago_start/seconds_ago_end）；严禁把「对话」「聊天」「听到」当关键词塞进 query，也严禁不调用工具就凭印象作答。
"""


def _load_persona() -> str:
    """人设可外置：VOX_PERSONA_FILE 指向的纯文本优先，否则用内置通用人设。"""
    path = os.environ.get("VOX_PERSONA_FILE", "").strip()
    if path:
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                return txt
        except OSError as exc:
            log.warning("PERSONA_FILE %s 读取失败：%s", path, exc)
    return DEFAULT_PERSONA


SYSTEM_PROMPT = _load_persona()

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "检索用户历史记忆（包括视觉情景、过往对话、长期事实）。用于回答关于过去某天/某件事的回忆问题。"
                "【聊了什么】用户问「某时段/一路上/刚才聊了什么、说了什么、听到什么」时，"
                "必须调用本工具并设 kind='dialog'、query 留空，只给时间窗（start_iso/end_iso 或 seconds_ago_*）"
                "按时间拉对话；严禁把『对话』『聊天』『听到』当关键词放进 query 做 LIKE 过滤（会一条都搜不到）。"
                "长时段查对话时把 limit 设到 48–64（kind='dialog' 会用满 limit），一次拉全，别反复分页。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "可选关键词（如 麦当劳、猫、快递），留空则按时间窗检索"},
                    "start_iso": {"type": "string", "description": "可选起始北京时间，格式 YYYY-MM-DD HH:MM"},
                    "end_iso": {"type": "string", "description": "可选结束北京时间，格式 YYYY-MM-DD HH:MM"},
                    "order": {"type": "string", "enum": ["desc", "asc"], "description": "desc=最新在前（默认）；问'最早/几点到/几点出发'必须用 asc"},
                    "kind": {"type": "string", "enum": ["all", "vision", "dialog"], "description": "all=视觉+对话；vision=只看画面转场；dialog=只看聊天"},
                    "near_here": {"type": "boolean", "description": "只看当前 GPS 150 米范围内的记录（问这附近来过没时用）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_past_frames",
            "description": "翻看历史原图：把 search_memory 返回的 frame_paths（最多 3 张）交给视觉模型细看，回答关于过去某个时刻画面细节的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "原图绝对路径，来自 search_memory 返回的 frame_paths",
                    },
                    "question": {
                        "type": "string",
                        "description": "要从这些图里看清的具体问题",
                    },
                },
                "required": ["frame_paths", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_live_view",
            "description": "直接查看眼镜当前最新实时画面来回答用户关于眼前物体、文字或屏幕的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "需要从当前画面中看清的具体问题"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_fact",
            "description": "当用户明确要求记住某条长期事实或偏好时，将其写入长期记忆库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicate": {"type": "string", "description": "事实类别，例如 fact:pet、pref:coffee、profile:home"},
                    "value": {"type": "string", "description": "具体内容"},
                },
                "required": ["predicate", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_place",
            "description": "用当前 GPS 记住一个常去地点（如 公司、健身房）。只存人话名字，不暴露坐标给模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "地点名，如 公司 / 家 / 游戏房"},
                    "radius_m": {"type": "number", "description": "围栏半径，默认 100 米"},
                },
                "required": ["name"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "where_am_i",
            "description": "回答「我在哪 / 离公司多远 / 车停哪了」。返回最近已知地点与关键地点距离，不含原始经纬度。",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    {
        "type": "function",
        "function": {
            "name": "list_ha_entities",
            "description": "查询家里 Home Assistant 的真实设备状态（只读）。问『客厅灯开着吗 / 屋里几度 / 家里有哪些灯』时先用它，别猜 entity_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "模糊匹配名称或 entity_id，如 客厅、台灯、温度"},
                    "domain": {"type": "string", "description": "限定域：light/switch/climate/cover/fan/scene/script/sensor 等，可留空"},
                    "only_on": {"type": "boolean", "description": "只看处于开启状态(on/open/heating/cooling 等)的实体"},
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "call_home_assistant",
            "description": "控制家里已接入 Home Assistant 的设备（开关灯、调亮度、空调温度等）。控制前建议先 list_ha_entities 确认 entity_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "如 light, switch, climate"},
                    "service": {"type": "string", "description": "如 turn_on, turn_off, toggle, set_temperature"},
                    "entity_id": {"type": "string", "description": "HA 实体 ID"},
                    "data": {"type": "object", "description": "附加参数，如 {\"brightness_pct\":30} {\"color_temp_kelvin\":3000} {\"temperature\":26}"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "browse_vision_timeline",
            "description": "按时间正序浏览某段时间的视觉转场（只看画面，不含聊天记录）。回答几点到某地/几点出发/当天行程，必须先用它看转场时间线。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "起始北京时间 YYYY-MM-DD HH:MM"},
                    "end_iso": {"type": "string", "description": "结束北京时间 YYYY-MM-DD HH:MM"},
                    "max_points": {"type": "integer", "description": "最多返回多少个时间点，默认 24"},
                },
                "required": ["start_iso", "end_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_recent_frames",
            "description": "回到刚刚（默认 15 秒内）看实时环形池里的原图：路过的人长什么样、闪过去的牌子写了什么、刚才那辆车。不是历史归档。",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds_ago_start": {"type": "number", "description": "往前看多少秒开始，默认 15"},
                    "seconds_ago_end": {"type": "number", "description": "窗口结束（秒前），默认 0"},
                    "limit": {
                        "type": "integer",
                        "description": "最多返回几条（≤64，默认 16）；kind='dialog' 时间窗查对话请用 48–64",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "分页偏移（默认 0）；上一页满了要往下翻就填已返回条数",
                    },
                    "count": {"type": "integer", "description": "最多几张（≤3），默认 3"},
                },
            },
        },
    },

]
TOOLS_SCHEMA = list(TOOLS_SCHEMA) + list(TOOLFORGE_TOOLS_SCHEMA)
AGENCY_DECISION_TOOLS: list[dict[str, Any]] = [
    {
            "type": "function",
            "function": {
                "name": "control_glasses",
                "description": "控制用户戴着的这副雷鸟 X2 眼镜本机能力：亮度、音量、三模式、麦克风路由、立刻拍照归档、打开其他 App。值都用字符串。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set_brightness", "brightness_up", "brightness_down",
                                     "set_volume", "volume_up", "volume_down", "mute",
                                     "set_mode", "set_mic_mode", "take_photo", "open_app"],
                            "description": "要执行的设备动作",
                        },
                        "value": {
                            "type": "string",
                            "description": "set_brightness=0-100；set_volume=0-100；set_mode=LIVE/STANDBY/SLEEP_LISTEN；set_mic_mode=camcorder/voiceassistant/translation；open_app=包名或应用名；其余可留空",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "description": "决定这一轮不插嘴。用户在和别人聊天、或当下没有需要你出声的内容时用它；可以用 situation_note 记下现场正在聊什么。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "situation_note": {"type": "string", "description": "一句话概括现场局势/别人正在聊什么（会写进社交记忆）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proactive_speak",
            "description": "后台心跳决定主动开口（1-2 句口语，不提系统标签或数值）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要播报的口语内容"},
                    "reason": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
]
TOOLS_SCHEMA = TOOLS_SCHEMA + TODO_GOAL_TOOLS_SCHEMA
TOOLS_SCHEMA = TOOLS_SCHEMA + AGENCY_DECISION_TOOLS
TOOLS_SCHEMA = TOOLS_SCHEMA + AMAP_TOOLS_SCHEMA

# 2026-11-22 MOSS-Live 双脑：前台只负责秒答 + 派单，重活交给后台工兵。
# 只在 coordinator 存在时挂到「前台回合」工具表上，心跳/工兵自身不暴露，防递归派发。
DUPLEX_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dispatch_bg_task",
            "description": (
                "将耗时/多步操作异步派发给后台工兵（如米家/Home Assistant 控制、"
                "翻查历史记忆或原图、查天气/路线导航、ToolForge 造工具），"
                "调用后立刻口头简短答应并释放对话，不必原地干等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "给后台工兵的清晰任务目标"},
                    "need_vision": {"type": "boolean", "description": "是否需要看当前或历史眼镜画面"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_bg_task",
            "description": "当用户改口（如“算了别关了/不用查了”）时，取消在途或未播报的后台任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "可选任务 ID；留空则取消该用户全部在途任务"},
                },
            },
        },
    },
]


def _parse_iso_bj(val: str | None) -> float | None:
    if not val:
        return None
    s = val.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=TZ_SHANGHAI)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _strip_images(messages: list[dict[str, Any]]) -> bool:
    """把多模态 user 消息降级成纯文字；返回是否真的摘掉了图。"""
    hit = False
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
        if any(isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            hit = True
            m["content"] = "\n".join(t for t in texts if t) or "（图片已移除）"
    return hit


def _build_time_line() -> str:
    now = datetime.now(tz=TZ_SHANGHAI)
    return "[CURRENT_TIME | %s %s (北京时间)]" % (now.strftime("%Y-%m-%d %H:%M:%S"),
                                                  WEEKDAYS_ZH[now.weekday()])


def _parse_hhmm_list(text: str) -> list[int]:
    """从回复里抠出「10点30」「10:30」「16点51」等时刻，返回当天分钟数。"""
    out: list[int] = []
    for m in re.finditer(r"(\d{1,2})\s*[:：]\s*(\d{2})", text or ""):
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h < 24 and 0 <= mi < 60:
            out.append(h * 60 + mi)
    for m in re.finditer(r"(\d{1,2})\s*点\s*(\d{1,2})?", text or ""):
        h = int(m.group(1))
        mi = int(m.group(2)) if m.group(2) else 0
        if 0 <= h < 24 and 0 <= mi < 60:
            out.append(h * 60 + mi)
    return out


def _label_hhmm(label: str) -> int | None:
    m = re.search(r"(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", label or "")
    if not m:
        m2 = re.search(r"(\d{1,2}):(\d{2})", label or "")
        if not m2:
            return None
        return int(m2.group(1)) * 60 + int(m2.group(2))
    return int(m.group(3)) * 60 + int(m.group(4))


def _pick_frames_by_reply_time(reply: str, frames: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """挑时间戳最接近回复所述时刻的 1-2 张；抠不到时刻就返回空。"""
    wanted = _parse_hhmm_list(reply)
    if not wanted:
        return []
    scored = []
    for label, fp in frames:
        mm = _label_hhmm(label)
        if mm is None:
            continue
        d = min(abs(mm - w) for w in wanted)
        scored.append((d, label, fp))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    best = []
    seen = set()
    for d, label, fp in scored:
        if fp in seen:
            continue
        seen.add(fp)
        best.append((label, fp))
        if len(best) >= 2:
            break
    return best


def split_sentences(text: str) -> list[str]:
    """首句抢跑：第一段遇到逗号/顿号/冒号且已够 6 字就先切一刀，让 TTS 立刻开工；
    其余仍按句末标点切。"""
    t = (text or "").strip()
    if not t:
        return []
    head = ""
    rest = t
    m = re.search(r"[，,、：:；;]", t)
    if m and len(re.sub(r"\s", "", t[: m.start()])) >= 6:
        head = t[: m.start()].strip()
        rest = t[m.end():].strip()
    parts = re.split(r"(?<=[。！？!?；;\n])", rest) if rest else []
    out = ([head] if head else []) + [p.strip() for p in parts if p.strip()]
    return [p for p in out if p]


@dataclass
class BrainResult:
    reply_text: str
    hud_patches: list[dict[str, Any]] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tts_speed: float = 1.0
    nav_patches: list[dict[str, Any]] = field(default_factory=list)
    pre_spoken: str = ""


LlmCaller = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[dict[str, Any]]]


class VoxBrain:
    """Direct FAEX1 OpenAI-compatible orchestrator with native tool calling."""

    def __init__(
        self,
        config: VoxConfig,
        memory: MemoryStore,
        vision: VisionStore,
        http_session: Any = None,
        llm_caller: LlmCaller | None = None,
        ring: Any = None,
        coordinator: DuplexCoordinator | None = None,
    ) -> None:
        self.cfg = config
        self.coordinator = coordinator
        self.memory = memory
        self.vision = vision
        self.http_session = http_session
        self._custom_caller = llm_caller
        self.ring = ring
        # 心跳与主回合共用同一份静态 prompt / 工具表（保 KV 前缀缓存）
        self.system_prompt = SYSTEM_PROMPT
        self.tools_schema = TOOLS_SCHEMA
        # 前台回合工具表（含派单工具）；后台工兵工具表剔除派单工具，防递归
        self.turn_tools = (
            TOOLS_SCHEMA + DUPLEX_TOOLS_SCHEMA if coordinator is not None else TOOLS_SCHEMA
        )
        self.bg_tools = [
            t for t in TOOLS_SCHEMA
            if t.get("function", {}).get("name") not in ("dispatch_bg_task", "cancel_bg_task")
        ]
        self.hormones = HormonesV2()
        self.amap = AmapNavEngine(http_session=http_session)
        self.todo_goals = TodoGoalStore(config.db_path)
        self.speaker_id = SpeakerIdentifier()
        self.social = SocialTracker()
        self.tool_forge = ToolForge(os.path.join(os.path.dirname(__file__), "custom_tools"))
        self._silenced_turn = False
        self.location_fn: Callable[[], dict[str, Any]] | None = None
        self._ha_cache: tuple[float, list[dict[str, Any]]] | None = None
        self.device_state_fn: Callable[[], dict[str, Any]] | None = None
        self.device_cmd_fn: Callable[[str, str], dict[str, Any]] | None = None
        self._turn_frames: list[tuple[str, str]] = []
        self._inspected_this_turn = False

    def _evidence_roots(self) -> list[str]:
        # 环形池（/dev/shm/vox-ring）也要在白名单里，否则瞬时回溯的原图推不上 HUD
        return [
            self.cfg.frame_dir,
            FRAME_ARCHIVE_DIR,
            os.environ.get("VOX_RING_DIR", "/dev/shm/vox-ring"),
        ]

    def _evidence_patch(self, header: str, lines: list[str], items: list[tuple[str, str]]) -> dict | None:
        try:
            patch = build_evidence_hud_patch(
                header=header,
                lines=lines,
                frame_items=items,
                allowed_roots=self._evidence_roots(),
            )
        except Exception as exc:
            log.warning("EVIDENCE_HUD err %s", exc)
            return None
        return patch if patch.get("images") else None


    async def _call_llm_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = TOOLS_SCHEMA,
        timeout_s: float | None = None,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """SSE 流式调用：正文按句抢跑（回调给 TTS），tool_calls 攒完照常返回。"""
        payload: dict[str, Any] = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature,
            "stream": True,
        }
        # 开源兼容：留空就完全不注入推理参数（标准 OpenAI / vLLM / Ollama / llama.cpp）
        _eff = (getattr(self.cfg, "llm_reasoning_effort", "") or "").strip().lower()
        if _eff:
            payload["reasoning_effort"] = _eff
            if _eff != "none":
                payload["max_thinking_tokens"] = int(getattr(self.cfg, "llm_max_thinking", 128) or 0)
        if not getattr(self.cfg, "llm_thinking", False):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": "Bearer " + self.cfg.llm_api_key,
                   "Content-Type": "application/json"}
        _t = float(timeout_s or self.cfg.llm_timeout_s)
        _ct = aiohttp.ClientTimeout(total=_t, sock_read=_t, connect=min(10.0, _t))
        content = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        _first_logged = False
        # 2026-11-22：模型偶尔把 "[主动] xxx" 或舞台说明"（保持安静…）"当正文吐出来。
        # 这类内容只属于主动播报通道，绝不能在对话回合里念出来（否则会被记忆自我强化）。
        _marked = False
        spoken_text = ""
        buf = ""
        _t0 = time.time()
        _ttft = None
        async with self.http_session.post(self.cfg.llm_url, json=payload,
                                          headers=headers, timeout=_ct) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if _ttft is None:
                        _ttft = (time.time() - _t0) * 1000
                    content += piece
                    if on_sentence is not None:
                        buf += piece
                        while True:
                            cut = -1
                            for idx, ch in enumerate(buf):
                                if ch in "，、：；。！？,;:!?" and idx >= 5:
                                    cut = idx + 1
                                    break
                            if cut < 0 and len(buf) >= 40:
                                cut = len(buf)
                            if cut <= 0:
                                break
                            sent_piece = buf[:cut].strip()
                            buf = buf[cut:]
                            if not sent_piece:
                                continue
                            if not _marked and not spoken_text and _MARK_HEAD_RE.match(sent_piece):
                                _marked = True
                                log.info("STREAM_MARK_DROP head=%r", sent_piece[:24])
                            if _marked:
                                continue
                            if _META_ONLY_RE.match(sent_piece):
                                log.info("STREAM_META_DROP piece=%r", sent_piece[:24])
                                continue
                            if not _first_logged:
                                _first_logged = True
                                log.info("LLM_TTFT %.0fms first_sent=%r", _ttft or 0, sent_piece[:24])
                            spoken_text += sent_piece
                            try:
                                await on_sentence(sent_piece)
                            except Exception as exc:
                                log.warning("STREAM_TTS err %s", exc)
                for tc in (delta.get("tool_calls") or []):
                    i = int(tc.get("index", 0))
                    cur = tool_calls.setdefault(i, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        cur["name"] += fn["name"]
                    if fn.get("arguments"):
                        cur["arguments"] += fn["arguments"]
        if on_sentence is not None and not tool_calls and not _marked:
            _tail = buf.strip()
            if _tail and _META_ONLY_RE.match(_tail):
                log.info("STREAM_META_DROP tail=%r", _tail[:24])
                _tail = ""
            if _tail:
                spoken_text += _tail
                try:
                    await on_sentence(_tail)
                except Exception as exc:
                    log.warning("STREAM_TTS tail err %s", exc)
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        msg["_spoken"] = spoken_text
        msg["_agency_marked"] = _marked
        if tool_calls:
            msg["tool_calls"] = [
                {"id": v["id"] or ("call_%d" % k), "type": "function",
                 "function": {"name": v["name"], "arguments": v["arguments"] or "{}"}}
                for k, v in sorted(tool_calls.items())
            ]
        if _ttft is not None:
            log.info("LLM_STREAM ttft=%.0fms chars=%d tools=%d", _ttft, len(content), len(tool_calls))
        return msg
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = TOOLS_SCHEMA,
        timeout_s: float | None = None,
        tool_choice: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if self._custom_caller is not None:
            return await self._custom_caller(messages, tools)
        payload: dict[str, Any] = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature,
        }
        _eff = (reasoning_effort or getattr(self.cfg, "llm_reasoning_effort", "") or "").strip().lower()
        if _eff:
            payload["reasoning_effort"] = _eff
            if _eff != "none":
                payload["max_thinking_tokens"] = int(getattr(self.cfg, "llm_max_thinking", 128) or 0)
        if not getattr(self.cfg, "llm_thinking", False):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        headers = {
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
            "Content-Type": "application/json",
        }
        # 2026-11-22: 传裸数字只约束 total，sock_read 仍是无限 —— 上游一旦半开连接
        # 就会永远挂在读 body 上（实测卡死 >2 分钟）。三档都钉死。
        _t = float(timeout_s or self.cfg.llm_timeout_s)
        _ct = aiohttp.ClientTimeout(total=_t, sock_read=_t, connect=min(10.0, _t))
        async with self.http_session.post(
            self.cfg.llm_url, json=payload, headers=headers, timeout=_ct,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            _msg = data["choices"][0]["message"]
            try:
                _msg = dict(_msg)
                _msg["finish_reason"] = data["choices"][0].get("finish_reason")
            except Exception:
                pass
            return _msg

    def _build_context_header(self, client_id: str, guest: bool = False) -> str:
        now = datetime.now(tz=TZ_SHANGHAI)
        wd = WEEKDAYS_ZH[now.weekday()]
        time_line = f"[CURRENT_TIME | {now.strftime('%Y-%m-%d %H:%M:%S')} {wd} (北京时间)]"
        live_line = self.vision.format_live_view_header(client_id)
        facts = self.memory.get_known_facts(client_id, limit=10)
        facts_block = (
            "[KNOWN_FACTS | 长期事实]\n" + "\n".join(f"- {f}" for f in facts)
            if facts
            else "[KNOWN_FACTS | 暂无]"
        )
        dev_line = ""
        try:
            if self.device_state_fn is not None:
                ds = self.device_state_fn() or {}
                if ds:
                    _bat = ds.get("battery_pct", -1)
                    _chg = "充电中" if ds.get("charging") else "未充电"
                    dev_line = (
                        "[DEVICE] 电量%s%%（%s） 机身%s°C 亮度%s%% 音量%s/%s 模式:%s"
                        % (_bat, _chg, ds.get("temp_c", "?"), ds.get("brightness", "?"),
                           ds.get("volume", "?"), ds.get("volume_max", "?"), ds.get("mode", "LIVE"))
                    )
        except Exception:
            dev_line = ""
        head = f"{time_line}\n{live_line}\n{facts_block}"
        out = head + ("\n" + dev_line if dev_line else "")
        try:
            loc = self.location_fn() if self.location_fn else None
            place = (loc or {}).get("place") or ""
            road = (loc or {}).get("road") or ""
            if not guest and (place or road):
                _bits = [b for b in (place, road) if b]
                out += "\n[PLACE] " + " · ".join(_bits)
        except Exception:
            pass
        return out

    async def run_bg_worker(
        self, client_id: str, goal: str, need_vision: bool = False,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """后台工兵：独立 messages，复用静态 SYSTEM_PROMPT 保 FAEX1 前缀缓存。

        返回 (口语结果, HUD patches, NAV patches)。工具表已剔除派单工具，不会有递归。
        """
        hud_patches: list[dict[str, Any]] = []
        _saved_nav = list(getattr(self, "_nav_patches", []) or [])
        _saved_frames = list(getattr(self, "_turn_frames", []) or [])
        self._nav_patches = []
        self._turn_frames = []
        _nav: list[dict[str, Any]] = []
        _vis = (
            "需要判断眼前画面时可调用 describe_live_view / inspect_recent_frames。"
            if need_vision else "本次不需要看画面。"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[BG_WORKER] 你是后台执行工兵，目标是：{goal}\n"
                    f"{_vis}\n{_build_time_line()}\n"
                    "可以用工具一步步做完，全部做完后只输出 1 句自然口语结果（不超过 28 字），"
                    "不要复述过程、不要提工具名或系统标签。"
                ),
            },
        ]
        reply = ""
        try:
            for _ in range(6):
                msg = await self._call_llm(
                    messages, tools=self.bg_tools, reasoning_effort="medium",
                )
                tcalls = msg.get("tool_calls") or []
                if not tcalls:
                    reply = str(msg.get("content") or "").strip()
                    break
                messages.append(msg)
                for tc in tcalls:
                    fn = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except Exception:
                        args = {}
                    log.info("BG_TOOL id_goal=%r tool=%s", goal[:24], fn)
                    out = await self._exec_tool(
                        client_id, fn, args, hud_patches, messages, None,
                        guest=False, guest_since=0.0,
                    )
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            if not reply:
                messages.append({
                    "role": "system",
                    "content": "请直接用一句口语汇报最终结果，不要再调用工具。",
                })
                _final = await self._call_llm(
                    messages, tools=self.bg_tools, tool_choice="none",
                    reasoning_effort="medium",
                )
                reply = str(_final.get("content") or "").strip()
        finally:
            _nav = list(getattr(self, "_nav_patches", []) or [])
            self._nav_patches = _saved_nav
            self._turn_frames = _saved_frames
        return reply, hud_patches, _nav

    async def _exec_tool(
        self, client_id: str, name: str, args: dict[str, Any],
        hud_patches: list[dict[str, Any]], messages: list[dict[str, Any]] | None = None,
        on_slow_tool: Any = None, guest: bool = False, guest_since: float = 0.0,
    ) -> str:
        # 访客模式硬隔离：入口先拦，不靠模型自觉
        if guest:
            if name == "control_glasses":
                _act = str(args.get("action", "") or "")
                if _act not in GUEST_CTRL_ALLOW:
                    log.info("GUEST_BLOCK tool=control_glasses action=%s", _act)
                    return json.dumps({"ok": False, "reason": GUEST_BLOCK_MSG}, ensure_ascii=False)
            elif name == "inspect_recent_frames":
                try:
                    _elapsed = max(0.5, time.time() - guest_since)
                    args = dict(args)
                    args["seconds_ago_start"] = min(
                        float(args.get("seconds_ago_start", 15) or 15), _elapsed)
                    args["seconds_ago_end"] = min(
                        float(args.get("seconds_ago_end", 0) or 0), _elapsed)
                except Exception:
                    pass
            elif name not in GUEST_TOOL_ALLOW:
                log.info("GUEST_BLOCK tool=%s", name)
                return json.dumps({"ok": False, "reason": GUEST_BLOCK_MSG}, ensure_ascii=False)

        if name == "dispatch_bg_task":
            if self.coordinator is None:
                return json.dumps({"ok": False, "reason": "后台双脑未启用"}, ensure_ascii=False)
            _goal = str(args.get("goal") or "").strip()
            if not _goal:
                return json.dumps({"ok": False, "reason": "goal 不能为空"}, ensure_ascii=False)
            try:
                rec = self.coordinator.dispatch(
                    client_id, _goal,
                    lambda t: self.run_bg_worker(client_id, t.goal, t.need_vision),
                    need_vision=bool(args.get("need_vision")),
                    guest_mode=guest,
                )
            except PermissionError:
                return json.dumps({"ok": False, "reason": GUEST_BLOCK_MSG}, ensure_ascii=False)
            except Exception as exc:
                log.warning("BG_DISPATCH err %s", exc)
                return json.dumps({"ok": False, "reason": str(exc)[:80]}, ensure_ascii=False)
            return json.dumps(
                {
                    "ok": True,
                    "task_id": rec.task_id,
                    "hint": "已转后台处理，请立刻用一句口语简短答应用户，不要原地等结果",
                },
                ensure_ascii=False,
            )

        if name == "cancel_bg_task":
            if self.coordinator is None:
                return json.dumps({"ok": False, "reason": "后台双脑未启用"}, ensure_ascii=False)
            _ids = self.coordinator.cancel(client_id, str(args.get("task_id") or ""))
            return json.dumps({"ok": True, "cancelled": _ids}, ensure_ascii=False)

        if name == "save_place":
            if guest:
                return json.dumps({"ok": False, "reason": GUEST_BLOCK_MSG}, ensure_ascii=False)
            loc = (self.location_fn() if self.location_fn else None) or {}
            if not loc.get("lat"):
                return json.dumps({"ok": False, "reason": "现在拿不到定位，等有 GPS 再记"}, ensure_ascii=False)
            r = self.memory.save_place(
                client_id, str(args.get("name", "")).strip(),
                float(loc["lat"]), float(loc["lon"]),
                float(args.get("radius_m") or 100.0),
            )
            hud_patches.append({
                "type": "HUD_PATCH", "level": 1, "slot": "right_evidence", "accent": "isac_cyan",
                "header": "PLACE · saved", "lines": [str(r.get("name") or args.get("name"))],
            })
            return json.dumps(r, ensure_ascii=False)

        if name == "where_am_i":
            if guest:
                return json.dumps({"ok": False, "reason": GUEST_BLOCK_MSG}, ensure_ascii=False)
            loc = (self.location_fn() if self.location_fn else None) or {}
            if not loc.get("lat"):
                return json.dumps({"ok": False, "reason": "现在没有有效定位"}, ensure_ascii=False)
            from .geo import haversine_m
            lat, lon = float(loc["lat"]), float(loc["lon"])
            near = self.memory.match_place(client_id, lat, lon)
            out = {"ok": True, "place": (near or {}).get("name") or "unknown地点"}
            # 精确路名：配了高德「Web 服务」Key 才补逆地理，没有就静默跳过
            try:
                rg = await self.amap.regeo(lat, lon)
                if rg:
                    if rg.get("road"):
                        out["road"] = rg["road"]
                        if rg.get("number"):
                            out["road"] += rg["number"] if rg["number"][:1].isdigit() else " " + rg["number"]
                    if rg.get("address"):
                        out["address"] = rg["address"]
                    if rg.get("district"):
                        out["district"] = rg["district"]
                    if out.get("place") in ("unknown地点", "未知地点") and (rg.get("road") or rg.get("address")):
                        out["place"] = rg.get("road") or rg.get("address")
            except Exception as exc:
                log.debug("where_am_i regeo skip: %s", exc)
            for label in ("家", "公司", "上次停车位"):
                pl = self.memory.get_place(client_id, label)
                if pl:
                    d = haversine_m(lat, lon, float(pl["lat_gcj"]), float(pl["lon_gcj"]))
                    out[label] = int(d) if d >= 1000 else round(d)
            return json.dumps(out, ensure_ascii=False)

        if name in ("list_custom_tools", "save_custom_tool", "run_custom_tool"):
            try:
                return json.dumps(self.tool_forge.exec_tool(name, args), ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "reason": str(exc)[:120]}, ensure_ascii=False)

        if name == "do_nothing":
            note = str(args.get("situation_note", "") or "").strip()
            if note:
                try:
                    self.social.update_situation(note)
                except Exception:
                    pass
            self._silenced_turn = True
            log.info("DO_NOTHING reason=%r note=%r", args.get("reason"), note[:60])
            return json.dumps({"ok": True, "silent": True}, ensure_ascii=False)

        if name == "control_glasses":
            act = str(args.get("action", "") or "").strip()
            val = str(args.get("value", "") or "").strip()
            try:
                res = self.device_cmd_fn(act, val) if self.device_cmd_fn else {"ok": False, "reason": "no device"}
            except Exception as exc:
                res = {"ok": False, "reason": str(exc)[:80]}
            hud_patches.append({
                "type": "HUD_PATCH", "level": 1, "slot": "right_evidence", "accent": "isac_cyan",
                "header": "DEVICE · " + (act or "cmd"),
                "lines": [("已下发 " + act + (" " + val if val else "")).strip()],
            })
            log.info("DEVICE_CMD_TOOL %s=%r -> %s", act, val, res)
            return json.dumps(res, ensure_ascii=False)

        if name == "search_memory":
            _q = str(args.get("query", "") or "").strip()
            _lo = _parse_iso_bj(args.get("start_iso"))
            _hi = _parse_iso_bj(args.get("end_iso"))
            log.info(
                "SEARCH_MEM q=%r lo=%s hi=%s", _q,
                args.get("start_iso"), args.get("end_iso"),
            )
            _near = bool(args.get("near_here"))
            _loc = (self.location_fn() if self.location_fn else None) or {}
            res = self.memory.search_memory(
                client_id=client_id,
                query=_q,
                start_ts=_lo,
                end_ts=_hi,
                limit=int(args.get("limit") or 16),
                order=str(args.get("order") or "desc"),
                kind=str(args.get("kind") or "all"),
                offset=int(args.get("offset") or 0),
                near_lat=float(_loc["lat"]) if (_near and _loc.get("lat")) else None,
                near_lon=float(_loc["lon"]) if (_near and _loc.get("lon")) else None,
                near_radius_m=150.0,
            )
            # A keyword the captions never used must not hide an otherwise recorded
            # time window. Retry on the window alone and label the fallback so the
            # model knows it is looking at time evidence, not a keyword match.
            if not res["found"] and _q and (_lo is not None or _hi is not None):
                res = self.memory.search_memory(
                    client_id=client_id,
                    query="",
                    start_ts=_lo,
                    end_ts=_hi,
                    limit=int(args.get("limit") or 16),
                    order=str(args.get("order") or "desc"),
                    kind=str(args.get("kind") or "all"),
                    offset=int(args.get("offset") or 0),
                )
                if res["found"]:
                    res["mode"] = "time_window_fallback"
            # Always surface the evidence panel — when nothing is found the HUD must
            # say so, otherwise the user cannot tell "checked and empty" from "did nothing".
            if res["found"]:
                lines = [f"{h['time']} {h['text'][:42]}" for h in res["hits"][:4]]
            else:
                _win = "%s ~ %s" % (
                    args.get("start_iso") or "不限",
                    args.get("end_iso") or "不限",
                )
                lines = [f"检索窗口：{_win}", "该时间段没有可用记录"]
            hud_patches.append(
                {
                    "type": "HUD_PATCH",
                    "level": 2,
                    "slot": "right_evidence",
                    "accent": "isac_cyan",
                    "header": "MEMORY · Evidence" if res["found"] else "MEMORY · Empty",
                    "lines": lines,
                }
            )
            log.info(
                "SEARCH_RET found=%s hits=%d frames=%d",
                res.get("found"), res.get("count", 0), len(res.get("frame_paths") or []),
            )
            _vh = [h for h in (res.get("hits") or []) if h.get("frame_path")]
            for _h in _vh:
                self._turn_frames.append((str(_h.get("time", "")), _h["frame_path"]))
            _items = [
                (h["frame_path"], (str(h.get("time", "")) + " " + str(h.get("text", ""))[:24]).strip())
                for h in _vh[:2]
            ]
            _lines = [(str(h.get("time", "")) + " " + str(h.get("text", ""))[:24]).strip() for h in _vh[:3]] or lines
            _ep = self._evidence_patch("MEMORY · Evidence", lines=_lines, items=_items) if _items else None
            if _ep:
                hud_patches.append(_ep)
            return json.dumps(res, ensure_ascii=False)

        if name == "inspect_past_frames":
            # 看图要十几秒，先出一句垫场话（语音场景下别让她干等着）
            if on_slow_tool is not None:
                try:
                    await on_slow_tool("我翻一下当时的照片")
                except Exception:
                    pass
            # 2026-11-22 设计决定：不再另起一次 VLM 调用。把原图作为 role:user 多模态
            # 直接追加进本轮主上下文，让主脑亲眼看图；只活在本次 run_turn 的
            # messages 里，不写库、不进历史（record_turn 只存文字）。
            raw_paths = args.get("frame_paths") or []
            if isinstance(raw_paths, str):
                raw_paths = [raw_paths]
            question = str(args.get("question") or "简要描述画面里的关键物体、文字与人物。")
            allowed_roots = [self.cfg.frame_dir, FRAME_ARCHIVE_DIR]
            picked: list[str] = []
            rejected: list[str] = []
            for p in raw_paths:
                if len(picked) >= 3:
                    break
                try:
                    rp = os.path.realpath(str(p))
                except Exception:
                    rejected.append(str(p))
                    continue
                if not any(rp.startswith(os.path.realpath(root) + os.sep) for root in allowed_roots):
                    rejected.append(str(p))
                    continue
                if not os.path.isfile(rp):
                    rejected.append(str(p))
                    continue
                picked.append(rp)
            if not picked:
                return json.dumps(
                    {"ok": False, "reason": "没有可用的原图（路径不合法或文件已被清理）",
                     "rejected": rejected[:3]},
                    ensure_ascii=False,
                )
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": f"（我翻出了当时拍的原图，{len(picked)} 张，请直接看图回答：{question}）"}
            ]
            for fp in picked:
                try:
                    with open(fp, "rb") as f:
                        raw = f.read()
                except Exception:
                    continue
                ext = os.path.splitext(fp)[1].lower()
                mime = "image/webp" if ext == ".webp" else "image/jpeg"
                b64 = base64.b64encode(raw).decode("ascii")
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            if len(parts) == 1:
                return json.dumps({"ok": False, "reason": "原图读取失败"}, ensure_ascii=False)
            if messages is not None:
                messages.append({"role": "user", "content": parts})
            self._inspected_this_turn = True
            _items = [(fp, "") for fp in picked[:2]]
            _ep = self._evidence_patch("PHOTO · Evidence", lines=[question[:36]], items=_items)
            if _ep:
                hud_patches.append(_ep)
            return json.dumps(
                {
                    "ok": True,
                    "attached": len(parts) - 1,
                    "note": "原图已附在本轮上下文（仅本轮可见，不入历史），直接看图回答，不要再说看不到。",
                },
                ensure_ascii=False,
            )

        if name == "get_weather":
            try:
                res = await self.amap.get_weather(client_id, city=str(args.get("city", "") or ""))
            except Exception as exc:
                res = {"ok": False, "reason": str(exc)}
            return json.dumps(res, ensure_ascii=False)

        if name == "start_navigation":
            try:
                res = await self.amap.start_navigation(client_id, str(args.get("dest", "") or ""), str(args.get("city", "") or ""))
            except Exception as exc:
                res = {"ok": False, "reason": str(exc)}
            _np = res.get("nav_patch") if isinstance(res, dict) else None
            if _np:
                self._nav_patches.append(_np)
            return json.dumps(res, ensure_ascii=False)

        if name == "stop_navigation":
            res = self.amap.stop_navigation(client_id, reason="user_request")
            _np = res.get("nav_patch") if isinstance(res, dict) else None
            if _np:
                self._nav_patches.append(_np)
            return json.dumps(res, ensure_ascii=False)

        if name == 'do_nothing':
            return json.dumps({'ok': True, 'note': 'no_action'}, ensure_ascii=False)

        if name == 'proactive_speak':
            return json.dumps({'ok': True, 'note': 'handled_by_agency'}, ensure_ascii=False)

        if name in ('manage_todo', 'manage_goal'):
            try:
                res = self.todo_goals.exec_tool(client_id, name, args)
            except Exception as exc:
                return json.dumps({'ok': False, 'reason': str(exc)}, ensure_ascii=False)
            return json.dumps(res, ensure_ascii=False)

        if name == "inspect_recent_frames":
            if self.ring is None:
                return json.dumps({"ok": False, "reason": "环形池未启用"}, ensure_ascii=False)
            _start = float(args.get("seconds_ago_start", 15) or 15)
            _end = float(args.get("seconds_ago_end", 0) or 0)
            _cnt = max(1, min(3, int(args.get("count", 3) or 3)))
            frames = self.ring.recent(client_id, _start, _end, _cnt)
            if not frames:
                return json.dumps(
                    {"ok": False, "reason": "刚刚这段没有画面（环形池为空或时间窗太窄）"},
                    ensure_ascii=False,
                )
            parts = [{"type": "text", "text": "（这是刚刚 %d 秒内的原图，共 %d 张，请直接看图回答）" % (int(_start), len(frames))}]
            items = []
            for fr in frames:
                mime = "image/jpeg" if str(fr["path"]).endswith(".jpg") else "image/webp"
                b64 = base64.b64encode(fr["bytes"]).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}})
                items.append((fr["path"], time.strftime("%H:%M:%S", time.localtime(fr["ts"]))))
            if messages is not None:
                messages.append({"role": "user", "content": parts})
            self._inspected_this_turn = True
            _ep = self._evidence_patch("PHOTO · Just now", lines=[it[1] for it in items][:3], items=items)
            if _ep:
                hud_patches.append(_ep)
            return json.dumps(
                {"ok": True, "attached": len(frames), "note": "原图已附在本轮上下文，直接看图回答。"},
                ensure_ascii=False,
            )

        if name == "browse_vision_timeline":
            _lo = _parse_iso_bj(args.get("start_iso"))
            _hi = _parse_iso_bj(args.get("end_iso"))
            tl = self.memory.browse_vision_timeline(
                client_id=client_id,
                start_ts=_lo,
                end_ts=_hi,
                max_points=int(args.get("max_points") or 24),
            )
            log.info(
                "TIMELINE lo=%s hi=%s points=%d total=%d",
                args.get("start_iso"), args.get("end_iso"),
                tl.get("count", 0), tl.get("total_in_window", 0),
            )
            if tl.get("found"):
                hud_patches.append(
                    {
                        "type": "HUD_PATCH",
                        "level": 2,
                        "slot": "right_evidence",
                        "accent": "isac_cyan",
                        "header": "VISION · Timeline",
                        "lines": [
                            p["time"] + "  " + (p["caption"] or "")[:34]
                            for p in tl["points"][:4]
                        ],
                    }
                )
            _items = []
            for _pt in (tl.get("points") or []):
                _fp = _pt.get("best_frame_path") or ""
                if not _fp:
                    continue
                _items.append((_fp, (str(_pt.get("time", "")) + " " + str(_pt.get("caption", ""))[:24]).strip()))
                self._turn_frames.append((str(_pt.get("time", "")), _fp))
                if len(_items) >= 2:
                    break
            _ep = self._evidence_patch(
                "VISION · Timeline",
                lines=[str(_pt.get("time", "")) + " " + str(_pt.get("caption", ""))[:24] for _pt in (tl.get("points") or [])[:3]],
                items=_items,
            ) if _items else None
            if _ep:
                hud_patches.append(_ep)
            return json.dumps(tl, ensure_ascii=False)

        if name == "describe_live_view":
            img = self.vision.get_fresh_image_b64(client_id)
            if not img:
                return json.dumps({"ok": False, "reason": "当前没有新鲜的眼镜画面（已过期或未采集）"}, ensure_ascii=False)
            mime, b64 = img
            q = str(args.get("question", "简要描述眼前画面里的关键物体与文字。"))
            v_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ]
            msg = await self._call_llm(v_msgs, tools=None)
            return json.dumps({"ok": True, "description": msg.get("content", "")}, ensure_ascii=False)

        if name == "save_user_fact":
            fid = self.memory.save_fact(
                client_id=client_id,
                predicate=str(args.get("predicate", "fact:note")),
                object_text=str(args.get("value", "")),
            )
            return json.dumps({"ok": True, "fact_id": fid}, ensure_ascii=False)

        if name == "set_reminder":
            _title = str(args.get("title", ""))
            _due = str(args.get("due_str", ""))
            jid = self.memory.add_reminder(client_id=client_id, title=_title, due_str=_due)
            # 2026-11-22：提醒与待办统一落到 todos（单一事实源），
            # 否则模型习惯性选旧 set_reminder 时 todos 永远是空的。
            try:
                self.todo_goals.add_todo(client_id, _title, due_iso=_due, note="提醒")
            except Exception as exc:
                log.warning("TODO mirror err %s", exc)
            return json.dumps({"ok": True, "job_id": jid, "todo_mirrored": True}, ensure_ascii=False)

        if name == "list_ha_entities":
            if not (self.cfg.ha_url and self.cfg.ha_token and self.http_session):
                return json.dumps({"ok": False, "reason": "Home Assistant 未配置"}, ensure_ascii=False)
            now = time.time()
            states: list[dict[str, Any]] = []
            if self._ha_cache is not None and now - self._ha_cache[0] < 30.0:
                states = self._ha_cache[1]
            else:
                try:
                    _u = self.cfg.ha_url.rstrip("/") + "/api/states"
                    _h = {"Authorization": "Bearer " + self.cfg.ha_token}
                    async with self.http_session.get(_u, headers=_h) as r:
                        if r.status >= 300:
                            return json.dumps({"ok": False, "status": r.status}, ensure_ascii=False)
                        states = await r.json()
                    self._ha_cache = (now, states)
                except Exception as exc:
                    return json.dumps({"ok": False, "reason": str(exc)[:120]}, ensure_ascii=False)
            _dom = str(args.get("domain", "") or "").strip().lower()
            _q = str(args.get("query", "") or "").strip().lower()
            _only_on = bool(args.get("only_on"))
            _tokens = [t for t in re.split(r"[\s,，、]+", _q) if t]
            _on_states = {"on", "open", "opening", "heating", "cooling", "heat", "cool", "playing", "home"}
            _keep = ("friendly_name", "brightness", "color_temp_kelvin", "temperature",
                     "current_temperature", "hvac_mode", "unit_of_measurement", "device_class")
            def _collect(use_tokens: bool) -> list[dict[str, Any]]:
                picked: list[tuple[int, dict[str, Any]]] = []
                for st in states:
                    eid = str(st.get("entity_id", ""))
                    if _dom and not eid.startswith(_dom + "."):
                        continue
                    attrs = st.get("attributes") or {}
                    name = str(attrs.get("friendly_name") or eid)
                    state = str(st.get("state", ""))
                    if _only_on and state.lower() not in _on_states:
                        continue
                    score = 0
                    if _q:
                        hay = (name + " " + eid).lower()
                        if use_tokens and _tokens:
                            score = sum(1 for t in _tokens if t in hay)
                            if score == 0:
                                continue
                        else:
                            if _q not in hay:
                                continue
                            score = 1
                    picked.append((score, {
                        "entity_id": eid,
                        "name": name,
                        "state": state,
                        "attrs": {k: attrs[k] for k in _keep if k in attrs},
                    }))
                picked.sort(key=lambda x: -x[0])
                return [p[1] for p in picked]

            out = _collect(True)
            _note = ""
            if not out and _q and _dom:
                # 大模型把「游戏房 窗帘」当整串搜会 0 命中；这里给回该域全量清单，
                # 让它能直接回答"没有这个设备，只有这些"，而不是无限重试。
                out = _collect(False)
                _note = "query 没命中，下面是该域的全部设备"
            log.info("HA_QUERY dom=%r q=%r only_on=%s -> %d %s", _dom, _q, _only_on, len(out), _note)
            res_obj = {"ok": True, "total": len(out), "entities": out[:20]}
            if _note:
                res_obj["note"] = _note
            return json.dumps(res_obj, ensure_ascii=False)

        if name == "call_home_assistant":
            _dom = str(args.get("domain", "") or "").strip().lower()
            # 安全护栏优先于任何配置检查：门锁/安防永远拒绝
            if _dom in ("lock", "alarm_control_panel"):
                log.warning("HA_BLOCKED domain=%s service=%s", _dom, args.get("service"))
                return json.dumps(
                    {"ok": False, "reason": "门锁/安防设备不允许语音控制，请在 App 里操作"},
                    ensure_ascii=False,
                )
            if not (self.cfg.ha_url and self.cfg.ha_token and self.http_session):
                return json.dumps({"ok": False, "reason": "Home Assistant 未配置"}, ensure_ascii=False)
            _payload: dict[str, Any] = {"entity_id": args["entity_id"]}
            _extra = args.get("data")
            if isinstance(_extra, dict):
                _payload.update(_extra)
            url = f"{self.cfg.ha_url.rstrip('/')}/api/services/{_dom}/{args['service']}"
            headers = {"Authorization": f"Bearer {self.cfg.ha_token}", "Content-Type": "application/json"}
            async with self.http_session.post(url, json=_payload, headers=headers) as r:
                _ok = r.status < 300
                self._ha_cache = None  # 状态已变，清缓存
                log.info("HA_CALL %s.%s %s data=%s -> %s", _dom, args.get("service"),
                         args.get("entity_id"), _extra, r.status)
                return json.dumps({"ok": _ok, "status": r.status, "payload": _payload}, ensure_ascii=False)

        return json.dumps({"ok": False, "reason": f"unknown tool: {name}"}, ensure_ascii=False)

    async def run_turn(
        self, client_id: str, user_text: str, epoch_id: int = 0,
        on_slow_tool: Any = None, speaker: str = "", spk_sim: float = 0.0,
        social_block: str = "", guest_mode: bool = False, guest_since_ts: float = 0.0,
        first_sentence_cb: Any = None,
    ) -> BrainResult:
        guest = bool(guest_mode)
        guest_since = float(guest_since_ts or 0.0)
        self._turn_frames = []
        self._inspected_this_turn = False
        self._nav_patches = []
        self._silenced_turn = False
        try:
            self.hormones.on_user_reply(text_len=len(user_text or ""))
        except Exception:
            pass
        _dyn_tail = ""
        try:
            _dyn_tail = "\n\n" + self.todo_goals.format_context_block(client_id) + "\n" + self.hormones.dynamic_tail_directive(is_proactive=False)
        except Exception:
            _dyn_tail = ""
        # 双脑：把在途/刚完成未播报的后台任务压成 [BG_TASKS] 拼进动态尾段
        _bg_ids: list[str] = []
        if self.coordinator is not None and not guest:
            try:
                _bg_block = self.coordinator.format_context_block(client_id, guest_mode=guest)
                if _bg_block:
                    _dyn_tail = _dyn_tail + "\n\n" + _bg_block
                _bg_ids = [
                    r.task_id for r in self.coordinator.list_tasks(client_id)
                    if r.status in ("done", "failed") and not r.announced
                ]
                _dyn_tail = _dyn_tail + (
                    "\n[双脑] 涉及米家/Home Assistant 设备控制、翻查旧记忆或原图、"
                    "查天气/路线导航、造工具这类要多步或要等的事，一律先 dispatch_bg_task "
                    "交给后台，然后立刻用一句短话答应用户、释放对话；只有纯聊天和"
                    "一眼就能答的小问题才自己答完。"
                )
            except Exception as exc:
                log.warning("BG_TAIL err %s", exc)
        if social_block:
            _dyn_tail = (
                _dyn_tail + "\n" + social_block +
                "\n[注意] 若当前这句是直接对你说的（提问或指令），必须正常回答；"
                "只有其他人之间闲聊、没在问你时，才 do_nothing(situation_note=...)。"
            )
        _user_store = (("[%s] " % speaker) if speaker else "") + user_text
        self.memory.record_turn(
            client_id, "user", _user_store, epoch_id=epoch_id,
            is_poisoned=1 if guest else 0,
        )
        # KV 缓存：system 只放静态 SYSTEM_PROMPT，动态上下文拼到最后一条 user 上，
        # 这样 system + 历史对话两段前缀都能被 FAEX1 完整命中。
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if guest:
            try:
                _recent = [
                    t for t in self.memory.get_recent_turns(client_id, limit=12)
                    if float(t.get("ts") or 0) >= guest_since
                ]
            except Exception:
                _recent = []
            for t in _recent:
                messages.append({"role": t["role"], "content": t["text"]})
            _ctx = "\n".join([
                _build_time_line(),
                "[GUEST_MODE] 当前佩戴者是访客而非owner，请礼貌简短接待、可介绍眼前画面或陪聊，"
                "严禁透露任何机主隐私或假装执行已禁用的权限。",
            ])
        else:
            for t in self.memory.get_recent_turns(client_id, limit=6)[:-1]:
                messages.append({"role": t["role"], "content": t["text"]})
            _ctx = self._build_context_header(client_id, guest=guest) + _dyn_tail
        messages.append({"role": "user", "content": user_text + "\n\n" + _ctx})

        hud_patches: list[dict[str, Any]] = []
        tools_called: list[str] = []
        reply = ""

        _use_stream = (os.environ.get("VOX_LLM_STREAM", "1") != "0" and self._custom_caller is None)
        _pre_spoken = ""
        _steps = 0
        _finish = ""
        for _ in range(6):
            _steps += 1
            try:
                _cb = first_sentence_cb if _use_stream else None
                if _use_stream:
                    msg = await self._call_llm_stream(messages, tools=self.turn_tools, on_sentence=_cb)
                else:
                    msg = await self._call_llm(messages, tools=self.turn_tools)
            except Exception as exc:
                # 本轮里塞了原图但当前视觉后端不可用（实测 FAEX1:8731 对图片直接 502）。
                # 把图片摘掉、留一句提示后重试一次，宁可少看一张图也不能整轮哑掉。
                if _strip_images(messages):
                    log.warning("VISION_UNAVAILABLE %s -> 降级为纯文本重试", exc)
                    messages.append(
                        {"role": "system", "content": "（本轮的原始画面取不到了，只能凭文字记录回答，别提图。）"}
                    )
                    msg = await self._call_llm(messages, tools=self.turn_tools)
                else:
                    raise
            _finish = str(msg.get("finish_reason") or "")
            if _cb is not None and msg.get("_spoken"):
                _pre_spoken = _pre_spoken + str(msg.get("_spoken") or "")
            tcalls = msg.get("tool_calls") or []
            if msg.get("_agency_marked"):
                # 本轮回复整段都是主动播报通道的内容：不念、不记、按静默处理
                log.info("AGENCY_MARK_DROP reply=%r", str(msg.get("content") or "")[:40])
                self._silenced_turn = True
                reply = ""
                break
            if not tcalls:
                reply = str(msg.get("content") or "").strip()
                break
            messages.append(msg)
            for tc in tcalls:
                fn = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                tools_called.append(fn)
                log.info("TOOL_CALL name=%s args=%s", fn, json.dumps(args, ensure_ascii=False)[:160])
                tool_out = await self._exec_tool(
                    client_id, fn, args, hud_patches, messages, on_slow_tool,
                    guest=guest, guest_since=guest_since,
                )
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_out})

        # 2026-11-22：本轮没看过原图时，按回复里提到的时刻挑最接近的证据帧
        if reply and not self._inspected_this_turn and self._turn_frames:
            _picks = _pick_frames_by_reply_time(reply, self._turn_frames)
            if _picks:
                _ep = self._evidence_patch(
                    "PHOTO · Evidence",
                    lines=[lb for lb, _fp in _picks][:3],
                    items=_picks[:2],
                )
                if _ep:
                    hud_patches.append(_ep)
        hud_patches = merge_turn_hud_patches(hud_patches)
        if self._silenced_turn:
            # 明确选择不插嘴：不落 assistant 记录、不出声
            log.info("TURN_SILENT speaker=%r text=%r", speaker, user_text[:40])
            reply = ""
            try:
                _spd = float(self.hormones.tts_style().get("speed", 1.0))
            except Exception:
                _spd = 1.0
            return BrainResult(
                reply_text="", hud_patches=hud_patches, tools_called=tools_called, pre_spoken=_pre_spoken,
                tts_speed=_spd, nav_patches=list(getattr(self, "_nav_patches", []) or []),
            )
        if not reply and not self._silenced_turn:
            # 工具步数用尽或模型空回复：强制一次纯文本收尾，避免"我刚走神了"
            log.info("EMPTY_REPLY steps=%d tools=%s finish=%r raw=%r",
                     _steps, tools_called, _finish, (msg.get("content") if isinstance(msg, dict) else None))
            try:
                messages.append({
                    "role": "system",
                    "content": "请根据上面的工具执行结果，用简短口语直接回复用户；不要再调用工具。"
                               "若工具显示没有这个设备，就直接说明没有，并给出你查到的真实清单。",
                })
                _final = await self._call_llm(messages, tools=self.turn_tools, tool_choice="none")
                reply = str(_final.get("content") or "").strip()
                log.info("FINAL_FALLBACK reply=%r", reply[:80])
            except Exception as exc:
                log.warning("FINAL_FALLBACK err %s", exc)
                reply = ""
        reply = strip_proactive_marker(reply)
        if not reply:
            reply = "我刚走神了，麻烦你再说一遍。"
        # 前台已经在回复里带过话的后台结果，立刻标已播报，防 watcher 重复插话
        if _bg_ids and reply.strip():
            try:
                _marked = self.coordinator.mark_unannounced_consumed(client_id, _bg_ids)
                if _marked:
                    log.info("BG_ANNOUNCED ids=%s", _marked)
            except Exception:
                pass
        self.memory.record_turn(
            client_id, "assistant", reply, epoch_id=epoch_id,
            is_poisoned=1 if guest else 0,
        )
        try:
            self.hormones.log_causal_turn(trigger="dialog", reply_text=reply)
        except Exception:
            pass
        try:
            _speed = float(self.hormones.tts_style().get("speed", 1.0))
        except Exception:
            _speed = 1.0
        return BrainResult(
            reply_text=reply,
            pre_spoken=_pre_spoken,
            hud_patches=hud_patches,
            tools_called=tools_called,
            tts_speed=_speed,
            nav_patches=list(getattr(self, "_nav_patches", []) or []),
        )
