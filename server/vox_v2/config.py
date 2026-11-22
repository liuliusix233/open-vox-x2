"""Centralized environment configuration for vox_v2."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class VoxConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    db_path: str = "vox_unified.db"
    frame_dir: str = "/dev/shm/vox-frames"
    live_view_ttl_s: int = 300

    # Local FAEX1 Halogen (OpenAI-compatible)
    llm_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_api_key: str = "local"
    llm_timeout_s: float = 30.0
    llm_temperature: float = 0.35

    # Local ASR / TTS
    asr_url: str = ""  # Optional HTTP SenseVoice endpoint
    tts_url: str = "http://127.0.0.1:50002"
    tts_spk_id: str = "default"
    # 2026-11-22 开源：TTS 后端二选一
    #   cosyvoice = 本地 CosyVoice2 /inference_spk（默认，需 20 系以上显卡）
    #   openai    = 任意 OpenAI 兼容 /v1/audio/speech（OpenAI / Fish / GPT-SoVITS-API 等）
    tts_backend: str = "cosyvoice"
    tts_openai_url: str = ""
    tts_openai_key: str = ""
    tts_openai_model: str = "tts-1"
    tts_openai_voice: str = "alloy"

    # Optional Home Assistant integration
    tts_engine: str = "cosyvoice"
    gsv_url: str = ""
    gsv_ref_wav: str = ""
    gsv_prompt_text: str = ""
    llm_thinking: bool = False
    # 留空 = 不向接口注入 reasoning_effort / max_thinking_tokens（开箱兼容标准 OpenAI 协议）
    llm_reasoning_effort: str = ""
    llm_max_thinking: int = 128
    ha_url: str = ""
    ha_token: str = ""

    # Runtime guards
    default_client: str = "voxx2"
    auth_token: str = ""
    half_duplex_guard: bool = False  # Set VOX_HALF_DUPLEX_GUARD=1 while AEC is tuned

    # 2026-11-22: keep the legacy archiver alive for captioning / frame_caps.
    # vox_v2 has no caption worker, so each ingested frame is mirrored to this URL
    # (legacy POST /frame). Empty string disables mirroring.
    legacy_ingest_url: str = ""

    @classmethod
    def from_env(cls) -> "VoxConfig":
        return cls(
            host=os.getenv("VOX_HOST", "0.0.0.0"),
            port=int(os.getenv("VOX_PORT", "8766")),
            db_path=os.getenv("VOX_DB", "vox_unified.db"),
            frame_dir=os.getenv("VOX_FRAME_DIR", "/dev/shm/vox-frames"),
            live_view_ttl_s=int(os.getenv("VOX_LIVE_VIEW_TTL_S", "300")),
            llm_url=os.getenv("VOX_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions"),
            llm_model=os.getenv("VOX_LLM_MODEL", "qwen2.5-7b-instruct"),
            llm_api_key=os.getenv("VOX_LLM_API_KEY", "local"),
            llm_timeout_s=float(os.getenv("VOX_LLM_TIMEOUT_S", "30.0")),
            llm_temperature=float(os.getenv("VOX_LLM_TEMP", "0.35")),
            asr_url=os.getenv("VOX_ASR_URL", ""),
            tts_url=os.getenv("VOX_CV_TTS_URL", "http://127.0.0.1:50002"),
            tts_spk_id=os.getenv("VOX_CV_SPK_ID", "default"),
            tts_backend=os.getenv("VOX_TTS_BACKEND", "cosyvoice"),
            tts_openai_url=os.getenv("VOX_TTS_OPENAI_URL", ""),
            tts_openai_key=os.getenv("VOX_TTS_OPENAI_KEY", ""),
            tts_openai_model=os.getenv("VOX_TTS_OPENAI_MODEL", "tts-1"),
            tts_openai_voice=os.getenv("VOX_TTS_OPENAI_VOICE", "alloy"),
            tts_engine=os.getenv("VOX_TTS_ENGINE", "cosyvoice"),
            gsv_url=os.getenv("VOX_GSV_URL", ""),
            gsv_ref_wav=os.getenv("VOX_GSV_REF", ""),
            gsv_prompt_text=os.getenv("VOX_GSV_PROMPT", ""),
            llm_thinking=_env_bool("VOX_LLM_THINKING", False),
            llm_reasoning_effort=os.getenv("VOX_LLM_REASONING_EFFORT", ""),
            llm_max_thinking=int(os.getenv("VOX_LLM_MAX_THINKING", "128")),
            ha_url=os.getenv("VOX_HA_URL", ""),
            ha_token=os.getenv("VOX_HA_TOKEN", ""),
            default_client=os.getenv("VOX_DEFAULT_CLIENT", "voxx2"),
            auth_token=os.getenv("VOX_TOKEN", ""),
            half_duplex_guard=_env_bool("VOX_HALF_DUPLEX_GUARD", False),
            legacy_ingest_url=os.getenv("VOX_LEGACY_INGEST_URL", ""),
        )
