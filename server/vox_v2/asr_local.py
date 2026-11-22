"""Local SenseVoice ASR adapter for vox_v2 (in-process sherpa-onnx, CPU int8).

2026-11-22 对齐旧链路（server.py:10686 的真实写法）：
    原始 PCM -> _asr_normalize -> _trim_silence -> _agc_pcm -> SenseVoice
之前只做了 trim+agc，漏掉 _asr_normalize —— 它对 X2 这种低电平近嘴麦克风是决定性的：
RMS 对齐到约 -20dBFS（最多 16x），带噪声门与风噪冻结，避免"小声 → 全糊"。
另外补上旧链路的 min_valid_chars=2：去掉「。」「嗯」这种单字误触发。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time

log = logging.getLogger("vox_v2.asr")

VOICE_SERVER_DIR = os.environ.get(
    "VOX_LEGACY_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_server"),
)
SENSE_MODEL_DIR = os.path.join(
    VOICE_SERVER_DIR, "models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
)
AUDIO_SAMPLE_RATE = 16000

# 旧链路 duplex 的 min_utter=8000B（0.25s@16k）：更短的片段一律不送 ASR
MIN_UTTER_BYTES = int(os.environ.get("VOX_ASR_MIN_UTTER_BYTES", "8000"))
# 旧链路 min_valid_chars=2：去标点后少于 2 个字视为空
MIN_VALID_CHARS = int(os.environ.get("VOX_ASR_MIN_VALID_CHARS", "2"))
DUMP_DIR = os.environ.get("VOX_ASR_DUMP_DIR", "/tmp/vox_dump")
_dump_count = 0

_model = None
_lock = threading.Lock()
_helpers = None


def _load_helpers():
    global _helpers
    if _helpers is not None:
        return _helpers
    if VOICE_SERVER_DIR not in sys.path:
        sys.path.insert(0, VOICE_SERVER_DIR)
    from vox_ser.audio_io import _agc_pcm, _asr_normalize, _trim_silence  # noqa: WPS433

    _helpers = (_asr_normalize, _trim_silence, _agc_pcm)
    return _helpers


def _init_model() -> bool:
    global _model
    if _model is not None:
        return True
    try:
        import sherpa_onnx

        lang = os.environ.get("VOX_ASR_LANG", "zh")
        kw = dict(
            model=os.path.join(SENSE_MODEL_DIR, "model.int8.onnx"),
            tokens=os.path.join(SENSE_MODEL_DIR, "tokens.txt"),
            num_threads=4,
            use_itn=True,
            debug=False,
        )
        try:
            _model = sherpa_onnx.OfflineRecognizer.from_sense_voice(language=lang, **kw)
        except TypeError:
            _model = sherpa_onnx.OfflineRecognizer.from_sense_voice(**kw)
        log.info("SenseVoice int8 loaded (lang=%s)", lang)
        return True
    except Exception as exc:  # pragma: no cover - hardware dependent
        log.error("SenseVoice init failed: %s", exc)
        _model = None
        return False


def _dump(pcm: bytes, tag: str) -> None:
    global _dump_count
    if os.environ.get("VOX_ASR_DUMP") != "1" or _dump_count >= 40:
        return
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        fn = os.path.join(DUMP_DIR, "%d_%s.raw" % (int(time.time() * 1000), tag))
        with open(fn, "wb") as f:
            f.write(pcm)
        _dump_count += 1
    except Exception:
        pass


def _stats(pcm: bytes) -> str:
    try:
        import numpy as np

        x = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
        if x.size == 0:
            return "empty"
        return "peak=%d rms=%.0f" % (int(abs(x).max()), float(np.sqrt((x.astype(np.float64) ** 2).mean())))
    except Exception:
        return "?"


def transcribe_blocking(pcm16k: bytes) -> str:
    if not _init_model():
        return ""
    if len(pcm16k) < MIN_UTTER_BYTES:
        return ""
    import numpy as np

    norm, trim, agc = _load_helpers()
    raw_stat = _stats(pcm16k)

    def _peak(b: bytes) -> int:
        try:
            arr = np.frombuffer(b[: len(b) - (len(b) % 2)], dtype=np.int16)
            return int(abs(arr).max()) if arr.size else 0
        except Exception:
            return 0

    _raw_peak = _peak(pcm16k)
    _pcm_n = agc(trim(norm(pcm16k)))
    # 2026-11-22：实测 norm 后 peak 经常被顶到 32000（AGC 二次放大后削波），
    # 而 raw 只有 6000 —— 削波把"图书馆不去了"听成"补条电池机"。规则：
    #   · raw 已经够响（>=24000，本来就会削）→ 直接用 raw，不再放大
    #   · norm 后顶到 24000 以上而 raw 没顶 → 退回 trim(raw)，放弃二次增益
    #   · 其它情况沿用 norm + trim + agc
    if _raw_peak >= 24000:
        pcm = trim(pcm16k)
        log.info("ASR_NOSKIP reason=raw_loud peak=%d", _raw_peak)
    elif _peak(_pcm_n) >= 24000:
        pcm = trim(pcm16k)
        log.info("ASR_NOSKIP reason=norm_clip peak=%d raw_peak=%d", _peak(_pcm_n), _raw_peak)
    else:
        pcm = _pcm_n
    _dump(pcm16k, "raw")
    _dump(pcm, "norm")

    def _decode(buf: bytes) -> str:
        x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        with _lock:
            st = _model.create_stream()
            st.accept_waveform(AUDIO_SAMPLE_RATE, x)
            _model.decode_stream(st)
            return st.result.text or ""

    # 2026-11-22 双解码：归一化版本通常是主力；
    # 但风噪/爆音下归一化会把噪声抬成"字"，这时裸音频反而更准。
    # 两个都解，按"有效字符数 + 重复惩罚"取分高者，并把两者都写日志便于后续调参。
    result = _decode(pcm)
    result_raw = _decode(trim(pcm16k))

    def _score(t: str) -> float:
        core = re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]", "", t or "")
        if not core:
            return -1.0
        rep = 0
        for i in range(1, len(core)):
            if core[i] == core[i - 1]:
                rep += 1
        return len(core) - rep * 1.5

    if _score(result_raw) > _score(result) + 1.0:
        log.info("ASR_PICK raw text=%r (norm=%r)", result_raw[:30], result[:30])
        result = result_raw
    text = re.sub(r"<\|[^|]*\|>", "", result).strip()
    # 旧链路 min_valid_chars=2：只剩标点/单字的不算一句话
    if len(re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]", "", text)) < MIN_VALID_CHARS:
        log.info("ASR_DROP text=%r %s -> norm(%s)", text[:20], raw_stat, _stats(pcm))
        return ""
    log.info("ASR_OK text=%r %s -> norm(%s)", text[:40], raw_stat, _stats(pcm))
    return text


async def sensevoice_asr(pcm16k: bytes) -> str:
    """Async callable matching vox_v2.audio.AsrCallable."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, transcribe_blocking, pcm16k)
