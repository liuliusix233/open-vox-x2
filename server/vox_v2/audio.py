"""Binary wire framing, streaming VAD state, and pluggable ASR/TTS adapters."""

from __future__ import annotations

import array
import json
import math
import os
import struct
from collections import deque
from typing import Any, AsyncIterator, Awaitable, Callable

def normalize_for_tts(text: str) -> str:
    """Number/unit normalization reusing the legacy helper (weText pipeline).

    Falls back to the raw string so TTS never breaks if the helper is missing.
    """
    try:
        import sys

        legacy = os.environ.get(
            "VOX_LEGACY_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_server"),
        )
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from vox_ser.audio_io import normalize_for_tts as _norm

        return _norm(text)
    except Exception:
        return text


FT_AUDIO_UP = 0x01
FT_AUDIO_DOWN = 0x02
FT_VISION_UP = 0x03
FT_CONTROL = 0x04

DTX_HEARTBEAT = b"\x00\x00"


def pack_frame(frame_type: int, epoch_id: int, payload: bytes) -> bytes:
    return struct.pack(">BI", frame_type, epoch_id) + payload


def unpack_frame(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < 5:
        raise ValueError(f"Frame too short: {len(data)} bytes")
    ftype, epoch = struct.unpack(">BI", data[:5])
    return ftype, epoch, data[5:]


def pack_control(epoch_id: int, msg_type: str, **fields: Any) -> bytes:
    obj = {"type": msg_type.upper(), "epoch_id": epoch_id, **fields}
    return pack_frame(FT_CONTROL, epoch_id, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def unpack_vision_payload(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 2:
        raise ValueError("Vision payload too short")
    (meta_len,) = struct.unpack(">H", payload[:2])
    if len(payload) < 2 + meta_len:
        raise ValueError("Vision payload truncated")
    meta = json.loads(payload[2 : 2 + meta_len].decode("utf-8"))
    return meta, payload[2 + meta_len :]


def pcm_rms(pcm16le: bytes) -> float:
    if len(pcm16le) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm16le[: len(pcm16le) - (len(pcm16le) % 2)])
    if not samples:
        return 0.0
    acc = sum(s * s for s in samples)
    return math.sqrt(acc / len(samples))


def apply_fade_in(pcm16le: bytes, fade_samples: int = 192) -> bytes:
    """Linear 0.0 -> 1.0 fade-in (8ms @ 24kHz) to eliminate click/pop."""
    if len(pcm16le) < 2:
        return pcm16le
    samples = array.array("h")
    samples.frombytes(pcm16le[: len(pcm16le) - (len(pcm16le) % 2)])
    n = min(fade_samples, len(samples))
    for i in range(n):
        samples[i] = int(samples[i] * (i / float(fade_samples)))
    return samples.tobytes()


class StreamingVad:
    """20ms frame VAD with 200ms pre-roll and 900ms silence endpointing.

    2026-11-22: 断句阈值放宽到 45 帧(900ms)。420ms 会把「先说半句再停顿」的话拦腰截断，
    ASR 只拿到前半句 → 表现为答非所问。放宽后代价是反应慢约 0.5s，换准确度。
    """

    def __init__(
        self,
        speech_onset_frames: int = 3,    # 60ms
        end_silence_frames: int = int(os.environ.get("VOX_VAD_END_FRAMES", "30")),
        # 600ms（原 45 = 900ms）；VOX_VAD_END_FRAMES 可覆盖
        max_utter_frames: int = 750,     # 15s cap
        pre_roll_frames: int = 10,       # 200ms
    ) -> None:
        self.speech_onset_frames = speech_onset_frames
        self.end_silence_frames = end_silence_frames
        self.max_utter_frames = max_utter_frames
        self.pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        self.noise_floor: float = 220.0
        self.in_speech: bool = False
        self._voiced_streak: int = 0
        self._silence_streak: int = 0
        self._buf: bytearray = bytearray()
        self._utter_frames: int = 0

    def reset(self) -> None:
        self.in_speech = False
        self._voiced_streak = 0
        self._silence_streak = 0
        self._buf.clear()
        self._utter_frames = 0

    def feed(self, frame: bytes) -> tuple[str, bytes | None]:
        """Returns (event, utterance_pcm_or_None). Events: 'none' | 'onset' | 'utterance'."""
        is_dtx = len(frame) <= 2
        rms = 0.0 if is_dtx else pcm_rms(frame)
        # 2026-11-22：阈值从 260/2.1x 放宽到 180/1.6x —— 12x 增益下辅音起音常被
        # 判成静音，丢了第一个字（"导航去…"听成"航去…"）。放宽后端点由 900ms 兜底。
        threshold = max(180.0, self.noise_floor * 1.6)
        is_voice = (not is_dtx) and (rms > threshold)

        if not self.in_speech:
            if not is_dtx:
                self.pre_roll.append(frame)
                if not is_voice:
                    self.noise_floor = self.noise_floor * 0.98 + rms * 0.02
            if is_voice:
                self._voiced_streak += 1
                if self._voiced_streak >= self.speech_onset_frames:
                    self.in_speech = True
                    self._silence_streak = 0
                    self._buf.clear()
                    for p in self.pre_roll:
                        self._buf.extend(p)
                    self._utter_frames = len(self.pre_roll)
                    return "onset", None
            else:
                self._voiced_streak = 0
            return "none", None

        # Currently inside speech
        if not is_dtx:
            self._buf.extend(frame)
            self._utter_frames += 1

        if is_voice:
            self._silence_streak = 0
        else:
            # 2026-11-22: DTX 心跳 400ms 一次，步进从 +20 降到 +10 —— 否则两三个心跳
            # 就把 900ms 窗口吃光，等于没放宽。真静音（有帧但能量低）仍然逐帧 +1。
            self._silence_streak += 10 if is_dtx else 1

        if self._silence_streak >= self.end_silence_frames or self._utter_frames >= self.max_utter_frames:
            pcm = bytes(self._buf)
            self.reset()
            return "utterance", pcm
        return "none", None


AsrCallable = Callable[[bytes], Awaitable[str]]
TtsCallable = Callable[[str], AsyncIterator[bytes]]


async def default_http_asr(session: Any, asr_url: str, pcm16k: bytes) -> str:
    """Adapts to a local SenseVoice HTTP endpoint accepting raw 16kHz PCM16LE."""
    if not asr_url:
        raise RuntimeError("VOX_ASR_URL is not configured; bind your local SenseVoice callable in app.py")
    async with session.post(
        asr_url, data=pcm16k, headers={"Content-Type": "application/octet-stream"}
    ) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "json" in ctype:
            obj = await resp.json()
            return str(obj.get("text", "")).strip()
        return (await resp.text()).strip()


async def cosyvoice2_tts_stream(
    session: Any,
    tts_url: str,
    spk_id: str,
    text: str,
    chunk_bytes: int = 3840,  # 80ms @ 24kHz PCM16LE
) -> AsyncIterator[bytes]:
    """Streams 24kHz PCM16LE chunks from CosyVoice2 (:50002).

    2026-11-22: this box's CosyVoice2 exposes /inference_spk (cached speaker
    embedding, raw s16le/24k/mono). /inference_sft answers 200 with zero bytes
    here, so it must not be used.
    """
    url = tts_url.rstrip("/") + os.environ.get("VOX_CV_TTS_PATH", "/inference_spk")
    payload = {"tts_text": normalize_for_tts(text), "spk_id": spk_id}
    first = True
    buf = bytearray()
    async with session.post(url, data=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content.iter_chunked(chunk_bytes):
            if not raw:
                continue
            buf.extend(raw)
            while len(buf) >= chunk_bytes:
                chunk = bytes(buf[:chunk_bytes])
                del buf[:chunk_bytes]
                if first:
                    chunk = apply_fade_in(chunk)
                    first = False
                yield chunk
        if buf:
            rem = bytes(buf[: len(buf) - (len(buf) % 2)])
            if rem:
                yield apply_fade_in(rem) if first else rem


async def openai_tts_stream(
    session: Any,
    base_url: str,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    chunk_bytes: int = 3840,  # 80ms @ 24kHz PCM16LE
) -> AsyncIterator[bytes]:
    """标准 OpenAI 兼容 /v1/audio/speech 流式适配器（开箱 5 分钟可用）。

    base_url 例：https://api.openai.com/v1  或  http://127.0.0.1:9880/v1
    返回 24kHz PCM16LE 单声道；若服务端只给 mp3/wav，本函数按响应 Content-Type
    自动转码（wav→PCM 直读，mp3 需系统有 ffmpeg）。
    """
    url = base_url.rstrip("/") + "/audio/speech"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    payload = {
        "model": model,
        "voice": voice,
        "input": normalize_for_tts(text),
        "response_format": "pcm",
        "speed": 1.0,
    }
    first = True
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pcm" in ctype or "octet-stream" in ctype or not ctype:
            buf = bytearray()
            async for raw in resp.content.iter_chunked(chunk_bytes):
                if not raw:
                    continue
                buf.extend(raw)
                while len(buf) >= chunk_bytes:
                    chunk = bytes(buf[:chunk_bytes])
                    del buf[:chunk_bytes]
                    if first:
                        chunk = apply_fade_in(chunk)
                        first = False
                    yield chunk
            if buf:
                rem = bytes(buf[: len(buf) - (len(buf) % 2)])
                if rem:
                    yield apply_fade_in(rem) if first else rem
            return
        # 非 PCM：整段收下再转码（延迟换兼容性）
        body = await resp.read()
    if "wav" in ctype or "wave" in ctype:
        pcm = _wav_to_pcm24k(body)
    else:
        pcm = _audio_to_pcm24k_ffmpeg(body)
    for i in range(0, len(pcm), chunk_bytes):
        chunk = pcm[i: i + chunk_bytes]
        if first:
            chunk = apply_fade_in(chunk)
            first = False
        yield chunk


def _audio_to_pcm24k_ffmpeg(blob: bytes) -> bytes:
    """mp3/ogg → 24kHz PCM16LE 单声道（依赖系统 ffmpeg）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", "pipe:0",
             "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", "pipe:1"],
            input=blob, capture_output=True, timeout=30, check=True,
        )
        return out.stdout
    except Exception as exc:
        raise RuntimeError(f"TTS 返回非 PCM 且 ffmpeg 转码失败：{exc}") from exc


def _wav_to_pcm24k(wav_bytes: bytes) -> bytes:
    """WAV(任意采样率/单声道) → 24kHz PCM16LE。GPT-SoVITS 输出 32k，这里做线性重采样。"""
    import io as _io
    import wave as _wave
    with _wave.open(_io.BytesIO(wav_bytes), 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    import array as _arr
    samples = _arr.array('h')
    samples.frombytes(raw[: len(raw) - (len(raw) % (2 * ch))])
    if ch > 1:
        samples = _arr.array('h', samples[0::ch])
    if sr == 24000:
        return samples.tobytes()
    n_out = int(len(samples) * 24000 / sr)
    out = _arr.array('h')
    step = sr / 24000.0
    for i in range(n_out):
        x = i * step
        i0 = int(x)
        i1 = min(i0 + 1, len(samples) - 1)
        frac = x - i0
        v = samples[i0] * (1 - frac) + samples[i1] * frac
        out.append(int(max(-32768, min(32767, v))))
    return out.tobytes()


async def gpt_sovits_tts_stream(
    session: Any,
    tts_url: str,
    ref_wav: str,
    prompt_text: str,
    text: str,
    chunk_bytes: int = 3840,
) -> AsyncIterator[bytes]:
    """GPT-SoVITS v2final zero-shot：POST /tts 拿整段 WAV，转 24k 后按块下发。"""
    payload = {
        "text": normalize_for_tts(text),
        "text_lang": "zh",
        "ref_audio_path": ref_wav,
        "prompt_text": prompt_text,
        "prompt_lang": "zh",
        "text_split_method": "cut5",
        "parallel_infer": True,
    }
    url = tts_url.rstrip("/") + "/tts"
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.read()
    pcm = _wav_to_pcm24k(data)
    first = True
    for off in range(0, len(pcm), chunk_bytes):
        chunk = pcm[off: off + chunk_bytes]
        if not chunk:
            continue
        if first:
            chunk = apply_fade_in(chunk)
            first = False
        yield chunk

