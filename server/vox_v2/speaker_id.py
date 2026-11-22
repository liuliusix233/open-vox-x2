"""speaker_id.py — Lightweight local speaker identification (sherpa-onnx) + SocialTracker.

Identifies who is speaking ("owner" / "partner" / "bystander") in ~15ms on CPU using a
sherpa-onnx speaker embedding ONNX model (e.g. 3D-Speaker / CAM++ / WeSpeaker),
and tracks multi-speaker social context ([SOCIAL_SITUATION]) so Vox understands
ongoing conversations between people without awkwardly interrupting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
import wave
from collections import deque
from typing import Any

import numpy as np

log = logging.getLogger("vox_v2.speaker")

DEFAULT_SPK_MODEL = os.environ.get(
    "VOX_SPK_MODEL",
    os.path.expanduser(
        "~/.openclaw/workspace/jarvis/voice_server/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    ),
)
DEFAULT_VOICEPRINTS = os.environ.get(
    "VOX_VOICEPRINTS_PATH",
    os.path.expanduser("~/.local/share/open-vox-x2/voiceprints.json"),
)
DEFAULT_THRESHOLD = float(os.environ.get("VOX_SPK_THRESHOLD", "0.52"))


def _l2_normalize(vec: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return [0.0] * len(arr)
    return (arr / norm).tolist()


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


class SpeakerIdentifier:
    """Thread-safe sherpa-onnx speaker embedding extractor and voiceprint matcher."""

    def __init__(
        self,
        model_path: str = DEFAULT_SPK_MODEL,
        voiceprints_path: str = DEFAULT_VOICEPRINTS,
        threshold: float = DEFAULT_THRESHOLD,
        embed_fn: Any = None,  # Optional callable(pcm16k: bytes) -> list[float] for tests
    ) -> None:
        self.model_path = model_path
        self.voiceprints_path = voiceprints_path
        self.threshold = threshold
        self._custom_embed_fn = embed_fn
        self._lock = threading.RLock()
        self._extractor: Any = None
        self._init_attempted = False
        self.voiceprints: dict[str, list[float]] = {}
        self.load_voiceprints()

    def load_voiceprints(self) -> None:
        with self._lock:
            if not self.voiceprints_path or not os.path.isfile(self.voiceprints_path):
                self.voiceprints = {}
                return
            try:
                with open(self.voiceprints_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.voiceprints = {
                    str(k): _l2_normalize(v) for k, v in data.items() if isinstance(v, list) and v
                }
            except Exception as exc:
                log.warning("Failed to load voiceprints: %s", exc)
                self.voiceprints = {}

    def save_voiceprints(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.voiceprints_path)), exist_ok=True)
            tmp = self.voiceprints_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.voiceprints, f, ensure_ascii=False)
            os.replace(tmp, self.voiceprints_path)

    def _ensure_extractor(self) -> bool:
        if self._custom_embed_fn is not None:
            return True
        if self._extractor is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        if not self.model_path or not os.path.isfile(self.model_path):
            log.info("Speaker ONNX model not found at %s; speaker ID running in passive mode", self.model_path)
            return False
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model_path,
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            log.info("SpeakerEmbeddingExtractor loaded: %s", os.path.basename(self.model_path))
            return True
        except Exception as exc:
            log.warning("SpeakerEmbeddingExtractor init failed: %s", exc)
            return False

    def extract_embedding(self, pcm16k: bytes) -> list[float] | None:
        if len(pcm16k) < 6400:  # < 0.2s @ 16kHz
            return None
        if self._custom_embed_fn is not None:
            vec = self._custom_embed_fn(pcm16k)
            return _l2_normalize(vec) if vec else None
        with self._lock:
            if not self._ensure_extractor():
                return None
            samples = np.frombuffer(pcm16k[: len(pcm16k) - (len(pcm16k) % 2)], dtype=np.int16).astype(np.float32) / 32768.0
            stream = self._extractor.create_stream()
            stream.accept_waveform(16000, samples)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                return None
            emb = self._extractor.compute(stream)
            return _l2_normalize(emb)

    def enroll_pcm(self, speaker_name: str, pcm_clips: list[bytes]) -> bool:
        embs = [e for c in pcm_clips if (e := self.extract_embedding(c)) is not None]
        if not embs:
            return False
        avg = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
        with self._lock:
            self.voiceprints[speaker_name.strip()] = _l2_normalize(avg)
            self.save_voiceprints()
        return True

    def identify_pcm(self, pcm16k: bytes) -> tuple[str, float]:
        """Return (speaker_label, similarity). Falls back to ('bystander', sim) or ('unknown', 0.0)."""
        with self._lock:
            if not self.voiceprints:
                return "unknown", 0.0
        emb = self.extract_embedding(pcm16k)
        if not emb:
            return "unknown", 0.0
        best_name, best_sim = "bystander", -1.0
        with self._lock:
            for name, ref in self.voiceprints.items():
                s = cosine_sim(emb, ref)
                if s > best_sim:
                    best_sim = s
                    best_name = name
        if best_sim >= self.threshold:
            return best_name, round(best_sim, 3)
        return "bystander", round(max(0.0, best_sim), 3)


class SocialTracker:
    """Tracks multi-speaker chatter and maintains a live [SOCIAL_SITUATION] summary."""

    def __init__(self, owner_label: str = "owner", window_s: float = 45.0) -> None:
        self.owner_label = owner_label
        self.window_s = window_s
        self._lock = threading.RLock()
        self._recent: deque[tuple[float, str, str]] = deque(maxlen=20)
        self.current_situation: str = ""
        self.situation_ts: float = 0.0

    def note_utterance(self, speaker: str, text: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._recent.append((now, speaker or "unknown", text.strip()[:120]))

    def update_situation(self, note: str, now: float | None = None) -> None:
        clean = (note or "").strip()
        if not clean:
            return
        now = time.time() if now is None else now
        with self._lock:
            self.current_situation = clean[:160]
            self.situation_ts = now

    def is_multi_speaker_active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            speakers = {spk for ts, spk, _ in self._recent if ts >= cutoff and spk != "unknown"}
            has_non_owner = any(spk != self.owner_label for spk in speakers)
            return len(speakers) >= 2 or has_non_owner

    def format_context_block(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        lines: list[str] = []
        with self._lock:
            if self.current_situation and (now - self.situation_ts) <= 300.0:
                lines.append(f"[CURRENT_SITUATION] {self.current_situation}")
            if self.is_multi_speaker_active(now):
                lines.append(
                    "[MULTI_SPEAKER_ACTIVE] 现场有他人在场，两人可能在互相聊天而非对你说话。"
                    "请先判断这句话是不是在对你（Vox）说：如果不是对你说的，必须直接调用 "
                    "do_nothing(reason='他人互聊', situation_note='用一句话概括现场正在聊什么') 保持安静不插嘴。"
                )
        return "\n".join(lines)


def _read_audio_file(path: str) -> bytes:
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as wf:
            return wf.readframes(wf.getnframes())
    with open(path, "rb") as f:
        return f.read()


def main() -> None:
    ap = argparse.ArgumentParser(description="Enroll or test Vox speaker voiceprints")
    ap.add_argument("--enroll", help="Speaker label to enroll, e.g. owner or partner")
    ap.add_argument("--test", help="Audio file (.wav or 16k .raw) to identify")
    ap.add_argument("--model", default=DEFAULT_SPK_MODEL)
    ap.add_argument("--db", default=DEFAULT_VOICEPRINTS)
    ap.add_argument("files", nargs="*", help="Audio files (.wav or 16k .raw) for enrollment")
    args = ap.parse_args()

    sid = SpeakerIdentifier(model_path=args.model, voiceprints_path=args.db)
    if args.enroll:
        clips = [_read_audio_file(p) for p in args.files]
        ok = sid.enroll_pcm(args.enroll, clips)
        print(f"enroll {args.enroll}: ok={ok} (voiceprints={list(sid.voiceprints.keys())})")
    elif args.test:
        spk, sim = sid.identify_pcm(_read_audio_file(args.test))
        print(f"identify {args.test}: speaker={spk} sim={sim}")
    else:
        print(f"voiceprints={list(sid.voiceprints.keys())} model_exists={os.path.isfile(args.model)}")


if __name__ == "__main__":
    main()
