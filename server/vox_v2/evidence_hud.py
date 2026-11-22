"""Evidence photo HUD push helper for vox_v2.

Encodes verified archived frames (from inspect_past_frames / search_memory /
browse_vision_timeline) into compact 320x240 JPEG thumbnails carried directly
inside the existing FT_CONTROL HUD_PATCH message for RayNeo X2 glasses.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

# Pillow 9.0.x 没有 Image.Resampling（9.1 才加），这里做兼容
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS


def is_allowed_frame_path(path: str, allowed_roots: list[str]) -> bool:
    """Ensure frame path exists and resolves inside an allowed archive directory."""
    if not path:
        return False
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    if not os.path.isfile(real):
        return False
    for root in allowed_roots:
        if not root:
            continue
        real_root = os.path.realpath(root)
        if real == real_root or real.startswith(real_root + os.sep):
            return True
    return False


def encode_hud_thumbnail(
    frame_path: str,
    allowed_roots: list[str],
    max_w: int = 320,
    max_h: int = 240,
    quality: int = 72,
    max_bytes: int = 36_000,
) -> str | None:
    """Resize and JPEG-encode an archived photo for RayNeo X2 waveguide HUD."""
    if not is_allowed_frame_path(frame_path, allowed_roots):
        return None
    try:
        with Image.open(frame_path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w, max_h), _LANCZOS)
            for q in (quality, 58, 45):
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                raw = buf.getvalue()
                if len(raw) <= max_bytes or q == 45:
                    return base64.b64encode(raw).decode("ascii")
    except Exception:
        return None
    return None


def build_evidence_hud_patch(
    header: str,
    lines: list[str],
    frame_items: list[tuple[str, str]],
    allowed_roots: list[str],
    ttl_ms: int = 12_000,
    max_images: int = 2,
) -> dict[str, Any]:
    """Build a backward-compatible HUD_PATCH payload with up to 2 evidence thumbnails.

    Args:
        header: Short card title, e.g. "MEMORY · Evidence" or "PHOTO · 11-19 10:30".
        lines: Up to 3 concise text lines for the evidence card.
        frame_items: Ordered list of (frame_path, label) pairs; duplicates removed.
        allowed_roots: Realpath whitelist (e.g. [cfg.frame_dir, voice_frames_root]).
        ttl_ms: Display duration on RayNeo X2 HUD (default 12s).
        max_images: Maximum number of thumbnails to attach (default 2).
    """
    seen: set[str] = set()
    images: list[dict[str, str]] = []
    for fpath, label in frame_items:
        if len(images) >= max_images:
            break
        if not fpath:
            continue
        real = os.path.realpath(fpath)
        if real in seen:
            continue
        b64 = encode_hud_thumbnail(real, allowed_roots)
        if not b64:
            continue
        seen.add(real)
        images.append({"b64": b64, "label": (label or "").strip()[:48]})

    patch: dict[str, Any] = {
        "type": "HUD_PATCH",
        "level": 2,
        "slot": "right_evidence",
        "accent": "isac_cyan",
        "header": header[:36],
        "lines": [str(x)[:48] for x in lines[:3]],
        "ttl_ms": int(ttl_ms),
    }
    if images:
        patch["images"] = images
        # Convenience fallback field for single-image consumers
        patch["image_b64"] = images[0]["b64"]
        patch["image_label"] = images[0]["label"]
    return patch


def merge_turn_hud_patches(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the strongest right_evidence patch per turn (inspected photo beats text-only)."""
    best_evidence: dict[str, Any] | None = None
    others: list[dict[str, Any]] = []
    for p in patches:
        if p.get("slot") != "right_evidence":
            others.append(p)
            continue
        if best_evidence is None:
            best_evidence = p
        elif p.get("images") and not best_evidence.get("images"):
            best_evidence = p
        elif p.get("images") and best_evidence.get("images"):
            # Later patch (e.g. inspect_past_frames) supersedes earlier search_memory patch
            best_evidence = p
    if best_evidence is not None:
        others.append(best_evidence)
    return others
