"""Smoke tests for server/evidence_hud.py."""

from __future__ import annotations

import base64
import json
import os
import tempfile

from PIL import Image

try:
    from .evidence_hud import (
        build_evidence_hud_patch,
        encode_hud_thumbnail,
        is_allowed_frame_path,
        merge_turn_hud_patches,
    )
except ImportError:  # python3 test_evidence_hud.py
    from evidence_hud import (
        build_evidence_hud_patch,
        encode_hud_thumbnail,
        is_allowed_frame_path,
        merge_turn_hud_patches,
    )


def run_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_ev_test_") as tmp:
        allowed_dir = os.path.join(tmp, "voice-frames", "2026-11-19", "voxx2")
        outside_dir = os.path.join(tmp, "outside")
        os.makedirs(allowed_dir, exist_ok=True)
        os.makedirs(outside_dir, exist_ok=True)

        img1 = os.path.join(allowed_dir, "102900_001.jpg")
        img2 = os.path.join(allowed_dir, "103005_002.jpg")
        img_out = os.path.join(outside_dir, "secret.jpg")

        Image.new("RGB", (1280, 960), color=(40, 120, 180)).save(img1, format="JPEG", quality=90)
        Image.new("RGB", (1280, 960), color=(180, 90, 60)).save(img2, format="JPEG", quality=90)
        Image.new("RGB", (640, 480), color=(255, 0, 0)).save(img_out, format="JPEG", quality=90)

        roots = [os.path.join(tmp, "voice-frames")]

        # 1. Security boundary
        assert is_allowed_frame_path(img1, roots)
        assert not is_allowed_frame_path(img_out, roots)
        assert encode_hud_thumbnail(img_out, roots) is None

        # 2. Thumbnail size and JPEG validity
        b64 = encode_hud_thumbnail(img1, roots)
        assert b64 is not None
        raw = base64.b64decode(b64)
        assert raw[:2] == b"\xff\xd8"  # JPEG SOI
        assert len(raw) <= 36_000

        # 3. Multi-image patch + deduplication
        patch = build_evidence_hud_patch(
            header="MEMORY · Evidence",
            lines=["11-19 10:30 CULTURAL CENTER 玻璃门", "11-19 10:31 进入大厅阅览区"],
            frame_items=[
                (img1, "11-19 10:29 户外广场"),
                (img1, "duplicate should be ignored"),
                (img2, "11-19 10:30 图书馆门口"),
                (img_out, "outside should be ignored"),
            ],
            allowed_roots=roots,
        )
        assert len(patch["images"]) == 2
        assert patch["image_b64"] == patch["images"][0]["b64"]
        assert patch["ttl_ms"] == 12_000
        # Ensure JSON serializable
        encoded = json.dumps(patch, ensure_ascii=False)
        assert len(encoded) < 120_000

        # 4. Merge priority: inspected photo patch supersedes earlier text-only patch
        p_text = {"type": "HUD_PATCH", "slot": "right_evidence", "header": "MEMORY", "lines": ["text only"]}
        merged = merge_turn_hud_patches([p_text, patch])
        assert len(merged) == 1
        assert len(merged[0]["images"]) == 2

    print("ALL evidence_hud SMOKE TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
