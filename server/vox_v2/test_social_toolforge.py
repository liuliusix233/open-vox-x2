"""Smoke tests for speaker_id.py (SpeakerIdentifier + SocialTracker) and tool_forge.py."""

from __future__ import annotations

import os
import tempfile
import time

try:
    from .speaker_id import SocialTracker, SpeakerIdentifier
    from .tool_forge import ToolForge
except ImportError:  # 直接在本目录下 python3 test_social_toolforge.py 时
    from speaker_id import SocialTracker, SpeakerIdentifier
    from tool_forge import ToolForge


def test_speaker_and_social() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_spk_") as tmp:
        vp_path = os.path.join(tmp, "voiceprints.json")

        def fake_embed(pcm: bytes) -> list[float]:
            if pcm.startswith(b"OWNER"):
                return [1.0, 0.0, 0.0, 0.0]
            if pcm.startswith(b"WIFE"):
                return [0.0, 1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0]

        sid = SpeakerIdentifier(voiceprints_path=vp_path, embed_fn=fake_embed, threshold=0.52)
        assert sid.enroll_pcm("owner", [b"OWNER" + b"\x00" * 7000])
        assert sid.enroll_pcm("partner", [b"WIFE" + b"\x00" * 7000])

        spk1, sim1 = sid.identify_pcm(b"OWNER" + b"\x00" * 7000)
        spk2, sim2 = sid.identify_pcm(b"WIFE" + b"\x00" * 7000)
        spk3, sim3 = sid.identify_pcm(b"OTHER" + b"\x00" * 7000)
        assert spk1 == "owner" and sim1 > 0.99
        assert spk2 == "partner" and sim2 > 0.99
        assert spk3 == "bystander" and sim3 < 0.1

        st = SocialTracker(owner_label="owner")
        t0 = time.time()
        st.note_utterance("owner", "早知道吃螺蛳粉了", now=t0)
        assert not st.is_multi_speaker_active(now=t0 + 1)
        st.note_utterance("partner", "那我们再点一份吧", now=t0 + 2)
        assert st.is_multi_speaker_active(now=t0 + 3)

        st.update_situation("owner和partner在聊晚饭吃螺蛳粉和炸酱面，未对Vox说话", now=t0 + 4)
        blk = st.format_context_block(now=t0 + 5)
        assert "[CURRENT_SITUATION]" in blk and "螺蛳粉" in blk
        assert "[MULTI_SPEAKER_ACTIVE]" in blk


def test_tool_forge() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_tf_") as tmp:
        tf = ToolForge(tools_dir=os.path.join(tmp, "custom_tools"))

        # 1. Security blocks dangerous code & traversal
        bad1 = tf.save_tool("../evil", "bad", "def run(args): return {}")
        assert not bad1["ok"]
        bad2 = tf.save_tool("bad_shell", "bad", "import subprocess\ndef run(args): return {}")
        assert not bad2["ok"] and any("subprocess" in v for v in bad2["violations"])
        bad3 = tf.save_tool("bad_eval", "bad", "def run(args): return eval('1+1')")
        assert not bad3["ok"]

        # 2. Save & run a real safe mini-tool
        code = (
            "import random\n"
            "def run(args):\n"
            "    opts = args.get('options') or ['螺蛳粉', '炸酱面']\n"
            "    rng = random.Random(args.get('seed', 42))\n"
            "    return {'choice': rng.choice(opts), 'count': len(opts)}\n"
        )
        ok_save = tf.exec_tool(
            "save_custom_tool",
            {"name": "dinner_picker", "description": "随机决定晚饭吃什么", "code": code},
        )
        assert ok_save["ok"]

        listed = tf.exec_tool("list_custom_tools", {})
        assert listed["count"] == 1 and listed["tools"][0]["name"] == "dinner_picker"

        run_res = tf.exec_tool(
            "run_custom_tool",
            {"name": "dinner_picker", "args": {"options": ["螺蛳粉", "炸酱面"], "seed": 7}},
        )
        assert run_res["ok"] and run_res["result"]["choice"] in ("螺蛳粉", "炸酱面")


def main() -> None:
    test_speaker_and_social()
    test_tool_forge()
    print("ALL SOCIAL & TOOLFORGE SMOKE TESTS PASSED!")


if __name__ == "__main__":
    main()
