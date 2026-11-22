"""Automated smoke tests for Phase 2+3 (amap_tools + agency_loop_v2)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "vox_phase1_agency")
    ),
)

from .agency_loop_v2 import AGENCY_CONTROL_TOOLS, AgencyLoopV2
from .amap_tools import (
    AMAP_TOOLS_SCHEMA,
    AmapNavEngine,
    bearing_deg,
    haversine_m,
    wgs84_to_gcj02,
)
from .hormones_v2 import HormonesV2
from .todo_goals import TodoGoalStore


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeHttp:
    def get(self, url: str, params: dict, timeout: int = 10):
        if "/weather/weatherInfo" in url:
            if params.get("extensions") == "base":
                return FakeResponse(
                    {
                        "lives": [
                            {
                                "city": "Shanghai",
                                "weather": "多云",
                                "temperature": "11",
                                "winddirection": "北",
                                "windpower": "4",
                                "humidity": "58",
                                "reporttime": "2026-11-22 18:00:00",
                            }
                        ]
                    }
                )
            return FakeResponse(
                {
                    "forecasts": [
                        {
                            "casts": [
                                {"dayweather": "多云", "daytemp": "14", "nightweather": "晴", "nighttemp": "6"}
                            ]
                        }
                    ]
                }
            )
        if "/place/text" in url:
            return FakeResponse(
                {"pois": [{"name": "Shanghai Library", "location": "120.312500,31.548000"}]}
            )
        if "/direction/walking" in url:
            return FakeResponse(
                {
                    "route": {
                        "paths": [
                            {
                                "distance": "1240",
                                "steps": [
                                    {
                                        "instruction": "沿钟书路向东步行85米右转",
                                        "action": "右转",
                                        "distance": "85",
                                        "polyline": "120.300000,31.548000;120.300900,31.548000",
                                    },
                                    {
                                        "instruction": "继续步行1155米到达Shanghai Library",
                                        "action": "直行",
                                        "distance": "1155",
                                        "polyline": "120.300900,31.548000;120.312500,31.548000",
                                    },
                                ],
                            }
                        ]
                    }
                }
            )
        return FakeResponse({})


def test_geo_math() -> None:
    # Wuxi WGS-84 -> GCJ-02 shift should be ~300-650m
    lat, lon = 31.5480, 120.3000
    glat, glon = wgs84_to_gcj02(lat, lon)
    shift = haversine_m(lat, lon, glat, glon)
    assert 200.0 < shift < 800.0, f"Unexpected GCJ02 shift: {shift}"

    # Cardinal bearings
    assert abs(bearing_deg(31.5, 120.3, 31.6, 120.3) - 0.0) < 0.5
    assert abs(bearing_deg(31.5, 120.3, 31.5, 120.4) - 90.0) < 0.5
    assert abs(bearing_deg(31.5, 120.3, 31.4, 120.3) - 180.0) < 0.5
    assert abs(bearing_deg(31.5, 120.3, 31.5, 120.2) - 270.0) < 0.5


async def test_amap_nav_and_weather() -> None:
    eng = AmapNavEngine(api_key="test", http_session=FakeHttp())
    w = await eng.get_weather("voxx2", "Shanghai")
    assert w["ok"] and w["temp_c"] == "11" and "多云" in w["weather"]

    res = await eng.start_navigation(
        "voxx2", "Shanghai Library", origin_lat_gcj=31.548000, origin_lon_gcj=120.300000
    )
    assert res["ok"] and res["dest"] == "Shanghai Library"
    patch = res["nav_patch"]
    assert patch["active"] is True
    assert len(patch["waypoints"]) == 2
    assert patch["waypoints"][0]["kind"] == "turn"
    assert patch["waypoints"][1]["kind"] == "dest"
    assert len(json.dumps(patch, ensure_ascii=False)) < 500

    # Arrival detection within 25m
    p_arr = eng.record_gps("voxx2", 31.548010, 120.312510, is_wgs84=False)
    assert p_arr is not None and p_arr["arrived"] is True and p_arr["active"] is False


async def test_3step_silent_investigation() -> None:
    with tempfile.TemporaryDirectory(prefix="vox_p23_") as tmp:
        h = HormonesV2(snapshot_path=os.path.join(tmp, "h.json"), causal_log_path=os.path.join(tmp, "c.jsonl"))
        tg = TodoGoalStore(os.path.join(tmp, "vox.db"))
        eng = AmapNavEngine(api_key="test", http_session=FakeHttp())
        spoken: list[str] = []

        step_counter = {"n": 0}

        async def fake_llm(messages, tools):
            step_counter["n"] += 1
            if step_counter["n"] == 1:
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                        }
                    ],
                }
            # Step 2: saw 11°C + 4-level north wind -> proactively care about jacket
            assert "11" in messages[-1]["content"]
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "proactive_speak",
                            "arguments": json.dumps(
                                {"text": "外面刮 4 级北风才 11 度，顺手带件厚外套再出门吧。", "reason": "weather_care"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

        async def exec_tool(name: str, args: dict) -> str:
            if name == "get_weather":
                return json.dumps(await eng.get_weather("voxx2", args.get("city", "")), ensure_ascii=False)
            return '{"ok":false}'

        async def speak_cb(text: str, speed: float) -> None:
            spoken.append(text)

        loop = AgencyLoopV2(
            client_id="voxx2",
            system_prompt="STATIC_SYS",
            unified_tools=AMAP_TOOLS_SCHEMA + AGENCY_CONTROL_TOOLS,
            hormones=h,
            todo_goals=tg,
            llm_caller=fake_llm,
            tool_executor=exec_tool,
            speak_cb=speak_cb,
            state_provider=lambda: {
                "busy": False,
                "last_user_ts": time.time() - 120.0,
                "latest_caption": "男子站在门口换鞋，穿着单薄卫衣",
                "recent_turns": [],
            },
            enabled=True,
        )
        out = await loop.step_once(force=True)
        assert out["action"] == "proactive_speak"
        assert out["tools"] == ["get_weather", "proactive_speak"]
        assert len(spoken) == 1 and "11 度" in spoken[0]
        tg.close()


def main() -> None:
    test_geo_math()
    asyncio.run(test_amap_nav_and_weather())
    asyncio.run(test_3step_silent_investigation())
    print("ALL PHASE 2+3 WEATHER & NAV SMOKE TESTS PASSED!")


if __name__ == "__main__":
    main()
