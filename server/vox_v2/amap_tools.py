"""amap_tools.py — Amap (高德) weather, geocoding, and walking navigation engine for vox_v2.

Includes:
  - Pure-Python WGS-84 -> GCJ-02 conversion (critical for Android GPS in China).
  - Geodesic Haversine distance & true-north bearing (0..360 deg) for 60fps AR compass HUD.
  - Native OpenAI Tool Calling schemas: get_weather, start_navigation, stop_navigation.
  - Compact NAV_PATCH (<400 bytes) pushed over existing FT_CONTROL channel.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("vox_v2.amap")

AMAP_BASE = "https://restapi.amap.com/v3"

AMAP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市或当前定位的实时天气、体感温度、风力和今明预报（高德天气）。关心用户穿衣或出行前可静默调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 Shanghai、上海；留空则优先用当前 GPS 或默认Shanghai"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_navigation",
            "description": "为用户启动步行 AR 导航，把带距离和方位角的游戏风顶部导航条推送到雷鸟 X2 眼镜上。",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地名称，如 Shanghai Library、万象城"},
                    "city": {"type": "string", "description": "城市名，默认 Shanghai"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_navigation",
            "description": "结束当前步行 AR 导航并收起眼镜顶部的导航条。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "结束原因，如 已到达、用户取消"},
                },
            },
        },
    },
]


# ── Coordinate & Geodesic Math ────────────────────────────────────────────────

_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lat: float, lon: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lat: float, lon: float) -> tuple[float, float]:
    """Convert Android GPS WGS-84 coordinates to Amap GCJ-02."""
    if _out_of_china(lat, lon):
        return lat, lon
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1.0 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1.0 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lat + dlat, lon + dlon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true-north bearing in degrees [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ── Navigation State ──────────────────────────────────────────────────────────

@dataclass
class NavStep:
    lat: float       # GCJ-02
    lon: float       # GCJ-02
    action: str      # e.g. "右转", "左转", "直行"
    instruction: str
    distance_m: float


@dataclass
class NavSession:
    dest_name: str
    dest_lat: float  # GCJ-02
    dest_lon: float  # GCJ-02
    steps: list[NavStep] = field(default_factory=list)
    step_idx: int = 0
    started_ts: float = field(default_factory=time.time)
    last_patch_ts: float = 0.0
    arrived: bool = False


class AmapNavEngine:
    """Async Amap weather + walking navigation engine with per-client sessions."""

    def __init__(
        self,
        api_key: str | None = None,
        default_city: str = "Shanghai",
        http_session: Any = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.environ.get("AMAP_API_KEY", "")).strip()
        self.default_city = os.environ.get("VOX_DEFAULT_CITY", default_city)
        self.http = http_session
        self._lock = threading.RLock()
        self._sessions: dict[str, NavSession] = {}
        self._last_gps: dict[str, tuple[float, float, float]] = {}  # client -> (gcj_lat, gcj_lon, ts)
        self._weather_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def record_gps(self, client_id: str, lat: float, lon: float, is_wgs84: bool = True) -> dict[str, Any] | None:
        """Record latest GPS fix from glasses; returns updated NAV_PATCH if nav is active."""
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (lat == 0.0 and lon == 0.0):
            return None
        glat, glon = wgs84_to_gcj02(lat, lon) if is_wgs84 else (lat, lon)
        now = time.time()
        with self._lock:
            self._last_gps[client_id] = (glat, glon, now)
            sess = self._sessions.get(client_id)
            if sess is None or sess.arrived:
                return None
            return self._build_nav_patch_locked(client_id, sess, glat, glon, now)

    def get_last_gps(self, client_id: str) -> tuple[float, float] | None:
        with self._lock:
            rec = self._last_gps.get(client_id)
            if not rec:
                return None
            return rec[0], rec[1]

    def has_active_nav(self, client_id: str) -> bool:
        with self._lock:
            s = self._sessions.get(client_id)
            return bool(s and not s.arrived)

    def stop_navigation(self, client_id: str, reason: str = "user_request") -> dict[str, Any]:
        with self._lock:
            self._sessions.pop(client_id, None)
        return {
            "ok": True,
            "reason": reason,
            "nav_patch": {"type": "NAV_PATCH", "active": False, "dest": "", "waypoints": []},
        }

    async def _amap_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("AMAP_API_KEY 未配置")
        if self.http is None:
            raise RuntimeError("http_session 未绑定")
        q = {**params, "key": self.api_key, "output": "JSON"}
        async with self.http.get(f"{AMAP_BASE}{path}", params=q, timeout=10) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def regeo(self, lat: float, lon: float, radius: int = 200) -> dict[str, Any] | None:
        """Reverse geocode: (lat, lon) -> nearest road / house number / district.

        Returns None when no Web-Service key is configured (never raises, never blocks
        the conversation). Amap expects "lon,lat", so the order is swapped here.
        """
        if not self.api_key or self.http is None:
            return None
        now = time.time()
        cache = getattr(self, "_regeo_cache", None)
        if cache and (now - cache[0]) < 120.0 and abs(cache[1] - lat) < 1e-4 and abs(cache[2] - lon) < 1e-4:
            return cache[3]
        try:
            j = await self._amap_get(
                "/geocode/regeo",
                {"location": f"{lon:.6f},{lat:.6f}", "extensions": "base", "radius": radius},
            )
        except Exception as exc:
            log.debug("regeo failed: %s", exc)
            return None
        if str(j.get("status")) != "1":
            log.info("RE_GEO unavailable status=%s info=%s", j.get("status"), j.get("info"))
            return None
        rg = j.get("regeocode") or {}
        comp = rg.get("addressComponent") or {}

        def _flat(v: Any) -> str:
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, dict):
                return str(v.get("name") or "").strip()
            if isinstance(v, list):
                return "".join(str(x.get("name") or "") if isinstance(x, dict) else str(x) for x in v).strip()
            return ""

        road = _flat(comp.get("road")) or _flat(comp.get("street"))
        number = _flat(comp.get("number"))
        township = _flat(comp.get("township"))
        district = _flat(comp.get("district")) or _flat(comp.get("city"))
        out = {
            "road": road,
            "number": number,
            "township": township,
            "district": district,
            "address": str(rg.get("formatted_address") or "").strip(),
        }
        self._regeo_cache = (now, lat, lon, out)
        return out

    async def get_weather(self, client_id: str, city: str = "") -> dict[str, Any]:
        target_city = (city or "").strip() or self.default_city
        now = time.time()
        with self._lock:
            cached = self._weather_cache.get(target_city)
            if cached and (now - cached[0]) < 900.0:
                return cached[1]

        # If no city explicitly given and we have GPS, resolve adcode via regeo
        adcode = target_city
        gps = self.get_last_gps(client_id)
        if not city.strip() and gps is not None:
            try:
                rj = await self._amap_get("/geocode/regeo", {"location": f"{gps[1]:.6f},{gps[0]:.6f}"})
                comp = (rj.get("regeocode") or {}).get("addressComponent") or {}
                if comp.get("adcode"):
                    adcode = str(comp["adcode"])
                    target_city = str(comp.get("district") or comp.get("city") or target_city)
            except Exception as exc:
                log.debug("regeo fallback to default city: %s", exc)

        try:
            live_j = await self._amap_get("/weather/weatherInfo", {"city": adcode, "extensions": "base"})
        except Exception as exc:
            # 2026-11-22：手头这把是高德 JS API key，REST 一律 10001 INVALID_USER_KEY。
            # 为了让"冷不冷"这类问题今天就能答，退到无 key 的 wttr.in，并在结果里标 source。
            log.warning("amap weather failed (%s), falling back to wttr.in", exc)
            fb = await self._wttr_weather(target_city)
            if fb is not None:
                with self._lock:
                    self._weather_cache[target_city] = (now, fb)
                return fb
            return {"ok": False, "reason": f"高德天气不可用：{exc}；备用源也失败", "city": target_city}
        lives = live_j.get("lives") or []
        live = lives[0] if lives else {}

        forecast_summary = ""
        try:
            fc_j = await self._amap_get("/weather/weatherInfo", {"city": adcode, "extensions": "all"})
            casts = ((fc_j.get("forecasts") or [{}])[0].get("casts")) or []
            if casts:
                c0 = casts[0]
                forecast_summary = (
                    f"今日白天{c0.get('dayweather','')}{c0.get('daytemp','')}°C，"
                    f"夜间{c0.get('nightweather','')}{c0.get('nighttemp','')}°C"
                )
        except Exception:
            pass

        result = {
            "ok": bool(live),
            "city": live.get("city", target_city),
            "weather": live.get("weather", ""),
            "temp_c": live.get("temperature", ""),
            "wind_dir": live.get("winddirection", ""),
            "wind_power": live.get("windpower", ""),
            "humidity": live.get("humidity", ""),
            "forecast": forecast_summary,
            "report_time": live.get("reporttime", ""),
        }
        if result["ok"]:
            with self._lock:
                self._weather_cache[target_city] = (now, result)
        return result

    async def _wttr_weather(self, city: str) -> dict[str, Any] | None:
        """无 key 备用源：wttr.in 的 json 接口（中文名可直接传）。"""
        if self.http is None:
            return None
        try:
            async with self.http.get(
                "https://wttr.in/" + city, params={"format": "j1", "lang": "zh"}, timeout=10
            ) as resp:
                resp.raise_for_status()
                j = await resp.json(content_type=None)
            cur = (j.get("current_condition") or [{}])[0]
            area = ((j.get("nearest_area") or [{}])[0].get("areaName") or [{}])[0].get("value", city)
            desc = (cur.get("lang_zh") or cur.get("weatherDesc") or [{}])[0].get("value", "")
            return {
                "ok": True,
                "source": "wttr.in",
                "city": area,
                "weather": desc,
                "temp_c": cur.get("temp_C", ""),
                "feels_like_c": cur.get("FeelsLikeC", ""),
                "wind_dir": cur.get("winddir16Point", ""),
                "wind_power": cur.get("windspeedKmph", ""),
                "humidity": cur.get("humidity", ""),
                "forecast": "",
                "report_time": cur.get("localObsDateTime", ""),
            }
        except Exception as exc:
            log.warning("wttr fallback failed: %s", exc)
            return None

    async def start_navigation(
        self,
        client_id: str,
        destination: str,
        city: str = "",
        origin_lat_gcj: float | None = None,
        origin_lon_gcj: float | None = None,
    ) -> dict[str, Any]:
        dest_q = destination.strip()
        if not dest_q:
            return {"ok": False, "reason": "缺少目的地"}
        use_city = (city or "").strip() or self.default_city

        # 1. Resolve destination POI
        poi_j = await self._amap_get(
            "/place/text",
            {"keywords": dest_q, "city": use_city, "citylimit": "true", "offset": "3", "page": "1"},
        )
        pois = poi_j.get("pois") or []
        if not pois:
            return {"ok": False, "reason": f"高德未找到目的地：{dest_q}"}
        poi = pois[0]
        dlon_s, dlat_s = str(poi["location"]).split(",")
        dest_lat, dest_lon = float(dlat_s), float(dlon_s)
        dest_name = str(poi.get("name") or dest_q)

        # 2. Determine origin (GCJ-02)
        if origin_lat_gcj is not None and origin_lon_gcj is not None:
            olat, olon = origin_lat_gcj, origin_lon_gcj
        else:
            gps = self.get_last_gps(client_id)
            if gps is None:
                # Start direct compass-only mode until first GPS fix arrives
                sess = NavSession(dest_name=dest_name, dest_lat=dest_lat, dest_lon=dest_lon)
                with self._lock:
                    self._sessions[client_id] = sess
                return {
                    "ok": True,
                    "dest": dest_name,
                    "note": "已锁定目的地，等待眼镜 GPS 定位更新后自动计算精确距离与路线。",
                    "nav_patch": {
                        "type": "NAV_PATCH",
                        "active": True,
                        "dest": dest_name,
                        "total_dist_m": -1,
                        "next_dist_m": -1,
                        "next_instruction": f"等待定位 · 前往{dest_name}",
                        "waypoints": [],
                    },
                }
            olat, olon = gps

        # 3. Walking directions
        walk_j = await self._amap_get(
            "/direction/walking",
            {"origin": f"{olon:.6f},{olat:.6f}", "destination": f"{dest_lon:.6f},{dest_lat:.6f}"},
        )
        paths = ((walk_j.get("route") or {}).get("paths")) or []
        steps: list[NavStep] = []
        total_m = haversine_m(olat, olon, dest_lat, dest_lon)
        if paths:
            p0 = paths[0]
            try:
                total_m = float(p0.get("distance", total_m))
            except (TypeError, ValueError):
                pass
            for st in p0.get("steps") or []:
                poly = str(st.get("polyline", ""))
                pts = poly.split(";") if poly else []
                if not pts:
                    continue
                end_lon_s, end_lat_s = pts[-1].split(",")
                action = str(st.get("action") or "直行").strip()
                if isinstance(st.get("action"), list):
                    action = "直行"
                steps.append(
                    NavStep(
                        lat=float(end_lat_s),
                        lon=float(end_lon_s),
                        action=action[:12] or "直行",
                        instruction=str(st.get("instruction") or "")[:44],
                        distance_m=float(st.get("distance") or 0.0),
                    )
                )

        sess = NavSession(dest_name=dest_name, dest_lat=dest_lat, dest_lon=dest_lon, steps=steps)
        with self._lock:
            self._sessions[client_id] = sess
            patch = self._build_nav_patch_locked(client_id, sess, olat, olon, time.time())

        return {
            "ok": True,
            "dest": dest_name,
            "total_dist_m": int(round(total_m)),
            "steps_count": len(steps),
            "next_instruction": patch.get("next_instruction", ""),
            "nav_patch": patch,
        }

    def _build_nav_patch_locked(
        self, client_id: str, sess: NavSession, cur_lat: float, cur_lon: float, now: float
    ) -> dict[str, Any]:
        dest_dist = haversine_m(cur_lat, cur_lon, sess.dest_lat, sess.dest_lon)
        dest_bear = bearing_deg(cur_lat, cur_lon, sess.dest_lat, sess.dest_lon)

        if dest_dist <= 25.0:
            sess.arrived = True
            return {
                "type": "NAV_PATCH",
                "active": False,
                "arrived": True,
                "dest": sess.dest_name,
                "total_dist_m": int(round(dest_dist)),
                "next_instruction": f"已到达 {sess.dest_name}",
                "waypoints": [],
            }

        # Advance waypoint index if user reached current turn point
        while sess.step_idx < len(sess.steps) - 1:
            st = sess.steps[sess.step_idx]
            if haversine_m(cur_lat, cur_lon, st.lat, st.lon) <= 18.0:
                sess.step_idx += 1
            else:
                break

        waypoints: list[dict[str, Any]] = []
        next_dist = dest_dist
        next_inst = f"前往 {sess.dest_name}"

        if sess.steps and sess.step_idx < len(sess.steps):
            st = sess.steps[sess.step_idx]
            sd = haversine_m(cur_lat, cur_lon, st.lat, st.lon)
            sb = bearing_deg(cur_lat, cur_lon, st.lat, st.lon)
            next_dist = sd
            next_inst = st.instruction or f"前方{int(round(sd))}米{st.action}"
            if sd > 12.0 and abs(sd - dest_dist) > 25.0:
                waypoints.append(
                    {
                        "label": st.action or "拐点",
                        "kind": "turn",
                        "bearing_deg": round(sb, 1),
                        "dist_m": int(round(sd)),
                    }
                )

        waypoints.append(
            {
                "label": sess.dest_name[:12],
                "kind": "dest",
                "bearing_deg": round(dest_bear, 1),
                "dist_m": int(round(dest_dist)),
            }
        )
        sess.last_patch_ts = now
        return {
            "type": "NAV_PATCH",
            "active": True,
            "arrived": False,
            "dest": sess.dest_name,
            "total_dist_m": int(round(dest_dist)),
            "next_dist_m": int(round(next_dist)),
            "next_instruction": next_inst[:44],
            "waypoints": waypoints,
        }
