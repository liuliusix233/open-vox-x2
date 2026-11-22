#!/usr/bin/env python3
"""OBD-II → UDP 10Hz 广播（可选模块，树莓派 / 任意带蓝牙的 Linux）。

用途：把车速、转速、水温、油门等实时量以 JSON 广播到局域网
（默认 UDP 18766），车机端 / 眼镜端只要监听这个端口就能做驾驶 HUD，
不需要连 Home Assistant，也不需要公网。

依赖：
    pip install obd pyserial

用法：
    OBD_PORT=/dev/rfcomm0 python3 obd_udp_broadcast.py
    # 蓝牙 OBD 适配器（Vgate iCar Pro 等）配对后一般是 /dev/rfcomm0

协议（每帧一个 JSON，UTF-8）：
    {"ts": 1795358400.12, "speed": 42, "rpm": 1830, "coolant": 88,
     "throttle": 14.5, "load": 31.2, "voltage": 13.9, "fuel": 62}

设计取舍：
  * 单线程轮询 + 分档频率：车速/转速 10Hz，水温/电压 1Hz，油量 0.1Hz
  * 蓝牙串口只能串行访问，所有请求共用一把锁
  * 收不到数据时广播 {"connected": 0}，下游据此进「熄火黑屏」而不是显示旧值
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any

try:
    import obd  # python-obd
except ImportError:  # 允许没有 obd 库时以「空数据」模式起来，方便先验证网络
    obd = None

UDP_HOST = os.environ.get("OBD_UDP_HOST", "255.255.255.255")
UDP_PORT = int(os.environ.get("OBD_UDP_PORT", "18766"))
UDP_HZ = float(os.environ.get("OBD_UDP_HZ", "10"))
OBD_PORT = os.environ.get("OBD_PORT", "/dev/rfcomm0")
OBD_BAUD = int(os.environ.get("OBD_BAUD", "38400"))

FAST_PIDS = [("SPEED", "speed"), ("RPM", "rpm")]
MID_PIDS = [("COOLANT_TEMP", "coolant"), ("CONTROL_MODULE_VOLTAGE", "voltage")]
SLOW_PIDS = [("FUEL_LEVEL", "fuel"), ("ENGINE_LOAD", "load"), ("THROTTLE_POS", "throttle")]

_lock = threading.Lock()
_state: dict[str, Any] = {"connected": 0}
_last_mid = 0.0
_last_slow = 0.0


def _connect() -> Any:
    if obd is None:
        return None
    try:
        conn = obd.OBD(OBD_PORT, baudrate=OBD_BAUD, fast=False, timeout=5)
        if conn.is_connected():
            print(f"[obd] connected on {OBD_PORT}", flush=True)
            return conn
        print(f"[obd] not connected on {OBD_PORT}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[obd] connect failed: {exc}", flush=True)
    return None


def _query(conn: Any, name: str) -> Any:
    """返回数值或 None；obd 库在断连时抛异常，这里统一兜住。"""
    if conn is None or obd is None:
        return None
    cmd = getattr(obd.commands, name, None)
    if cmd is None:
        return None
    try:
        with _lock:
            r = conn.query(cmd)
        if r.is_null():
            return None
        mag = r.magnitude if hasattr(r, "magnitude") else r.value
        return float(mag) if mag is not None else None
    except Exception:  # noqa: BLE001
        return None


def poll_loop() -> None:
    global _last_mid, _last_slow
    conn = _connect()
    while True:
        now = time.time()
        if conn is None:
            conn = _connect()
            time.sleep(2.0)
            if conn is None:
                _state.update({"connected": 0, "ts": now})
                continue

        for pid, key in FAST_PIDS:
            v = _query(conn, pid)
            if v is not None:
                _state[key] = v

        if now - _last_mid > 1.0:
            _last_mid = now
            for pid, key in MID_PIDS:
                v = _query(conn, pid)
                if v is not None:
                    _state[key] = v

        if now - _last_slow > 10.0:
            _last_slow = now
            for pid, key in SLOW_PIDS:
                v = _query(conn, pid)
                if v is not None:
                    _state[key] = v

        _state["connected"] = 1
        _state["ts"] = now
        time.sleep(max(0.0, 1.0 / max(1.0, UDP_HZ)))


def broadcast_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[udp] broadcast -> {UDP_HOST}:{UDP_PORT} @{UDP_HZ:.0f}Hz", flush=True)
    while True:
        payload = json.dumps(_state, ensure_ascii=False).encode("utf-8")
        try:
            sock.sendto(payload, (UDP_HOST, UDP_PORT))
        except OSError as exc:
            print(f"[udp] send failed: {exc}", flush=True)
        time.sleep(1.0 / max(1.0, UDP_HZ))


def main() -> None:
    threading.Thread(target=poll_loop, daemon=True, name="obd-poll").start()
    broadcast_loop()


if __name__ == "__main__":
    main()
