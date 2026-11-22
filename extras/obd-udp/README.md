# OBD-II → UDP 广播（可选）

车载 HUD 的最小闭环：树莓派读 OBD（蓝牙 ELM327/Vgate），10Hz 广播 JSON 到局域网，
车机或眼镜端监听 `UDP:18766` 即可画出车速/转速，不依赖 Home Assistant。

```bash
pip install obd pyserial
OBD_PORT=/dev/rfcomm0 python3 obd_udp_broadcast.py
```

字段：`ts / speed / rpm / coolant / voltage / throttle / load / fuel / connected`。

熄火后 `connected` 变 0，下游应据此切黑屏，而不是继续显示旧车速。
