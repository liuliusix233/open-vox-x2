# Open Vox X2

**把一副 RayNeo（雷鸟）X2 双目眼镜，变成一个会看、会记、会主动开口的全双工语音助手。**

服务端 + Android 客户端全开源。不绑定任何云厂商：任何 OpenAI 兼容的 `/v1/chat/completions`
都能当大脑，任何 `/v1/audio/speech` 都能当嗓子；有本地显卡时可以完全离线跑（CosyVoice2 + 本地推理）。

```
              双目画面 1280x480            语音上行 16kHz
   ┌──────────────────────────┐        ┌──────────────────────┐
   │   RayNeo X2 (Android 11) │◀──────▶│  WebSocket 全双工     │
   │  HUD / 麦克风 / 触摸 / IMU │        │  wss://host/ws/v1/... │
   └──────────────────────────┘        └───────────┬──────────┘
                                                    │
                        ┌───────────────────────────▼───────────────────────────┐
                        │                    vox_v2 (aiohttp)                    │
                        │                                                       │
                        │  ┌─────────────┐   ┌──────────────┐   ┌────────────┐  │
                        │  │ 前台快脑     │   │ 协调层        │   │ 后台工兵    │  │
                        │  │ 秒答 / 派单  │──▶│ 冷却·去重     │──▶│ 多步工具链  │  │
                        │  │ TTFT ~300ms │◀──│ 静默窗插话    │◀──│ 结果回灌    │  │
                        │  └──────┬──────┘   └──────────────┘   └─────┬──────┘  │
                        │         │                                    │         │
                        │  ┌──────▼────────────────────────────────────▼──────┐ │
                        │  │        vox_unified.db（单库：对话 / 视觉 / 记忆）  │ │
                        │  │  turns · memory_units · vision_frames · todos      │ │
                        │  └───────────────────────────────────────────────────┘ │
                        └───────────────────────────────────────────────────────┘
```

## 核心特性

| 能力 | 说明 |
|---|---|
| **双脑全双工** | 前台小脑用低延迟档位秒答并「派单」，后台工兵跑多步工具链；协调层负责冷却、原文去重、静默窗择机插话 |
| **轻索引 + 原图反刍** | 每帧只存一句 caption 建索引；问到细节才把原图交给视觉模型重看，省显存也省 token |
| **纯黑光波导 HUD** | 只用纯黑背景 + 高对比描边，避免光波导「看见一块灰底」；亮度有兜底与自适应 |
| **2×2 功耗自适应 / 真休眠** | 插电 / 电池 × 亮屏 / 息屏 四种策略；真休眠时停心跳、停插话、停联网 |
| **代码级访客隔离** | 访客模式下工具白名单硬拦截 + 上下文按进入时间戳隔离；门锁/安防域永远拒绝 |
| **9 套自动化冒烟** | 服务端 9 个测试文件覆盖工具链、双脑、打断、访客、导航、HUD、功耗协议 |

### 四维仿生激素节律与主动智能体（Hormone-Driven Agency）

市面上的语音助手「主动开口」基本是死板定时器：每 N 分钟评估一次，永远同一个语气。
这里换成一套**连续动力学状态机** + **因果反馈闭环**，让它自己学会什么时候该闭嘴。

**1. 四维连续动力学（`hormones_v2.py`）**

维护四个连续状态量，各自独立半衰期，随对话/场景/打断实时演化：

| 维度 | 含义 | 半衰期 | 被什么推高 |
|---|---|---|---|
| **NE** 去甲肾上腺素 | 警觉 | 45 s | 设备异常、场景突变、到期事项 |
| **DA** 多巴胺 | 探索/兴致 | 100 s | 用户回应、新画面、新鲜细节 |
| **OXT** 催产素 | 关系/关切 | 240 s | 连续对话、熟人语气 |
| **HT** 血清素 | 克制/收敛 | 180 s | 被打断、开口没人理、连续冷场 |

这四个量**直接决定**三件事：主动巡检间隔（`proactive_interval_s`）、本次开口的阈值、
以及 TTS 语速与措辞档位（NE→紧凑 +10%，HT→平淡 −5%）。

**2. 35 秒因果反馈闭环**

每次主动开口后记一个时间戳，35 秒后回看用户是否回应：

- **被回应** → DA +0.34、OXT +0.14、HT −0.10（正向强化，下次更愿意开口）
- **被无视** → HT +0.18（拉长沉默间隔，越来越懂分寸）
- **被直接打断** → HT +0.20、DA −0.15（当场收敛，且不记仇）

**3. 三步静默调查 + KV 缓存友好（`agency_loop_v2.py`）**

空闲时先**静默**调只读工具（查记忆 / 待办 / 画面）核实，再决定 `proactive_speak` 还是
`do_nothing`；配合单例互斥、90 秒冷却与 Jaccard 原文去重（>60% 直接丢）。
所有激素状态**只注入末条 user 动态尾**，静态 `SYSTEM_PROMPT` 一字不动，前缀 KV 缓存照常命中。

## 《RayNeo X2 硬件开发避坑实录》

真机上踩过的坑，按重要性排序——这些都不在官方文档里。

1. **麦克风必须走 `audio_source_record=camcorder`**
   默认只开朝前的前置麦，声音闷、必须靠 12 倍软件增益硬撑。用
   `AudioManager.setParameters("audio_source_record=camcorder")` 切到**双镜腿立体声**，
   录 `CHANNEL_IN_STEREO` 后自己 `(L+R)/2` 合成单声道 16kHz：ASR 准确率明显提升，
   而且没有固件那层激进的降噪。释放时务必 `setParameters("audio_source_record=off")`，
   否则别的 App 拿到的是坏路由。

2. **光波导要纯黑底 + 描边**
   半透明灰底在光波导上会变成一块「脏玻璃」。背景写死 `#000000`，
   文字/图形用高亮青或纯白 + 1~2px 描边；亮度低于 20% 时夜间几乎不可读，
   需要客户端兜底到 25% 以上。

3. **系统息屏会断 Wi-Fi**
   屏幕一灭，系统会把 Wi-Fi 降频甚至断开，WebSocket 随之掉线。
   要常驻就得：前台 Service + 息屏保活（本项目用 `VoxKeepAliveService`），
   并且把重连做成 LAN 快探（600ms）失败即切公网 WSS，而不是傻等 5 次。

4. **桌面入口是 `com.rayneo.mercury.app`**
   想「用眼镜原生桌面点开 App」，不要自己造 Launcher Activity 抢默认桌面，
   而是注册一个能被 Mercury 桌面识别的入口，避免把系统桌面顶掉。

5. **触控只有 `ACTION_UP` 可靠**
   镜腿触摸给的是持续 MotionEvent 流，`ACTION_DOWN` 经常被系统手势截胡；
   双击/长按一律以 `ACTION_UP` 的时间窗判定（本项目 300ms 双击窗 + 700ms 长按兜底）。

6. **图片帧会挤掉音频帧**
   双目 JPEG 一帧几百 KB，和音频共用一条 WS 时会把语音包堵住。
   客户端发送队列超过阈值要丢图不丢音；播报期间干脆暂停推图。

## 5 分钟快速上手

### 1) 起服务端

```bash
git clone https://github.com/<you>/open-vox-x2 && cd open-vox-x2
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt
cp .env.example .env    # 至少填 VOX_LLM_URL
cd server && PYTHONPATH=. python3 run_vox_v2.py
```

只有一台普通电脑、没有 2080 Ti？把 TTS 换成任意 OpenAI 兼容接口即可：

```bash
VOX_TTS_BACKEND=openai
VOX_TTS_OPENAI_URL=https://api.openai.com/v1
VOX_TTS_OPENAI_KEY=sk-...
VOX_TTS_OPENAI_VOICE=alloy
```

### 2) 跑通 9 套冒烟

```bash
cd server
for t in smoke phase1 phase23 social_toolforge bargein guest_mode \
         duplex_dualbrain power_mode_protocol places; do
  PYTHONPATH=. python3 -m vox_v2.test_$t || exit 1
done
```

### 3) 编译客户端

```bash
cd android
cat > local.properties <<'EOF'
sdk.dir=/path/to/Android/Sdk
EOF
./gradlew assembleDebug -PVOX_LAN_HOST=<your-lan-ip> \
    -PVOX_LAN_PORT=8766 \
    -PVOX_WAN_WS=wss://your-domain.example.com/vox/ws/v1/duplex \
    -PVOX_TOKEN=your-shared-token
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

接口约定：`ws://<host>:<port>/ws/v1/duplex?client=<clientId>`，公网要带 `X-Vox-Token` 头。

## 截图

实时模式与驾驶模式 HUD 实拍截图即将补充（docs/images/hud-live.png、docs/images/hud-drive.png）。

<!-- 截图补上后把下面两行取消注释
| 实时模式 | 驾驶模式 |
|---|---|
| ![HUD live](docs/images/hud-live.png) | ![HUD drive](docs/images/hud-drive.png) |
-->

## 目录结构

```
server/
  run_vox_v2.py              # 入口：SenseVoice ASR + TTS + aiohttp
  vox_v2/
    app.py                   # WebSocket 全双工 / 会话状态机 / TTS 串行队列
    brain.py                 # 原生 Tool Calling、流式分句、记忆反刍
    duplex_coordinator.py    # 双脑协调：派单 / 作废 / 静默窗插话 / 访客隔离
    agency_loop_v2.py        # 主动心跳：观察→查证→决定是否开口
    memory.py                # 单库检索：视觉 / 对话 / 事实 / 地点
    audio.py                 # VAD、CosyVoice2 与 OpenAI TTS 适配器
    vision.py photo_lib.py   # 帧索引、关键帧归档、后台打标
    speaker_id.py            # 15ms 级声纹（可选访客模式依赖）
    hormones_v2.py           # 主动开口的节律门控（不发神经）
    test_*.py                # 9 套冒烟
android/app/src/main/java/com/openvox/x2/
  MainActivity.kt            # HUD 组装、触摸、模式切换
  VoxAudioEngine.kt          # 双镜腿立体声采集（camcorder 路由）
  VoxDuplexClient.kt         # WS 收发、音频/图片分流、重连
  VoxDriveHudView.kt         # 驾驶 HUD
  VoxPowerPolicy.kt          # 2×2 功耗策略
extras/obd-udp/              # 树莓派 OBD 10Hz UDP 广播（可选）
```

## 可选模块（默认全关）

| 模块 | 打开方式 |
|---|---|
| Home Assistant 控制 | `VOX_HA_URL` + `VOX_HA_TOKEN` |
| 高德导航 | `AMAP_API_KEY` |
| 声纹访客模式 | `VOX_AUTO_GUEST=1` + enroll 声纹；默认只认口令 |
| 车载 OBD HUD | 跑 `extras/obd-udp`，客户端填 `VOX_DRIVE_SSID` |
| 双脑插话 | `VOX_DUPLEX_DUAL_BRAIN=1` |

## 隐私

所有数据落在你自己的 `vox_unified.db` 与本地帧目录；仓库里不含任何音频、图片、
声纹或密钥。`.gitignore` 已默认屏蔽 `*.db`、`*.wav`、`voiceprints.json`、`.env`。

## License

MIT
