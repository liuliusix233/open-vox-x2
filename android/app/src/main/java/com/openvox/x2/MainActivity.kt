package com.openvox.x2

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioManager
import android.media.ToneGenerator
import android.app.ActivityManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.IntentFilter
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.provider.Settings
import android.location.LocationManager
import android.location.LocationListener
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.*

/**
 * RayNeo X2 HUD —— 双目（640×480 ×2）+ 证据图卡片。
 *
 * 2026-11-22 两次合并：
 *  1) 双目：官方 framebuffer 是 1280×480，左/右各 640×480 一只眼，每只眼一套完整视图树。
 *  2) 证据图：服务端 HUD_PATCH 现在会带 images[]（320×240 JPEG base64），
 *     每只眼的证据卡里显示缩略图 + 时间戳标签，多图 4.5s 轮播，镜腿左右可翻页，默认 12s 淡出。
 */
class MainActivity : Activity(), VoxDuplexClient.Listener {
    companion object {
        private const val STATUS_BAR_H = 44
        private const val IDLE_MIN_MS = 15_000L
        private const val IDLE_MIN_M = 15f
        private const val NAV_MIN_MS = 2_000L
        private const val NAV_MIN_M = 3f
        private const val EYE_COUNT = 2
        private const val CYCLE_MS = 4500L
    }

    private class EyeWidgets {
        lateinit var reticle: WaveguideReticleView
        lateinit var statusBar: VoxStatusBarView
        lateinit var compass: ArCompassBarView
        lateinit var evidence: VoxEvidenceCardView
        lateinit var subtitle: VoxSubtitleView
        lateinit var driveHud: VoxDriveHudView
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val clockFmt = SimpleDateFormat("HH:mm", Locale.getDefault())
    private val eyes = ArrayList<EyeWidgets>(EYE_COUNT)


    private lateinit var config: VoxConfig
    private lateinit var duplex: VoxDuplexClient
    private lateinit var audioEngine: VoxAudioEngine
    private lateinit var sensorGate: SensorGate
    private lateinit var camera: VoxCamera

    private var activeBitmaps: List<Pair<Bitmap, String>> = emptyList()
    private var activeImageIdx = 0

    private val hideEvidenceTask = Runnable { clearEvidenceCard() }
    private val cycleImageTask = object : Runnable {
        override fun run() {
            if (activeBitmaps.size > 1 && eyes.firstOrNull()?.evidence?.visibility == View.VISIBLE) {
                activeImageIdx = (activeImageIdx + 1) % activeBitmaps.size
                renderActiveEvidenceImage()
                mainHandler.postDelayed(this, CYCLE_MS)
            }
        }
    }
    private val clockTick = object : Runnable {
        override fun run() {
            updateStatusText()
            mainHandler.postDelayed(this, 15_000L)
        }
    }

    private var headingTracker: HeadHeadingTracker? = null
    private var isConnected = false
    private var isLan = false
    private var lastTapAt = 0L
    private val telemetryTask = object : Runnable {
        override fun run() {
            sendTelemetry()
            mainHandler.postDelayed(this, 30_000L)
        }
    }
    private var lastTouchTapAt = 0L
    private var lastTelemBattery = -999
    /** 命令设过的亮度（无 WRITE_SETTINGS 时用窗口亮度生效，遥测以此为准） */
    private var appliedBrightnessPct = -1
    private var batteryRx: BroadcastReceiver? = null
    private var driveDetector: DriveModeDetector? = null
    private var powerPolicy: VoxPowerPolicy? = null
    private var locManager: LocationManager? = null
    private var locListener: LocationListener? = null
    @Volatile private var lastLoc: Triple<Double, Double, Float>? = null
    private var navActive = false
    private var inDriveMode = false
    private var tone: ToneGenerator? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            )
        setContentView(buildStereoRoot())

        config = VoxConfig()
        duplex = VoxDuplexClient(config, this)
        audioEngine = VoxAudioEngine(this, duplex)
        sensorGate = SensorGate(this)
        camera = VoxCamera(this, duplex, sensorGate)

        powerPolicy = VoxPowerPolicy(this) { p -> mainHandler.post { applyPowerPolicy(p) } }

        headingTracker = HeadHeadingTracker(this) { deg ->
            mainHandler.post {
                eyes.forEach { it.compass.setHeadingDeg(deg) }
            }
        }
        audioEngine.onMicLevel = { lvl -> eyes.forEach { it.reticle.setMicLevel(lvl) } }
        audioEngine.onTtsLevel = { lvl -> eyes.forEach { it.reticle.setTtsLevel(lvl) } }
        audioEngine.onVoiceBargeIn = {
            Log.i("VoxKey", "VOICE_BARGE_IN -> LISTENING")
            mainHandler.post { eyes.forEach { it.reticle.setMode(WaveguideReticleView.Mode.LISTENING) } }
        }
        audioEngine.onSpeechOnset = { camera.captureOnce("vad") }
        sensorGate.onGazeLocked = { camera.captureOnce("gaze") }
        sensorGate.onLongStill = { VoxModeController.setMode(VoxPowerMode.SLEEP_LISTEN) }
        VoxModeController.listener = { newMode, oldMode ->
            mainHandler.post {
                if (newMode == VoxPowerMode.SLEEP_LISTEN) {
                    powerPolicy?.enterSystemSleep("screen_off_or_still")
                } else {
                    applyPowerMode(newMode, oldMode)
                }
            }
        }

        requestPermissionsAndStart()
    }

    // ── 双目装配 ───────────────────────────────────────────────

    private fun buildStereoRoot(): View {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        val stereo = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (i in 0 until EYE_COUNT) {
            stereo.addView(
                buildEyeHost(),
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
            )
        }
        root.addView(
            stereo,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        return root
    }

    private fun buildEyeHost(): View {
        val host = FrameLayout(this)
        val w = EyeWidgets()

        w.reticle = WaveguideReticleView(this)
        host.addView(
            w.reticle,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        w.statusBar = VoxStatusBarView(this)
        // 显式高度：View 没有 onMeasure，WRAP_CONTENT 会被撑成整屏（之前状态栏跑到正中）
        host.addView(
            w.statusBar,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                STATUS_BAR_H,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL
            ).apply { topMargin = 2 }
        )

        w.compass = ArCompassBarView(this).apply { visibility = View.GONE }
        host.addView(
            w.compass,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                86,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL
            ).apply { topMargin = 52 }
        )

        w.evidence = VoxEvidenceCardView(this).apply { visibility = View.GONE }
        host.addView(
            w.evidence,
            FrameLayout.LayoutParams(
                268,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.END or Gravity.CENTER_VERTICAL
            ).apply { rightMargin = 14 }
        )

        w.driveHud = VoxDriveHudView(this).apply { visibility = View.GONE }
        host.addView(
            w.driveHud,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        w.subtitle = VoxSubtitleView(this)
        host.addView(
            w.subtitle,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            )
        )

        eyes.add(w)
        return host
    }

    // ── 启动与权限 ─────────────────────────────────────────────

    private fun requestPermissionsAndStart() {
        val perms = arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)
        val missing = perms.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 101)
        } else {
            startStack()
        }
    }

    override fun onRequestPermissionsResult(reqCode: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(reqCode, perms, results)
        startStack()
    }

    private fun startStack() {
        try {
            startForegroundService(Intent(this, VoxKeepAliveService::class.java))
        } catch (_: Throwable) {
        }
        powerPolicy?.start()
        sensorGate.start()
        if (VoxModeController.current == VoxPowerMode.LIVE) camera.start()
        duplex.connect()
        audioEngine.start()
        headingTracker?.start()
        // 开源默认 DRIVE_SSID 为空 = 不启用驾驶模式自动切换；填了才起侦测
        if (BuildConfig.DRIVE_SSID.isNotBlank()) {
            driveDetector = DriveModeDetector(
                context = this,
                targetSsid = BuildConfig.DRIVE_SSID,
                udpPort = 18766,
                onDriveModeChanged = { inDrive -> mainHandler.post { applyDriveMode(inDrive) } },
                onTelemetry = { t -> mainHandler.post { eyes.forEach { it.driveHud.updateTelemetry(t) } } }
            )
            driveDetector?.start()
        }
        startLocationSampling()

        try { AdbTunnel.start(this) } catch (t: Throwable) { Log.w("VoxAdbTunnel", "start failed: " + t.message) }

        mainHandler.post(clockTick)
        mainHandler.removeCallbacks(telemetryTask)
        mainHandler.postDelayed(telemetryTask, 1_500L)
        if (batteryRx == null) {
            batteryRx = object : BroadcastReceiver() {
                override fun onReceive(c: Context?, i: Intent?) {
                    val (pct, _, _) = batterySnapshot()
                    if (pct != lastTelemBattery) {
                        lastTelemBattery = pct
                        sendTelemetry()
                    }
                }
            }
            try {
                registerReceiver(batteryRx, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            } catch (_: Throwable) {}
        }
    }

    override fun onDestroy() {
        VoxModeController.listener = null
        try { tone?.release() } catch (_: Throwable) {}
        tone = null
        mainHandler.removeCallbacksAndMessages(null)
        mainHandler.removeCallbacks(telemetryTask)
        try { locListener?.let { locManager?.removeUpdates(it) } } catch (_: Throwable) {}
        try { powerPolicy?.stop() } catch (_: Throwable) {}
        try { driveDetector?.stop() } catch (_: Throwable) {}
        driveDetector = null
        try { batteryRx?.let { unregisterReceiver(it) } } catch (_: Throwable) {}
        batteryRx = null
        clearEvidenceCard()
        headingTracker?.stop()
        audioEngine.stop()
        camera.stop()
        sensorGate.stop()
        duplex.close()
        super.onDestroy()
    }

    // 镜腿：中间=打断，左/右=翻证据图
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        Log.i("VoxKey", "DOWN code=" + keyCode + " repeat=" + (event?.repeatCount ?: -1)
            + " mode=" + VoxModeController.current)
        val showing = eyes.firstOrNull()?.evidence?.visibility == View.VISIBLE
        if (keyCode == KeyEvent.KEYCODE_DPAD_RIGHT && activeBitmaps.size > 1) {
            eyes.forEach { it.evidence.stepImage(1) }
            return true
        }
        if (keyCode == KeyEvent.KEYCODE_DPAD_LEFT && activeBitmaps.size > 1) {
            eyes.forEach { it.evidence.stepImage(-1) }
            return true
        }
        val templeKey = keyCode == KeyEvent.KEYCODE_DPAD_CENTER
            || keyCode == KeyEvent.KEYCODE_ENTER
            || keyCode == KeyEvent.KEYCODE_HEADSETHOOK
            || keyCode == KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE
        if (templeKey) {
            // 单击即刻打断（不等双击窗口，保证零延迟）
            audioEngine.bargeInClear()
            duplex.clientInterrupt("temple_tap")
            // 两击 = LIVE <-> STANDBY（窗口放宽到 700ms，实机连击常落在 350-600ms）
            val now = SystemClock.uptimeMillis()
            if (lastTapAt != 0L && now - lastTapAt <= 700L) {
                lastTapAt = 0L
                powerPolicy?.toggleLiveStandby()
            } else {
                lastTapAt = now
            }
            try { event?.startTracking() } catch (_: Throwable) {}
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    // ── 三模式（LIVE / STANDBY / SLEEP_LISTEN）───────────────

    private val hideHudTask = Runnable { hideHud() }

    override fun onGuestMode(active: Boolean) {
        Log.i("VoxDevice", "GUEST_MODE=" + active)
        mainHandler.post {
            eyes.forEach { it.statusBar.setModeLabel(if (active) "GUEST" else VoxModeController.current.name) }
        }
    }

    private fun applyPowerMode(newMode: VoxPowerMode, oldMode: VoxPowerMode) {
        duplex.sendModeChange(newMode.name)
        when (newMode) {
            VoxPowerMode.LIVE -> {
                camera.start()
                showHud()
                beep(true)
                Log.i("VoxMode", "MODE_CHANGE -> LIVE")
            }
            VoxPowerMode.STANDBY -> {
                camera.stop()
                beep(false)
                flashStatus("STANDBY")
                Log.i("VoxMode", "MODE_CHANGE -> STANDBY")
            }
            VoxPowerMode.SLEEP_LISTEN -> {
                camera.stop()
                hideHud()
                Log.i("VoxMode", "MODE_CHANGE -> SLEEP_LISTEN")
            }
        }
        eyes.forEach { it.statusBar.setModeLabel(newMode.name) }
        sendTelemetry()
    }

    /** 连上车载热点：中央视野清空，信息只贴下缘余光带 */
    private fun startLocationSampling() {
        try {
            locManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
            locListener = LocationListener { loc ->
                if (loc.accuracy <= 60f) {
                    lastLoc = Triple(loc.latitude, loc.longitude, loc.accuracy)
                    sensorGate.updateGeo(loc.latitude, loc.longitude, loc.accuracy)
                }
            }
            requestLocationUpdates(if (navActive) NAV_MIN_MS else IDLE_MIN_MS,
                                   if (navActive) NAV_MIN_M else IDLE_MIN_M)
        } catch (t: Throwable) {
            Log.w("VoxLoc", "start failed: " + t.message)
        }
    }

    private fun requestLocationUpdates(minMs: Long, minM: Float) {
        val lm = locManager ?: return
        val l = locListener ?: return
        try { lm.removeUpdates(l) } catch (_: Throwable) {}
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, minMs, minM, l, Looper.getMainLooper())
        } catch (_: Throwable) {}
        try {
            lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, minMs, minM, l, Looper.getMainLooper())
        } catch (_: Throwable) {}
        Log.i("VoxLoc", "updates minMs=" + minMs + " minM=" + minM)
    }

    private fun sendParkingPin() {
        val loc = lastLoc ?: return
        duplex.sendControlMessage("PARKING_PIN", mapOf(
            "lat" to loc.first, "lon" to loc.second, "acc" to loc.third
        ))
        Log.i("VoxLoc", "PARKING_PIN sent")
    }

    private fun applyDriveMode(inDrive: Boolean) {
        if (inDrive == inDriveMode) return
        if (!inDrive && inDriveMode) sendParkingPin()
        navActive = inDrive
        requestLocationUpdates(if (inDrive) NAV_MIN_MS else IDLE_MIN_MS,
                               if (inDrive) NAV_MIN_M else IDLE_MIN_M)
        inDriveMode = inDrive
        eyes.forEach { eye ->
            eye.driveHud.visibility = if (inDrive) View.VISIBLE else View.GONE
            eye.reticle.visibility = if (inDrive) View.INVISIBLE else View.VISIBLE
            eye.statusBar.setModeLabel(if (inDrive) "DRIVE" else VoxModeController.current.name)
            if (!inDrive) eye.driveHud.resetTelemetry()
            (eye.subtitle.layoutParams as? FrameLayout.LayoutParams)?.let { lp ->
                lp.bottomMargin = if (inDrive) 68 else 0
                eye.subtitle.layoutParams = lp
            }
        }
        Log.i("VoxDrive", "DRIVE_MODE=" + inDrive + " ssid=" + BuildConfig.DRIVE_SSID)
    }

    // ── 2×2 供电自适应 ───────────────────────────────────────

    private fun applyPowerPolicy(p: VoxPowerPolicy.EffectivePolicy) {
        Log.i("VoxPower", "POLICY mode=" + p.userMode + " power=" + p.power +
            " cam=" + p.cameraMode + " badge=" + p.statusBadge + " hb=" + p.heartbeatMult)

        // 1) 真休眠：停相机、上报 SLEEP、停麦、断 WS
        if (!p.keepAudioAndWs) {
            try { locListener?.let { locManager?.removeUpdates(it) } } catch (_: Throwable) {}
            try { camera.stop() } catch (_: Throwable) {}
            duplex.sendControlMessage(
                "MODE_CHANGE",
                mapOf("mode" to "SLEEP", "power_profile" to p.power.name)
            )
            try { audioEngine.stop() } catch (_: Throwable) {}
            try { duplex.close() } catch (_: Throwable) {}
            eyes.forEach {
                it.driveHud.visibility = View.GONE
                it.reticle.visibility = View.INVISIBLE
                it.statusBar.visibility = View.GONE
                it.evidence.visibility = View.GONE
                it.subtitle.visibility = View.GONE
            }
            return
        }

        // 2) 唤醒恢复：连 WS、开麦、按档位切相机
        requestLocationUpdates(if (navActive) NAV_MIN_MS else IDLE_MIN_MS,
                               if (navActive) NAV_MIN_M else IDLE_MIN_M)
        if (duplex.isClosed) duplex.connect()
        if (!audioEngineIsRunning()) {
            try { audioEngine.start() } catch (_: Throwable) {}
        }
        when (p.cameraMode) {
            VoxPowerPolicy.CameraMode.CONTINUOUS_STREAM -> {
                camera.captureMode = "STREAM"
                if (VoxModeController.current == VoxPowerMode.LIVE) camera.start()
            }
            VoxPowerPolicy.CameraMode.EVENT_AND_30S -> {
                camera.captureMode = "EVENT_30S"
                camera.start()
            }
            VoxPowerPolicy.CameraMode.SILENT_ARCHIVE_15S -> {
                camera.captureMode = "ARCHIVE_15S"
                camera.start()
            }
            VoxPowerPolicy.CameraMode.OFF -> camera.stop()
        }

        // 3) HUD
        if (p.showHud) {
            if (inDriveMode) {
                eyes.forEach { it.driveHud.visibility = View.VISIBLE; it.reticle.visibility = View.INVISIBLE }
            } else {
                showHud()
            }
        } else {
            hideHud()
        }
        eyes.forEach { it.statusBar.setModeLabel(p.statusBadge) }

        // 4) 上报模式与省电节拍
        duplex.sendControlMessage(
            "MODE_CHANGE",
            mapOf(
                "mode" to p.userMode.name,
                "power_profile" to p.power.name,
                "heartbeat_mult" to p.heartbeatMult
            )
        )
        sendTelemetry()
    }

    private fun audioEngineIsRunning(): Boolean = try {
        val f = audioEngine.javaClass.getDeclaredField("running")
        f.isAccessible = true
        f.getBoolean(audioEngine)
    } catch (_: Throwable) {
        true
    }

    private fun showHud() {
        mainHandler.removeCallbacks(hideHudTask)
        eyes.forEach {
            it.reticle.visibility = View.VISIBLE
            it.statusBar.visibility = View.VISIBLE
            it.subtitle.visibility = View.VISIBLE
        }
    }

    private fun hideHud() {
        eyes.forEach {
            it.reticle.visibility = View.INVISIBLE
            it.statusBar.visibility = View.GONE
            it.evidence.visibility = View.GONE
            it.subtitle.visibility = View.GONE
            it.compass.visibility = View.GONE
        }
    }

    /** 切换瞬间闪 1.2s 状态字后彻底隐藏（波导黑＝透光，APL≈0） */
    private fun flashStatus(label: String) {
        eyes.forEach {
            it.reticle.visibility = View.INVISIBLE
            it.evidence.visibility = View.GONE
            it.subtitle.visibility = View.GONE
            it.compass.visibility = View.GONE
            it.statusBar.visibility = View.VISIBLE
            it.statusBar.setModeLabel(label)
        }
        mainHandler.removeCallbacks(hideHudTask)
        mainHandler.postDelayed(hideHudTask, 1_200L)
    }

    private fun beep(up: Boolean) {
        try {
            if (tone == null) tone = ToneGenerator(AudioManager.STREAM_MUSIC, 60)
            tone?.startTone(
                if (up) ToneGenerator.TONE_PROP_ACK else ToneGenerator.TONE_PROP_NACK,
                80
            )
        } catch (_: Throwable) {}
    }

    /**
     * 2026-11-22 关键修正：X2 镜腿敲击在这台机器上是以 MotionEvent 进来的
     * （旧 desk 版就是这么实现的：onTouchEvent 里 380ms 双击 → toggleLive）。
     * 新客户端只接了 onKeyDown，所以真机敲击完全收不到；adb input keyevent 能进是因为走了按键通道。
     */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_UP) {
            val now = ev.eventTime
            Log.i("VoxKey", "TAP_TOUCH up x=" + ev.x.toInt() + " y=" + ev.y.toInt()
                + " mode=" + VoxModeController.current)
            audioEngine.bargeInClear()
            if (lastTouchTapAt != 0L && now - lastTouchTapAt <= 700L) {
                lastTouchTapAt = 0L
                Log.i("VoxKey", "TAP_TOUCH double -> toggle")
                powerPolicy?.toggleLiveStandby()
            } else {
                lastTouchTapAt = now
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    /** 长按镜腿 700ms 同样切模式（双击窗口不稳时的兜底） */
    override fun onKeyLongPress(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER || keyCode == KeyEvent.KEYCODE_ENTER
            || keyCode == KeyEvent.KEYCODE_HEADSETHOOK || keyCode == KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE) {
            Log.i("VoxKey", "LONG_PRESS toggle mode=" + VoxModeController.current)
            lastTapAt = 0L
            VoxModeController.toggleStandby()
            return true
        }
        return super.onKeyLongPress(keyCode, event)
    }

    // ── 本机遥测 / 控制 ───────────────────────────────────────

    private fun batterySnapshot(): Triple<Int, Boolean, Float> {
        return try {
            val i = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val lvl = i?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = i?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
            val status = i?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            val temp = (i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
            val pct = if (scale > 0) (lvl * 100 / scale) else -1
            val charging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
            Triple(pct, charging, temp)
        } catch (_: Throwable) {
            Triple(-1, false, 0f)
        }
    }

    private fun sendTelemetry() {
        try {
            val (pct, charging, temp) = batterySnapshot()
            val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val brightnessRaw = try {
                Settings.System.getInt(contentResolver, Settings.System.SCREEN_BRIGHTNESS, 0)
            } catch (_: Throwable) { 0 }
            var rssi = 0
            try {
                val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                rssi = wifi.connectionInfo.rssi
            } catch (_: Throwable) {}
            val memMb = try {
                val mi = ActivityManager.MemoryInfo()
                (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
                (mi.availMem / (1024 * 1024)).toInt()
            } catch (_: Throwable) { 0 }
            eyes.forEach { it.statusBar.setTelemetry(pct, charging, temp) }
            duplex.sendControlMessage(
                "DEVICE_TELEMETRY",
                mapOf(
                    "battery_pct" to pct,
                    "charging" to charging,
                    "temp_c" to Math.round(temp * 10f) / 10f,
                    "brightness" to (if (appliedBrightnessPct >= 0) appliedBrightnessPct
                                     else Math.round(brightnessRaw * 100f / 255f)),
                    "volume" to am.getStreamVolume(AudioManager.STREAM_MUSIC),
                    "volume_max" to am.getStreamMaxVolume(AudioManager.STREAM_MUSIC),
                    "wifi_rssi" to rssi,
                    "mode" to VoxModeController.current.name,
                    "mem_avail_mb" to memMb,
                    "lat" to (lastLoc?.first ?: 0.0),
                    "lon" to (lastLoc?.second ?: 0.0),
                    "acc" to (lastLoc?.third ?: 0.0),
                )
            )
        } catch (t: Throwable) {
            Log.w("VoxDevice", "telemetry failed: " + t.message)
        }
    }

    override fun onDeviceCommand(action: String, value: String) {
        Log.i("VoxDevice", "DEVICE_CMD " + action + "=" + value)
        mainHandler.post { applyDeviceCommand(action, value) }
    }

    private fun applyDeviceCommand(action: String, value: String) {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val maxVol = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        try {
            when (action) {
                "set_brightness", "brightness_up", "brightness_down" -> {
                    val curPct = if (appliedBrightnessPct >= 0) appliedBrightnessPct else Math.round(
                        Settings.System.getInt(contentResolver, Settings.System.SCREEN_BRIGHTNESS, 128) * 100f / 255f
                    )
                    val target = when (action) {
                        "set_brightness" -> value.toFloatOrNull()?.toInt() ?: curPct
                        "brightness_up" -> curPct + 10
                        else -> curPct - 10
                    }.coerceIn(0, 100)
                    // 1) 先写系统亮度（需要 WRITE_SETTINGS；失败也不影响窗口生效）
                    var sysOk = true
                    try {
                        sysOk = Settings.System.putInt(
                            contentResolver, Settings.System.SCREEN_BRIGHTNESS, target * 255 / 100
                        )
                    } catch (t: Throwable) {
                        sysOk = false
                        Log.w("VoxDevice", "BRIGHTNESS sys write denied: " + t.message)
                    }
                    // 2) 窗口亮度：无需权限，整屏只有本应用，视觉上等价且立即生效
                    window.attributes = window.attributes.apply { screenBrightness = target / 100f }
                    appliedBrightnessPct = target
                    Log.i("VoxDevice", "BRIGHTNESS -> " + target + " sys=" + sysOk)
                }
                "set_volume", "volume_up", "volume_down", "mute" -> {
                    when (action) {
                        "set_volume" -> {
                            val pct = (value.toFloatOrNull() ?: 50f).coerceIn(0f, 100f)
                            am.setStreamVolume(AudioManager.STREAM_MUSIC, (pct * maxVol / 100f).toInt(), 0)
                        }
                        "volume_up" -> am.adjustStreamVolume(
                            AudioManager.STREAM_MUSIC, AudioManager.ADJUST_RAISE, 0
                        )
                        "volume_down" -> am.adjustStreamVolume(
                            AudioManager.STREAM_MUSIC, AudioManager.ADJUST_LOWER, 0
                        )
                        "mute" -> am.setStreamVolume(AudioManager.STREAM_MUSIC, 0, 0)
                    }
                    Log.i("VoxDevice", "VOLUME -> " + am.getStreamVolume(AudioManager.STREAM_MUSIC) + "/" + maxVol)
                }
                "set_mode" -> {
                    val m = value.uppercase()
                    if (m in setOf("LIVE", "STANDBY", "SLEEP_LISTEN")) {
                        VoxModeController.setMode(VoxPowerMode.valueOf(m))
                    }
                }
                "set_mic_mode" -> audioEngine.setMicRoute(value)
                "take_photo" -> {
                    camera.lanMode = lanModeForCapture()
                    camera.captureOnce("manual")
                    Log.i("VoxDevice", "TAKE_PHOTO manual")
                }
                "open_app" -> {
                    val pm = packageManager
                    var pkg = value.trim()
                    if (pm.getLaunchIntentForPackage(pkg) == null) {
                        for (ai in pm.getInstalledApplications(0)) {
                            val label = pm.getApplicationLabel(ai).toString()
                            if (label.contains(value, true) || ai.packageName.contains(value, true)) {
                                pkg = ai.packageName
                                break
                            }
                        }
                    }
                    val it2 = pm.getLaunchIntentForPackage(pkg)
                    if (it2 != null) {
                        it2.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        startActivity(it2)
                        Log.i("VoxDevice", "OPEN_APP " + pkg)
                    } else {
                        Log.w("VoxDevice", "OPEN_APP not found: " + value)
                    }
                }
                else -> Log.w("VoxDevice", "unknown action " + action)
            }
        } catch (t: Throwable) {
            Log.w("VoxDevice", "cmd " + action + " failed: " + t.message)
        }
        // 执行后立刻回传最新遥测
        mainHandler.postDelayed({ sendTelemetry() }, 400L)
    }

    private fun lanModeForCapture(): Boolean = isLan

    private fun updateStatusText() {
        eyes.forEach { it.statusBar.setConnection(isConnected, isLan) }
    }

    // ── 证据卡 ────────────────────────────────────────────────

    private fun renderActiveEvidenceImage() {
        // 新证据卡自己管理多图轮播；这里只在方向键翻页时被调用
    }

    private fun clearEvidenceCard() {
        mainHandler.removeCallbacks(cycleImageTask)
        for ((bmp, _) in activeBitmaps) {
            if (!bmp.isRecycled) bmp.recycle()
        }
        activeBitmaps = emptyList()
        activeImageIdx = 0
        eyes.forEach {
            it.evidence.releaseImmediate()
            it.evidence.visibility = View.GONE
        }
    }

    // ── 传输层回调 ─────────────────────────────────────────────

    override fun onConnectionChanged(connected: Boolean, lanMode: Boolean) {
        mainHandler.post {
            isConnected = connected
            isLan = lanMode
            camera.lanMode = lanMode   // 出门自动降 JPEG 质量省上行
            updateStatusText()
            val mode = if (connected) WaveguideReticleView.Mode.IDLE else WaveguideReticleView.Mode.OFF
            eyes.forEach { it.reticle.setMode(mode) }
        }
    }

    override fun onAudioDown(epochId: Int, pcm24k: ByteArray) {
        audioEngine.feedDownlinkPcm(pcm24k)
    }

    override fun onBargeInClear(newEpochId: Int) {
        audioEngine.bargeInClear()
    }

    override fun onNavPatch(
        active: Boolean,
        instruction: String,
        waypoints: List<ArCompassBarView.Waypoint>
    ) {
        mainHandler.post {
            eyes.forEach { eye ->
                eye.compass.updateNav(active, instruction, waypoints)
                eye.compass.visibility = if (active) View.VISIBLE else View.GONE
                eye.statusBar.setCompactForNav(active)
            }
        }
    }

    override fun onDuckAudio(duck: Boolean) {
        audioEngine.setDucking(duck)
    }

    override fun onSpeechState(state: String) {
        mainHandler.post {
            val m = when (state) {
                "LISTENING" -> WaveguideReticleView.Mode.LISTENING
                "THINKING" -> WaveguideReticleView.Mode.THINKING
                "SPEAKING" -> WaveguideReticleView.Mode.SPEAKING
                else -> if (isConnected) WaveguideReticleView.Mode.IDLE else WaveguideReticleView.Mode.OFF
            }
            eyes.forEach { it.reticle.setMode(m) }
        }
    }

    override fun onTranscript(epochId: Int, role: String, text: String, isFinal: Boolean) {
        mainHandler.post {
            eyes.forEach { it.subtitle.showTranscript(role, text) }
        }
    }

    override fun onHudPatch(
        epochId: Int,
        header: String,
        lines: List<String>,
        images: List<VoxDuplexClient.HudImage>,
        ttlMs: Long
    ) {
        // 后台 WS 线程解码，避免卡 UI
        val decoded = images.mapNotNull { img ->
            val bmp = BitmapFactory.decodeByteArray(img.jpegBytes, 0, img.jpegBytes.size)
            if (bmp != null) bmp to img.label else null
        }
        mainHandler.post {
            mainHandler.removeCallbacks(hideEvidenceTask)
            mainHandler.removeCallbacks(cycleImageTask)
            for ((old, _) in activeBitmaps) {
                if (!old.isRecycled) old.recycle()
            }
            activeBitmaps = decoded
            activeImageIdx = 0

            eyes.forEach {
                it.evidence.showEvidence(header, lines, decoded, ttlMs)
                it.evidence.visibility = View.VISIBLE
            }
            if (activeBitmaps.size > 1) {
                mainHandler.postDelayed(cycleImageTask, CYCLE_MS)
            }
            mainHandler.postDelayed(hideEvidenceTask, ttlMs)
        }
    }
}
