package com.openvox.x2

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.BatteryManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * 2026-11-22 XR2 压榨版抓帧：连续预览流 + YUV 直出 JPEG。
 *
 *  * Camera2 TEMPLATE_PREVIEW 1280×960 常连续出帧（3A 连续自动曝光，不再单拍抽风）
 *  * 在 Y 通道上算清晰度（Laplacian 方差）与帧间 MAD（运动量），不经过 Bitmap/WebP 二次转码
 *  * YUV_420_888 → NV21 → JPEG(q85) 一步直出，meta 带 fmt=jpeg / w / h / sharpness
 *  * 动态推流：静止 1fps，运动 2–3fps；每个时间窗只发该窗内最清晰的一帧
 *  * 温控：电池 >43°C 降到 1fps，<40°C 恢复
 */
class VoxCamera(
    private val context: Context,
    private val duplex: VoxDuplexClient,
    private val sensorGate: SensorGate
) {
    companion object {
        private const val TAG = "VoxCam"
        private const val TARGET_W = 1280
        private const val TARGET_H = 960
        private const val JPEG_Q = 82
        private const val THERMAL_HIGH_C = 43f
        private const val THERMAL_LOW_C = 40f
        private const val MAD_STILL = 3.0
        private const val MAD_MOVE = 8.0
    }

    private val camManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val uploadExec = Executors.newSingleThreadExecutor()
    private var camThread: HandlerThread? = null
    private var camHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var chosenW = TARGET_W
    private var chosenH = TARGET_H
    private var prevY: ByteArray? = null
    private var lastWindowStart = 0L
    private var lastSentAt = 0L
    private var lastTempCheck = 0L
    private var batteryTempC = Float.NaN
    private var thermalThrottled = false
    @Volatile private var forceSend = false
    // 2026-11-22：LAN 内 1920x1080 q82；出门（WAN/热点）自动降到 1280x720 q70 + 1-2fps
    /**
     * 2026-11-22 供电自适应：STREAM 连续流 / EVENT_30S 按需+30s一帧 / ARCHIVE_15S 每15s静默一帧
     * 切换会重建采集会话（分辨率/质量随档位变）。
     */
    @Volatile var captureMode: String = "STREAM"
        set(value) {
            if (field == value) return
            field = value
            Log.i(TAG, "CAPTURE_MODE -> " + value)
            if (camThread != null && !modeSwitching) {
                modeSwitching = true
                Thread {
                    try { stop(); Thread.sleep(250); start() }
                    catch (t: Throwable) { Log.w(TAG, "capture restart failed: " + t.message) }
                    finally { modeSwitching = false }
                }.start()
            }
        }
    @Volatile var lanMode: Boolean = true
        set(value) {
            if (field == value) return
            field = value
            Log.i(TAG, if (value) "MODE LAN 1920x1080 q" + JPEG_Q + " 1-3fps"
                          else "MODE WAN 1280x720 q70 1-2fps")
            // 采集分辨率在 session 建立时固定，切网后重建一次相机（只在切换瞬间闪一下）
            if (camThread != null && !modeSwitching) {
                modeSwitching = true
                Thread {
                    try {
                        stop()
                        Thread.sleep(250)
                        start()
                    } catch (t: Throwable) {
                        Log.w(TAG, "mode restart failed: " + t.message)
                    } finally {
                        modeSwitching = false
                    }
                }.start()
            }
        }
    @Volatile private var modeSwitching = false
    private var frameCount = 0L

    // 当前时间窗内最清晰的一帧
    private var bestNv21: ByteArray? = null
    private var bestSharp = -1.0
    private var bestW = 0
    private var bestH = 0
    private var bestReason = "stream"

    @SuppressLint("MissingPermission")
    fun start() {
        if (camThread != null) return
        camThread = HandlerThread("vox-cam").apply { start() }
        camHandler = Handler(camThread!!.looper)
        try {
            val camId = camManager.cameraIdList.firstOrNull() ?: return
            val chars = camManager.getCameraCharacteristics(camId)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
            val sizes = map.getOutputSizes(ImageFormat.YUV_420_888) ?: return
            Log.i(TAG, "YUV sizes=" + sizes.joinToString(",") { it.width.toString() + "x" + it.height })
            // 本机 YUV_420_888 实测没有 1280x960：2304x1728, 1920x1080, 1280x720, 960x720 …
            // 取「像素最多且不超过 1920 宽」的那档（=1920x1080），拿不到再退回 4:3 的 960x720。
            val picked = when (captureMode) {
                "EVENT_30S" -> sizes.filter { it.width <= 1280 && it.width >= 960 }
                    .maxByOrNull { it.width * it.height }
                "ARCHIVE_15S" -> sizes.filter { it.width <= 1280 && it.width >= 960 }
                    .maxByOrNull { it.width * it.height }
                else -> if (lanMode) {
                    sizes.filter { it.width <= 1920 && it.width >= 1280 }
                        .maxByOrNull { it.width * it.height }
                } else {
                    sizes.filter { it.width <= 1280 && it.width >= 960 }
                        .maxByOrNull { it.width * it.height }
                }
            }
                ?: sizes.filter { abs(it.width.toFloat() / it.height - 4f / 3f) < 0.09f }
                    .minByOrNull { abs(it.width - TARGET_W) }
                ?: sizes.minByOrNull { abs(it.width - TARGET_W) }
                ?: return
            chosenW = picked.width
            chosenH = picked.height
            Log.i(TAG, "Selected camera size: " + chosenW + "x" + chosenH + " (preview stream)")

            imageReader = ImageReader.newInstance(chosenW, chosenH, ImageFormat.YUV_420_888, 3).apply {
                setOnImageAvailableListener({ reader -> onFrame(reader) }, camHandler)
            }

            camManager.openCamera(camId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    val surface = imageReader?.surface ?: return
                    @Suppress("DEPRECATION")
                    camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            captureSession = session
                            startPreview(session, surface)
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            Log.e(TAG, "Camera session configure failed")
                        }
                    }, camHandler)
                }

                override fun onDisconnected(camera: CameraDevice) { camera.close() }
                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera error " + error)
                    camera.close()
                }
            }, camHandler)
        } catch (t: Throwable) {
            Log.w(TAG, "Camera init skipped: " + t.message)
        }
    }

    private fun startPreview(session: CameraCaptureSession, surface: android.view.Surface) {
        try {
            val dev = cameraDevice ?: return
            val req = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            }
            session.setRepeatingRequest(req.build(), null, camHandler)
            Log.i(TAG, "PREVIEW repeating request started")
        } catch (t: Throwable) {
            Log.w(TAG, "startPreview failed: " + t.message)
        }
    }

    /** 兼容旧调用：VAD 起音 / 注视锁定时立刻发一帧 */
    fun captureOnce(reason: String) {
        bestReason = reason
        forceSend = true
    }

    fun stop() {
        try { captureSession?.stopRepeating() } catch (_: Throwable) {}
        try { captureSession?.close() } catch (_: Throwable) {}
        try { cameraDevice?.close() } catch (_: Throwable) {}
        try { imageReader?.close() } catch (_: Throwable) {}
        camThread?.quitSafely()
        camThread = null
    }

    // ── 每帧处理 ───────────────────────────────────────────────

    private fun onFrame(reader: ImageReader) {
        val img = reader.acquireLatestImage() ?: return
        try {
            frameCount++
            val w = img.width
            val h = img.height
            val yPlane = img.planes[0]
            val yBuf = yPlane.buffer
            val rowStride = yPlane.rowStride
            val pixelStride = yPlane.pixelStride

            // 采样 Y（每 4 像素取一个）做清晰度 + 运动量
            val sw = w / 4
            val sh = h / 4
            val sample = ByteArray(sw * sh)
            val base = yBuf.position()
            var idx = 0
            var yy = 0
            while (yy < sh) {
                val rowOff = base + (yy * 4) * rowStride
                var xx = 0
                while (xx < sw) {
                    val off = rowOff + (xx * 4) * pixelStride
                    sample[idx++] = if (off < yBuf.limit()) yBuf.get(off) else 0
                    xx++
                }
                yy++
            }

            val sharp = laplacianVar(sample, sw, sh)
            val mad = prevY?.let { mad(sample, it) } ?: 99.0
            prevY = sample

            val now = SystemClock.uptimeMillis()
            if (now - lastTempCheck > 5000) {
                lastTempCheck = now
                readBatteryTemp()
            }

            val fps = currentFps(mad)
            val windowMs = when (captureMode) {
                "EVENT_30S" -> 30_000L
                "ARCHIVE_15S" -> 15_000L
                else -> (1000 / fps).toLong()
            }
            // 按需档：只在开口/强制/到点时才发
            if (captureMode != "STREAM" && !forcedNow() && bestReason != "vad") {
                val due = (now - lastSentAt) >= windowMs
                if (!due) {
                    bestNv21 = null; bestSharp = -1.0; bestReason = "stream"
                    return
                }
            }

            val isVad = bestReason == "vad"
            if (forcedNow() || isVad || sharp > bestSharp) {
                val nv21 = yuvToNv21(img)
                if (nv21 != null) {
                    bestNv21 = nv21
                    bestSharp = sharp
                    bestW = w
                    bestH = h
                }
            }

            if (forcedNow() || now - lastWindowStart >= windowMs) {
                val payload = bestNv21
                if (payload != null) {
                    val reason = bestReason
                    val pSharp = bestSharp
                    val pw = bestW
                    val ph = bestH
                    bestNv21 = null
                    bestSharp = -1.0
                    bestReason = "stream"
                    lastWindowStart = now
                    lastSentAt = now
                    forceSend = false
                    uploadExec.execute { encodeAndSend(payload, pw, ph, reason, pSharp) }
                } else {
                    lastWindowStart = now
                    forceSend = false
                }
                if (frameCount % 30L == 0L) {
                    Log.d(TAG, "STREAM frames=" + frameCount + " fps=" + fps + " mad=" + String.format("%.2f", mad)
                        + " sharp=" + String.format("%.1f", sharp) + " temp=" + batteryTempC)
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "onFrame error: " + t.message)
        } finally {
            img.close()
        }
    }

    private fun forcedNow(): Boolean = forceSend

    private fun currentFps(mad: Double): Int {
        if (thermalThrottled) return 1
        val base = when {
            mad < MAD_STILL -> 1
            mad < MAD_MOVE -> 2
            else -> 3
        }
        // 公网/热点：限制在 1-2fps，省上行保语音与 TTS 零卡顿
        return if (lanMode) base else minOf(base, 2)
    }

    private fun readBatteryTemp() {
        try {
            val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val t = i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
            if (t > 0) {
                batteryTempC = t / 10f
                if (!thermalThrottled && batteryTempC > THERMAL_HIGH_C) {
                    thermalThrottled = true
                    Log.w(TAG, "THERMAL throttle ON at " + batteryTempC + "C -> 1fps")
                } else if (thermalThrottled && batteryTempC < THERMAL_LOW_C) {
                    thermalThrottled = false
                    Log.i(TAG, "THERMAL throttle OFF at " + batteryTempC + "C -> 2-3fps")
                }
            }
        } catch (t: Throwable) {
            Log.d(TAG, "temp read failed: " + t.message)
        }
    }

    private fun laplacianVar(a: ByteArray, w: Int, h: Int): Double {
        if (w < 3 || h < 3) return 0.0
        var sum = 0.0
        var sumSq = 0.0
        var n = 0
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i = row + x
                val v = (-4 * (a[i].toInt() and 0xFF) + (a[i - 1].toInt() and 0xFF) + (a[i + 1].toInt() and 0xFF)
                    + (a[i - w].toInt() and 0xFF) + (a[i + w].toInt() and 0xFF)).toDouble()
                sum += v
                sumSq += v * v
                n++
            }
        }
        if (n == 0) return 0.0
        val mean = sum / n
        return sumSq / n - mean * mean
    }

    private fun mad(a: ByteArray, b: ByteArray): Double {
        val n = minOf(a.size, b.size)
        if (n == 0) return 99.0
        var s = 0.0
        for (i in 0 until n) s += abs((a[i].toInt() and 0xFF) - (b[i].toInt() and 0xFF))
        return s / n
    }

    private fun encodeAndSend(nv21: ByteArray, w: Int, h: Int, reason: String, sharp: Double) {
        try {
            val out = ByteArrayOutputStream()
            YuvImage(nv21, ImageFormat.NV21, w, h, null)
                .compressToJpeg(Rect(0, 0, w, h), when (captureMode) {
                    "EVENT_30S" -> 75
                    "ARCHIVE_15S" -> 72
                    else -> if (lanMode) JPEG_Q else 70
                }, out)
            val bytes = out.toByteArray()
            val meta = JSONObject()
                .put("reason", reason)
                .put("w", w)
                .put("h", h)
                .put("fmt", "jpeg")
                .put("sharpness", Math.round(sharp * 10.0) / 10.0)
            sensorGate.cachedGeo?.let {
                meta.put("lat", it.lat).put("lon", it.lon).put("acc", it.acc)
            }
            duplex.sendVisionFrame(bytes, meta.toString())
            Log.d(TAG, "VISION_UP reason=" + reason + " " + w + "x" + h + " bytes=" + bytes.size
                + " sharp=" + String.format("%.1f", sharp))
        } catch (t: Throwable) {
            Log.w(TAG, "encodeAndSend error: " + t.message)
        }
    }

    private fun yuvToNv21(image: Image): ByteArray? {
        return try {
            val yBuf = image.planes[0].buffer
            val uBuf = image.planes[1].buffer
            val vBuf = image.planes[2].buffer
            val ySize = yBuf.remaining()
            val uSize = uBuf.remaining()
            val vSize = vBuf.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuf.get(nv21, 0, ySize)
            vBuf.get(nv21, ySize, vSize)
            uBuf.get(nv21, ySize + vSize, uSize)
            nv21
        } catch (t: Throwable) {
            Log.w(TAG, "yuvToNv21 failed: " + t.message)
            null
        }
    }
}
