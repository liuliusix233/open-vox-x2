package com.openvox.x2

import android.util.Base64
import android.util.Log
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Single full-duplex binary WebSocket transport matching vox_v2 (/ws/v1/duplex).
 * Header: [1B FrameType][4B EpochID big-endian] + Payload.
 * Supports evidence photo thumbnails embedded in FT_CONTROL HUD_PATCH messages.
 */
class VoxDuplexClient(
    private val config: VoxConfig,
    private val listener: Listener
) {
    companion object {
        private const val TAG = "VoxDuplex"
        const val FT_AUDIO_UP: Byte = 0x01
        const val FT_AUDIO_DOWN: Byte = 0x02
        const val FT_VISION_UP: Byte = 0x03
        const val FT_CONTROL: Byte = 0x04
    }

    data class HudImage(
        val jpegBytes: ByteArray,
        val label: String
    )

    interface Listener {
        fun onConnectionChanged(connected: Boolean, lanMode: Boolean)
        fun onAudioDown(epochId: Int, pcm24k: ByteArray)
        fun onBargeInClear(newEpochId: Int)
        fun onDuckAudio(duck: Boolean)
        fun onSpeechState(state: String)

        /** 服务端下发的本机控制指令（DEVICE_CMD） */
        fun onDeviceCommand(action: String, value: String) {}

        /** 访客模式开关（GUEST_MODE） */
        fun onGuestMode(active: Boolean) {}
        fun onTranscript(epochId: Int, role: String, text: String, isFinal: Boolean)
        fun onNavPatch(
            active: Boolean,
            instruction: String,
            waypoints: List<ArCompassBarView.Waypoint>
        )

        fun onHudPatch(
            epochId: Int,
            header: String,
            lines: List<String>,
            images: List<HudImage>,
            ttlMs: Long
        )
    }

    private val epochRef = AtomicInteger(0)
    val activeEpochId: Int get() = epochRef.get()

    private val okClient = OkHttpClient.Builder()
        .connectTimeout(2500, TimeUnit.MILLISECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    @Volatile private var ws: WebSocket? = null
    /** 2026-11-22：用户正在说话时，视觉帧一律让路（音频绝对优先）。由 AudioEngine 维护。 */
    @Volatile var speechActive: Boolean = false
    private var consecutiveFailures = 0
    @Volatile private var closed = false
    private val audioTxQueue = ArrayBlockingQueue<ByteArray>(50)
    private var txThread: Thread? = null

    val isClosed: Boolean get() = closed

    fun connect() {
        closed = false
        if (txThread == null || !txThread!!.isAlive) {
            txThread = Thread(::txLoop, "vox-ws-tx").apply { isDaemon = true; start() }
        }
        spawnConnect(0L)
    }

    fun close() {
        closed = true
        txThread?.interrupt()
        ws?.close(1000, "bye")
        ws = null
    }

    fun enqueueAudioUp(pcm16k20ms: ByteArray) {
        val frame = packFrame(FT_AUDIO_UP, epochRef.get(), pcm16k20ms)
        if (!audioTxQueue.offer(frame)) {
            audioTxQueue.poll()
            audioTxQueue.offer(frame)
        }
    }

    fun sendVisionFrame(webpBytes: ByteArray, metaJson: String) {
        val socket = ws ?: return
        // 说话期间不推图：一张 1080p 帧 ~250KB，塞进 WS 队列会把麦克风帧挤丢，
        // 实测直接导致"图书馆不去了"被切成半句。
        if (speechActive) return
        if (socket.queueSize() > 131_072L) return
        val meta = metaJson.toByteArray(Charsets.UTF_8)
        val payload = ByteBuffer.allocate(2 + meta.size + webpBytes.size)
            .order(ByteOrder.BIG_ENDIAN)
            .putShort(meta.size.toShort())
            .put(meta)
            .put(webpBytes)
            .array()
        socket.send(packFrame(FT_VISION_UP, epochRef.get(), payload).toByteString())
    }

    fun sendControl(type: String, extra: Map<String, Any?> = emptyMap()) {
        val socket = ws ?: return
        val obj = JSONObject().put("type", type.uppercase()).put("epoch_id", epochRef.get())
        for ((k, v) in extra) obj.put(k, v)
        socket.send(packFrame(FT_CONTROL, epochRef.get(), obj.toString().toByteArray()).toByteString())
    }

    /** 通用控制帧（遥测 / 命令回执） */
    fun sendControlMessage(type: String, fields: Map<String, Any>) {
        try {
            sendControl(type, fields)
        } catch (t: Throwable) {
            Log.w(TAG, "CONTROL " + type + " failed: " + t.message)
        }
    }

    /** 三模式切换上报（服务端据此门控心跳 / 防社死） */
    fun sendModeChange(mode: String) {
        try {
            sendControl("MODE_CHANGE", mapOf("mode" to mode.uppercase()))
            Log.i(TAG, "MODE_CHANGE -> " + mode.uppercase())
        } catch (t: Throwable) {
            Log.w(TAG, "MODE_CHANGE failed: " + t.message)
        }
    }

    fun clientInterrupt(reason: String = "touch_cancel") {
        sendControl("CLIENT_INTERRUPT", mapOf("reason" to reason))
    }

    private fun txLoop() {
        while (!closed) {
            try {
                val frame = audioTxQueue.take()
                val s = ws
                if (s != null && s.queueSize() < 2_097_152L) {
                    s.send(frame.toByteString())
                }
            } catch (_: InterruptedException) {
                return
            }
        }
    }

    private fun spawnConnect(delayMs: Long) {
        Thread({
            if (delayMs > 0) {
                try { Thread.sleep(delayMs) } catch (_: InterruptedException) { return@Thread }
            }
            if (closed) return@Thread
            // 2026-11-22：前 2 次死磕内网（家里主路径），第 3 次起直接切公网，
            // 出门开热点时不再耗十几秒。
            val ep = if (consecutiveFailures < 2) config.pickBestEndpoint(forceLan = true)
                     else config.pickBestEndpoint()
            val req = Request.Builder().url(ep.wsUrl).apply {
                if (config.token.isNotEmpty()) header("X-Vox-Token", config.token)
            }.build()

            ws = okClient.newWebSocket(req, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    consecutiveFailures = 0
                    Log.i(TAG, "WS connected lan=${ep.isLan} url=${ep.wsUrl}")
                    sendControl("CLIENT_HELLO", mapOf(
                        "client" to config.clientId,
                        "fw" to BuildConfig.VERSION_NAME
                    ))
                    listener.onConnectionChanged(true, ep.isLan)
                }

                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    if (bytes.size < 5) return
                    val arr = bytes.toByteArray()
                    val bb = ByteBuffer.wrap(arr, 0, 5).order(ByteOrder.BIG_ENDIAN)
                    val ftype = bb.get()
                    val fepoch = bb.int
                    while (true) {
                        val cur = epochRef.get()
                        if (fepoch <= cur || epochRef.compareAndSet(cur, fepoch)) break
                    }
                    when (ftype) {
                        FT_AUDIO_DOWN -> {
                            if (fepoch < epochRef.get()) return
                            listener.onAudioDown(fepoch, arr.copyOfRange(5, arr.size))
                        }
                        FT_CONTROL -> {
                            handleControl(fepoch, String(arr, 5, arr.size - 5, Charsets.UTF_8))
                        }
                    }
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    consecutiveFailures++
                    Log.w(TAG, "WS failure(${consecutiveFailures}): ${t.message}")
                    listener.onConnectionChanged(false, false)
                    if (!closed) spawnConnect(1200L)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    listener.onConnectionChanged(false, false)
                    if (!closed) spawnConnect(1500L)
                }
            })
        }, "vox-ws-conn").apply { isDaemon = true; start() }
    }

    private fun handleControl(fepoch: Int, jsonStr: String) {
        val obj = try { JSONObject(jsonStr) } catch (_: Exception) { return }
        when (obj.optString("type").uppercase()) {
            "BARGE_IN_CLEAR" -> {
                epochRef.set(fepoch)
                listener.onBargeInClear(fepoch)
            }
            "DUCK_AUDIO" -> listener.onDuckAudio(obj.optBoolean("duck", false))
            "SPEECH_STATE" -> listener.onSpeechState(obj.optString("state", "IDLE").uppercase())
            "GUEST_MODE" -> listener.onGuestMode(obj.optBoolean("active", false))
            "DEVICE_CMD" -> listener.onDeviceCommand(
                obj.optString("action", ""), obj.optString("value", "")
            )
            "TRANSCRIPT" -> {
                if (fepoch < epochRef.get()) return
                listener.onTranscript(
                    fepoch,
                    obj.optString("role", "assistant"),
                    obj.optString("text", ""),
                    obj.optBoolean("final", false)
                )
            }
            "NAV_PATCH" -> {
                val active = obj.optBoolean("active", false)
                val instr = obj.optString("next_instruction", "")
                val list = mutableListOf<ArCompassBarView.Waypoint>()
                val arr = obj.optJSONArray("waypoints")
                if (arr != null) {
                    for (i in 0 until arr.length()) {
                        val o = arr.optJSONObject(i) ?: continue
                        list.add(
                            ArCompassBarView.Waypoint(
                                label = o.optString("label", ""),
                                kind = o.optString("kind", "turn"),
                                bearingDeg = o.optDouble("bearing_deg", 0.0).toFloat(),
                                distM = o.optInt("dist_m", 0)
                            )
                        )
                    }
                }
                listener.onNavPatch(active, instr, list)
            }
            "HUD_PATCH" -> {
                if (fepoch < epochRef.get()) return
                val header = obj.optString("header", "MEMORY")
                val lines = mutableListOf<String>()
                val arr = obj.optJSONArray("lines")
                if (arr != null) {
                    for (i in 0 until arr.length()) lines.add(arr.optString(i))
                }
                val images = parseHudImages(obj)
                val ttlMs = obj.optLong("ttl_ms", if (images.isNotEmpty()) 12_000L else 8_500L)
                    .coerceIn(3_000L, 30_000L)
                if (lines.isNotEmpty() || images.isNotEmpty()) {
                    Log.i(TAG, "HUD_PATCH header=$header lines=${lines.size} imgs=${images.size}")
                    listener.onHudPatch(fepoch, header, lines, images, ttlMs)
                }
            }
        }
    }

    private fun parseHudImages(obj: JSONObject): List<HudImage> {
        val out = mutableListOf<HudImage>()
        val imgArr = obj.optJSONArray("images")
        if (imgArr != null) {
            for (i in 0 until imgArr.length()) {
                val item = imgArr.optJSONObject(i) ?: continue
                val b64 = item.optString("b64", "")
                if (b64.isEmpty()) continue
                val raw = try { Base64.decode(b64, Base64.DEFAULT) } catch (_: Throwable) { continue }
                if (raw.isNotEmpty()) {
                    out.add(HudImage(raw, item.optString("label", "")))
                }
            }
        }
        if (out.isEmpty()) {
            val single = obj.optString("image_b64", "")
            if (single.isNotEmpty()) {
                val raw = try { Base64.decode(single, Base64.DEFAULT) } catch (_: Throwable) { null }
                if (raw != null && raw.isNotEmpty()) {
                    out.add(HudImage(raw, obj.optString("image_label", "")))
                }
            }
        }
        return out
    }

    private fun packFrame(type: Byte, epoch: Int, payload: ByteArray): ByteArray {
        return ByteBuffer.allocate(5 + payload.size)
            .order(ByteOrder.BIG_ENDIAN)
            .put(type)
            .putInt(epoch)
            .put(payload)
            .array()
    }
}
