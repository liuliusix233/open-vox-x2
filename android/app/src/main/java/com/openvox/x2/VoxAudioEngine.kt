package com.openvox.x2

import android.annotation.SuppressLint
import android.content.Context
import android.media.*
import android.media.AudioManager
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.os.SystemClock
import android.util.Log
import java.io.ByteArrayOutputStream
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Unified 16kHz uplink + 24kHz downlink audio engine for RayNeo X2 open-ear speakers.
 * Includes client-side playback echo mute (+450ms tail), widened 700ms VAD hangover,
 * lowered 85 RMS floor for X2 temple mics, 120ms prebuffer, and <10ms bargeInClear().
 */
class VoxAudioEngine(private val context: Context, private val duplex: VoxDuplexClient) {
    companion object {
        private const val TAG = "VoxAudio"
        private const val REC_RATE = 16000
        private const val FRAME_SAMPLES = 320        // 20ms @ 16kHz
        private const val PLAY_RATE = 24000
        private const val PREBUF_MIN_BYTES = 2880    // 60ms @ 24kHz PCM16LE（压低首字延迟）
        private const val PREBUF_MAX_BYTES = 7680    // 160ms adaptive cap（不再堆到 320ms）
        private const val FADE_IN_SAMPLES = 192      // 8ms forward linear fade-in
        private const val GAIN_BOOST = 2.8f          // 2026-11-22 提高响度；26000 knee 软限幅防破音
        private const val VAD_HANGOVER_FRAMES = 35   // 700ms hangover (aligned with server 900ms)
        private const val VAD_FLOOR_RMS = 85f        // Lowered for RayNeo X2 temple mic
        private const val MIC_GAIN = 2.0f            // 2026-11-22：开了 voiceassistant 录音路由后不再需要 12x 硬撑
        private val MIC_SOURCES = intArrayOf(
            MediaRecorder.AudioSource.CAMCORDER,        // 双镜腿麦，无固件降噪
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC
        )
        private const val IN_CHANNELS = 2               // CHANNEL_IN_STEREO：左右镜腿各一路
        private const val IN_CHANNEL_MASK = AudioFormat.CHANNEL_IN_STEREO
        private const val WD_ZERO_FRAMES = 150       // 连续 3s 全零 → 重建采音器
        private const val WD_MAX_REBUILDS = 5
        private const val ECHO_TAIL_MS = 450L        // Local acoustic tail guard after TTS stops
        private val DTX_HEARTBEAT = byteArrayOf(0, 0)
    }

    var onMicLevel: ((Float) -> Unit)? = null
    /** 下行 TTS 能量（0..1），给准星 SPEAKING 律动用 */
    var onTtsLevel: ((Float) -> Unit)? = null
    /** 大声说话打断成功（120ms 连续高能量判定） */
    var onVoiceBargeIn: (() -> Unit)? = null
    var onSpeechOnset: (() -> Unit)? = null
    var onPlaybackChanged: ((Boolean) -> Unit)? = null

    @Volatile private var running = false
    @Volatile private var clearRequested = false
    @Volatile private var isPlaying = false
    @Volatile private var muteUntilMs = 0L
    @Volatile private var prebufTarget = PREBUF_MIN_BYTES
    @Volatile private var lastWriteAt = 0L
    @Volatile private var currentPlayRms = 0f
    private var bargeStreak = 0

    private val playQueue = LinkedBlockingQueue<ByteArray>(400)
    private var track: AudioTrack? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private val windDsp = WindDsp()
    private var recThread: Thread? = null
    private var playThread: Thread? = null

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        // 2026-11-22 官方文档根因：X2 默认只开朝前的前置麦（闷、必须靠软件拉增益）。
        // 切到 voiceassistant 录音路由，走指向用户嘴部的那组麦。
        try {
            val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            am.setParameters("audio_source_record=camcorder")
            Log.i(TAG, "AUDIO_ROUTE audio_source_record=camcorder (stereo dual-temple)")
        } catch (t: Throwable) {
            Log.w(TAG, "AUDIO_ROUTE set failed: " + t.message)
        }
        running = true
        val minRecBuf = AudioRecord.getMinBufferSize(
            REC_RATE, IN_CHANNEL_MASK, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(REC_RATE * 2 * IN_CHANNELS)

        // 2026-11-22 设计决定（照旧客户端）：VOICE_COMMUNICATION + AEC/NS 把 X2 人声压到
        // rms 3~86（满量程 32767），VAD 当静音吞掉 → ASR 全是「嗯。」「啊。」。
        // 改回未处理的 VOICE_RECOGNITION 优先，并关闭两个效果器取原始信号。
        var picked: AudioRecord? = null
        var srcIdx = 0
        for (i in MIC_SOURCES.indices) {
            val cand = openRecorder(MIC_SOURCES[i], minRecBuf)
            if (cand != null) { picked = cand; srcIdx = i; break }
        }
        val recorder = picked ?: throw IllegalStateException("AudioRecord init failed for all sources")
        Log.i(TAG, "MIC_SRC idx=" + srcIdx + " src=" + MIC_SOURCES[srcIdx] + " minBuf=" + minRecBuf)
        val sessionId = recorder.audioSessionId
        Log.i(TAG, "AEC/NS disabled (level diagnosis, same as legacy client)")

        // 2026-11-22 响度：X2 开放扬声器在媒体总线上最响；不要复用 recorder 的 session
        // （高通 HAL 会按录音会话压低功放增益）。
        try {
            val amVol = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            Log.i(TAG, "VOL STREAM_MUSIC cur=" + amVol.getStreamVolume(AudioManager.STREAM_MUSIC)
                + "/" + amVol.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                + " usage=MEDIA gen_session=1")
        } catch (t: Throwable) {
            Log.w(TAG, "VOL read failed: " + t.message)
        }
        val minPlayBuf = AudioTrack.getMinBufferSize(
            PLAY_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(9600)
        track = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            AudioFormat.Builder()
                .setSampleRate(PLAY_RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build(),
            minPlayBuf,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE
        ).apply { play() }

        recThread = Thread({ recordLoop(recorder) }, "vox-rec").apply {
            priority = Thread.MAX_PRIORITY; start()
        }
        playThread = Thread(::playLoop, "vox-play").apply {
            priority = Thread.MAX_PRIORITY; start()
        }
        Log.i(TAG, "VoxAudioEngine started (aec=${aec != null}, ns=${ns != null})")
    }

    fun stop() {
        running = false
        bargeInClear()
        recThread?.interrupt()
        playThread?.interrupt()
        try { aec?.release() } catch (_: Throwable) {}
        try { ns?.release() } catch (_: Throwable) {}
        try { track?.stop(); track?.release() } catch (_: Throwable) {}
        track = null; aec = null; ns = null
        try {
            val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            am.setParameters("audio_source_record=off")
            Log.i(TAG, "AUDIO_ROUTE audio_source_record=off")
        } catch (t: Throwable) {
            Log.w(TAG, "AUDIO_ROUTE reset failed: " + t.message)
        }
    }

    fun feedDownlinkPcm(pcm24k: ByteArray) {
        if (running && pcm24k.isNotEmpty()) playQueue.offer(pcm24k)
    }

    /** 运行中切换录音路由：camcorder / voiceassistant / translation */
    fun setMicRoute(route: String) {
        val m = route.lowercase()
        if (m !in setOf("camcorder", "voiceassistant", "translation")) return
        try {
            val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            am.setParameters("audio_source_record=" + m)
            Log.i(TAG, "MIC_ROUTE -> " + m)
        } catch (t: Throwable) {
            Log.w(TAG, "MIC_ROUTE failed: " + t.message)
        }
    }

    fun bargeInClear(echoTail: Boolean = true) {
        clearRequested = true
        playQueue.clear()
        val t = track ?: return
        try {
            synchronized(t) {
                t.pause()
                t.flush()
                t.play()
            }
        } catch (_: Throwable) {}
        muteUntilMs = 0L
        setPlaying(false)
        if (!echoTail) muteUntilMs = 0L
    }

    fun setDucking(duck: Boolean) {
        try { track?.setVolume(if (duck) 0.25f else 1.0f) } catch (_: Throwable) {}
    }

    private fun setPlaying(active: Boolean) {
        if (isPlaying != active) {
            isPlaying = active
            if (!active) {
                setDucking(false)
                muteUntilMs = SystemClock.uptimeMillis() + ECHO_TAIL_MS
                currentPlayRms = 0f
                onTtsLevel?.invoke(0f)
            }
            onPlaybackChanged?.invoke(active)
        }
    }

    private fun recordLoop(initial: AudioRecord) {
        var rec = initial
        var zeroFrames = 0
        var rebuilds = 0
        val pcm = ShortArray(FRAME_SAMPLES)                 // 下混后的单声道
        val rawIn = ShortArray(FRAME_SAMPLES * IN_CHANNELS)  // 立体声原始（L,R 交错）
        val preRoll = Array(8) { ByteArray(FRAME_SAMPLES * 2) }
        var preIdx = 0
        var noiseFloor = 55f
        var hangover = 0
        var lastDtxMs = 0L
        var inSpeech = false
        var sentFrames = 0L

        try {
            rec.startRecording()
            while (running) {
                val readN = rec.read(rawIn, 0, FRAME_SAMPLES * IN_CHANNELS)
                if (readN <= 0) continue
                // 双镜腿 (L+R)/2 下混成 16k 单声道
                val n: Int
                if (IN_CHANNELS == 2) {
                    val frames = readN / 2
                    if (frames <= 0) continue
                    for (i in 0 until frames) {
                        val l = rawIn[2 * i].toInt()
                        val r = rawIn[2 * i + 1].toInt()
                        pcm[i] = ((l + r) / 2).toShort()
                    }
                    n = frames
                } else {
                    System.arraycopy(rawIn, 0, pcm, 0, readN)
                    n = readN
                }

                val now = SystemClock.uptimeMillis()
                var _sum0 = 0.0
                for (i in 0 until n) _sum0 += pcm[i].toDouble() * pcm[i].toDouble()
                val rmsNow = sqrt(_sum0 / maxOf(n, 1)).toFloat()

                // Hard local echo guard while open-ear TTS speakers are active (+450ms tail)
                if (isPlaying || now < muteUntilMs) {
                    // 大声打断：阈值 = max(2600, 当前 TTS 外放能量 * 0.45)，连续 6 帧（120ms）确认
                    if (isPlaying) {
                        val bargeThreshold = maxOf(2600f, currentPlayRms * 0.45f)
                        if (rmsNow > bargeThreshold) {
                            if (++bargeStreak >= 6) {
                                bargeStreak = 0
                                Log.i(TAG, "VOICE_BARGE_IN rms=" + rmsNow.toInt()
                                    + " thr=" + bargeThreshold.toInt() + " play=" + currentPlayRms.toInt())
                                bargeInClear(echoTail = false)
                                duplex.clientInterrupt("voice_barge_in")
                                onVoiceBargeIn?.invoke()
                            }
                        } else {
                            bargeStreak = 0
                        }
                    }
                    inSpeech = false
                    duplex.speechActive = false
                    hangover = 0
                    onMicLevel?.invoke(0f)
                    if (now - lastDtxMs >= 400L) {
                        lastDtxMs = now
                        duplex.enqueueAudioUp(DTX_HEARTBEAT)
                    }
                    continue
                }

                windDsp.process(pcm, n)

                // AudioRecord 卡死看门狗：连续 3s 全零 → 重建采音器
                var _zs = 0.0
                for (i in 0 until n) _zs += kotlin.math.abs(pcm[i].toInt()).toDouble()
                if (_zs / n < 0.5) {
                    zeroFrames++
                    if (zeroFrames >= WD_ZERO_FRAMES && rebuilds < WD_MAX_REBUILDS) {
                        zeroFrames = 0
                        rebuilds++
                        Log.w(TAG, "MIC_ZERO watchdog -> rebuild #" + rebuilds)
                        try { rec.stop(); rec.release() } catch (_: Throwable) {}
                        var again: AudioRecord? = null
                        for (src in MIC_SOURCES) {
                            again = openRecorder(src, AudioRecord.getMinBufferSize(
                                REC_RATE, IN_CHANNEL_MASK, AudioFormat.ENCODING_PCM_16BIT
                            ).coerceAtLeast(REC_RATE * 2 * IN_CHANNELS))
                            if (again != null) break
                        }
                        if (again != null) {
                            rec = again
                            try { rec.startRecording() } catch (_: Throwable) {}
                        } else {
                            Log.e(TAG, "MIC_REBUILD failed, giving up")
                            return
                        }
                    }
                } else zeroFrames = 0

                var sumSq = 0.0
                for (i in 0 until n) sumSq += pcm[i].toDouble() * pcm[i].toDouble()
                val rms = sqrt(sumSq / n).toFloat()
                onMicLevel?.invoke((rms / 2500f).coerceIn(0f, 1f))

                val threshold = (noiseFloor * 1.65f).coerceAtLeast(VAD_FLOOR_RMS)
                val active = rms > threshold
                if (!active) {
                    noiseFloor = (noiseFloor * 0.98f + rms * 0.02f).coerceIn(20f, 120f)
                }

                val bytes = shortsToLe(pcm, n)
                if (active) {
                    if (!inSpeech) {
                        inSpeech = true
                        duplex.speechActive = true
                        onSpeechOnset?.invoke()
                        for (k in 0 until 8) duplex.enqueueAudioUp(preRoll[(preIdx + k) % 8])
                    }
                    hangover = VAD_HANGOVER_FRAMES
                    duplex.enqueueAudioUp(bytes)
                    if (++sentFrames % 100L == 0L) Log.d(TAG, "MIC_UP frames=$sentFrames rms=${rms.toInt()}")
                } else if (hangover > 0) {
                    hangover--
                    duplex.enqueueAudioUp(bytes)
                    if (hangover == 0) { inSpeech = false; duplex.speechActive = false }
                } else {
                    System.arraycopy(bytes, 0, preRoll[preIdx], 0, bytes.size)
                    preIdx = (preIdx + 1) % 8
                    if (now - lastDtxMs >= 400L) {
                        lastDtxMs = now
                        duplex.enqueueAudioUp(DTX_HEARTBEAT)
                    }
                }
            }
        } finally {
            try { rec.stop(); rec.release() } catch (_: Throwable) {}
        }
    }

    private fun playLoop() {
        val preBuf = ByteArrayOutputStream(PREBUF_MAX_BYTES * 2)
        var needPrebuf = true
        var fadeIdx = 0
        var underruns = 0

        while (running) {
            try {
                if (clearRequested) {
                    clearRequested = false
                    preBuf.reset()
                    needPrebuf = true
                    fadeIdx = 0
                    underruns = 0
                    prebufTarget = PREBUF_MIN_BYTES
                }
                val chunk = playQueue.poll(120, TimeUnit.MILLISECONDS)
                if (chunk == null) {
                    if (preBuf.size() > 0) {
                        val tail = preBuf.toByteArray()
                        preBuf.reset()
                        fadeIdx = writeProcessed(tail, fadeIdx)
                    } else if (isPlaying) {
                        // 只有"数据还在流、突然断粮"才算 underrun；
                        // 句末正常播完（距上次写入 >150ms）不计入，避免锁死在 160ms 预缓冲。
                        val sinceWrite = SystemClock.uptimeMillis() - lastWriteAt
                        if (sinceWrite < 150L) {
                            underruns++
                            if (underruns >= 2) prebufTarget = PREBUF_MAX_BYTES
                        } else {
                            underruns = 0
                            prebufTarget = PREBUF_MIN_BYTES
                        }
                        needPrebuf = true
                        fadeIdx = 0
                        setPlaying(false)
                    }
                    continue
                }

                if (needPrebuf) {
                    preBuf.write(chunk)
                    if (preBuf.size() < prebufTarget) continue
                    needPrebuf = false
                    fadeIdx = 0
                    setPlaying(true)
                    val burst = preBuf.toByteArray()
                    preBuf.reset()
                    fadeIdx = writeProcessed(burst, fadeIdx)
                } else {
                    fadeIdx = writeProcessed(chunk, fadeIdx)
                }
            } catch (_: InterruptedException) {
                return
            }
        }
    }

    private fun writeProcessed(data: ByteArray, startFadeIdx: Int): Int {
        lastWriteAt = SystemClock.uptimeMillis()
        var _p = 0
        var _s = 0.0
        var _i = 0
        while (_i + 1 < data.size) {
            val _v = ((data[_i + 1].toInt() shl 8) or (data[_i].toInt() and 0xFF)).toShort().toDouble()
            _s += _v * _v
            _p++
            _i += 2
        }
        if (_p > 0) {
            val playRms = sqrt(_s / _p).toFloat()
            currentPlayRms = playRms
            onTtsLevel?.invoke((playRms / 6000f).coerceIn(0f, 1f))
        }
        var idx = startFadeIdx
        var off = 0
        while (off + 1 < data.size) {
            val s = ((data[off + 1].toInt() shl 8) or (data[off].toInt() and 0xFF)).toShort()
            var v = s * GAIN_BOOST
            val av = abs(v)
            if (av > 26000f) {
                val sign = if (v >= 0f) 1f else -1f
                v = sign * (26000f + (av - 26000f) * 0.15f)
            }
            if (idx < FADE_IN_SAMPLES) {
                v *= (idx.toFloat() / FADE_IN_SAMPLES.toFloat())
                idx++
            }
            val iv = v.toInt().coerceIn(-32768, 32767)
            data[off] = (iv and 0xFF).toByte()
            data[off + 1] = ((iv shr 8) and 0xFF).toByte()
            off += 2
        }
        val t = track ?: return idx
        synchronized(t) {
            if (!clearRequested) t.write(data, 0, data.size)
        }
        return idx
    }

    private fun openRecorder(source: Int, minBuf: Int): AudioRecord? {
        return try {
            val r = AudioRecord(
                source,
                REC_RATE,
                IN_CHANNEL_MASK,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuf
            )
            if (r.state != AudioRecord.STATE_INITIALIZED) {
                try { r.release() } catch (_: Throwable) {}
                null
            } else r
        } catch (t: Throwable) {
            Log.w(TAG, "openRecorder failed src=" + source + " : " + t.message)
            null
        }
    }

    private fun shortsToLe(src: ShortArray, n: Int): ByteArray {
        val out = ByteArray(n * 2)
        var j = 0
        for (i in 0 until n) {
            val v = (src[i] * MIC_GAIN).toInt().coerceIn(-32768, 32767)
            out[j++] = (v and 0xFF).toByte()
            out[j++] = ((v shr 8) and 0xFF).toByte()
        }
        return out
    }
}
