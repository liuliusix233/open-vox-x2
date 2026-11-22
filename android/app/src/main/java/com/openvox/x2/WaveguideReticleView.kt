package com.openvox.x2

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Unified 60fps living 4-dot constellation reticle for RayNeo X2 optical waveguide.
 * Pure black (#000000) background = 100% transparent; ultra-low APL to avoid glare.
 *
 * All 5 states (OFF, IDLE, LISTENING, THINKING, SPEAKING) morph smoothly from the
 * exact same 4 luminous dots + soft aura with zero jarring shape snaps:
 *   - OFF       : dim compact resting dot
 *   - IDLE      : calm breathing single cyan-white core (4 dots converged)
 *   - LISTENING : classic 4-dot horizontal audio-reactive equalizer wave
 *   - THINKING  : 4-dot warm-amber orbital constellation with smooth trails
 *   - SPEAKING  : emerald-cyan pulsing voice core + expanding ripple driven by real TTS level
 */
class WaveguideReticleView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    enum class Mode { OFF, IDLE, LISTENING, THINKING, SPEAKING }

    private var targetMode = Mode.OFF
    private var micLevel = 0f
    private var ttsLevel = 0f

    // Continuous animation clock (radians)
    private var animTime = 0f
    private var lastFrameNs = 0L

    // Smoothly interpolated state weights [0..1] for seamless morphing
    private var wOff = 1f
    private var wIdle = 0f
    private var wListen = 0f
    private var wThink = 0f
    private var wSpeak = 0f

    // Smoothly interpolated ARGB color channels
    private var curR = 90f
    private var curG = 110f
    private var curB = 130f
    private var curA = 120f

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 1000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            val now = System.nanoTime()
            val dt = if (lastFrameNs == 0L) 0.016f else ((now - lastFrameNs) / 1e9f).coerceIn(0.004f, 0.05f)
            lastFrameNs = now
            stepPhysics(dt)
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        lastFrameNs = 0L
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    fun setMode(newMode: Mode) {
        targetMode = newMode
    }

    fun setMicLevel(level: Float) {
        val clamped = level.coerceIn(0f, 1f)
        // Fast attack, smooth release
        val alpha = if (clamped > micLevel) 0.45f else 0.18f
        micLevel += (clamped - micLevel) * alpha
    }

    /** Optional real-time downlink TTS RMS [0..1] so SPEAKING pulses with her actual voice. */
    fun setTtsLevel(level: Float) {
        val clamped = level.coerceIn(0f, 1f)
        val alpha = if (clamped > ttsLevel) 0.42f else 0.16f
        ttsLevel += (clamped - ttsLevel) * alpha
    }

    private fun stepPhysics(dt: Float) {
        animTime += dt
        val k = (dt * 7.5f).coerceIn(0.05f, 0.45f)

        wOff += ((if (targetMode == Mode.OFF) 1f else 0f) - wOff) * k
        wIdle += ((if (targetMode == Mode.IDLE) 1f else 0f) - wIdle) * k
        wListen += ((if (targetMode == Mode.LISTENING) 1f else 0f) - wListen) * k
        wThink += ((if (targetMode == Mode.THINKING) 1f else 0f) - wThink) * k
        wSpeak += ((if (targetMode == Mode.SPEAKING) 1f else 0f) - wSpeak) * k

        // Target RGB + Alpha per state
        val (tR, tG, tB, tA) = when (targetMode) {
            Mode.OFF -> Quad(85f, 105f, 125f, 110f)
            Mode.IDLE -> Quad(124f, 232f, 255f, 200f)      // Calm ice cyan
            Mode.LISTENING -> Quad(0f, 240f, 255f, 245f)   // Vivid electric cyan
            Mode.THINKING -> Quad(255f, 194f, 71f, 240f)   // Warm amber gold
            Mode.SPEAKING -> Quad(56f, 242f, 166f, 240f)   // Luminous emerald cyan
        }
        curR += (tR - curR) * k
        curG += (tG - curG) * k
        curB += (tB - curB) * k
        curA += (tA - curA) * k

        // Gentle decay on audio levels when idle
        if (targetMode != Mode.LISTENING) micLevel *= 0.88f
        if (targetMode != Mode.SPEAKING) ttsLevel *= 0.88f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val cx = width * 0.5f
        val cy = height * 0.5f
        val unit = minOf(width, height) * 0.055f // Compact optical footprint

        val mainColor = Color.argb(
            curA.toInt().coerceIn(0, 255),
            curR.toInt().coerceIn(0, 255),
            curG.toInt().coerceIn(0, 255),
            curB.toInt().coerceIn(0, 255)
        )

        // 1. Subtle ambient halo (visible in IDLE / LISTENING / SPEAKING)
        val speakVoice = if (ttsLevel > 0.03f) ttsLevel else (0.35f + 0.25f * sin(animTime * 6.5f))
        val haloAlpha = ((wIdle * 36f) + (wListen * (42f + micLevel * 70f)) + (wSpeak * (55f + speakVoice * 85f))).toInt()
        if (haloAlpha > 4) {
            val haloR = unit * (0.75f + wIdle * 0.12f * sin(animTime * 1.75f) + wListen * (0.35f + micLevel * 0.55f) + wSpeak * (0.35f + speakVoice * 0.65f))
            ringPaint.color = Color.argb(haloAlpha.coerceIn(0, 140), curR.toInt(), curG.toInt(), curB.toInt())
            ringPaint.strokeWidth = 2.0f + wSpeak * 1.2f
            canvas.drawCircle(cx, cy, haloR, ringPaint)
        }

        // 2. Soft expanding wave ring while SPEAKING
        if (wSpeak > 0.05f) {
            val wavePhase = (animTime * 0.95f) % 1f
            val waveR = unit * (0.7f + wavePhase * (1.15f + speakVoice * 0.5f))
            val waveA = ((1f - wavePhase) * wSpeak * 110f).toInt().coerceIn(0, 140)
            ringPaint.color = Color.argb(waveA, curR.toInt(), curG.toInt(), curB.toInt())
            ringPaint.strokeWidth = 1.6f
            canvas.drawCircle(cx, cy, waveR, ringPaint)
        }

        // 3. Unified 4-dot living constellation (every state blends smoothly from these 4 nodes)
        val baseDotR = unit * (0.14f + wOff * 0.02f + wIdle * 0.08f + wSpeak * (0.09f + speakVoice * 0.06f))
        val listenSpacing = unit * 0.54f
        val orbitR = unit * (0.72f + 0.10f * sin(animTime * 2.8f))
        val idleBreath = 1f + 0.14f * sin(animTime * 1.75f)

        for (i in 0 until 4) {
            val phaseOffset = i * (PI.toFloat() / 2f)

            // OFF / IDLE target: tightly converged breathing nucleus
            val idleX = cx + cos(animTime * 0.6f + phaseOffset) * (unit * 0.08f * idleBreath)
            val idleY = cy + sin(animTime * 0.6f + phaseOffset) * (unit * 0.08f * idleBreath)

            // LISTENING target: 4-dot horizontal equalizer row reacting to micLevel
            val listenX = cx + (i - 1.5f) * listenSpacing
            val waveBounce = sin(animTime * 7.5f + i * 0.95f) * (unit * (0.14f + micLevel * 0.42f))
            val listenY = cy + waveBounce

            // THINKING target: 4-dot orbital constellation
            val angle = animTime * 2.9f + phaseOffset
            val thinkX = cx + cos(angle) * orbitR
            val thinkY = cy + sin(angle) * orbitR

            // SPEAKING target: 4-dot diamond pulse synced to TTS voice energy
            val speakR = unit * (0.22f + speakVoice * 0.34f + 0.06f * sin(animTime * 8.0f + phaseOffset))
            val speakAngle = animTime * 1.1f + phaseOffset
            val speakX = cx + cos(speakAngle) * speakR
            val speakY = cy + sin(speakAngle) * speakR

            // Blend coordinates smoothly by state weights
            val x = (wOff + wIdle) * idleX + wListen * listenX + wThink * thinkX + wSpeak * speakX
            val y = (wOff + wIdle) * idleY + wListen * listenY + wThink * thinkY + wSpeak * speakY

            // In LISTENING, stretch dots gently into vertical voice pills when mic is active
            if (wListen > 0.15f && micLevel > 0.06f) {
                val barHalfH = wListen * unit * (0.10f + micLevel * 0.48f * (0.65f + 0.35f * sin(animTime * 9.0f + i * 1.1f)))
                barPaint.color = mainColor
                barPaint.strokeWidth = (baseDotR * 2.0f).coerceAtLeast(4f)
                canvas.drawLine(x, y - barHalfH, x, y + barHalfH, barPaint)
            } else {
                // Draw subtle comet trail dots during THINKING
                if (wThink > 0.15f) {
                    for (t in 1..2) {
                        val trailAngle = angle - t * 0.28f
                        val tx = cx + cos(trailAngle) * orbitR
                        val ty = cy + sin(trailAngle) * orbitR
                        val ta = (curA * wThink * (0.42f / t)).toInt().coerceIn(0, 120)
                        dotPaint.color = Color.argb(ta, curR.toInt(), curG.toInt(), curB.toInt())
                        canvas.drawCircle(tx, ty, baseDotR * (1f - t * 0.24f), dotPaint)
                    }
                }
                dotPaint.color = mainColor
                canvas.drawCircle(x, y, baseDotR * (1f + wIdle * 0.12f * sin(animTime * 1.75f)), dotPaint)
            }
        }
    }

    private data class Quad(val r: Float, val g: Float, val b: Float, val a: Float)
}
