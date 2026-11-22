package com.openvox.x2

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Parafoveal "Light-Wing" automotive driving HUD for RayNeo X2 binocular waveguide.
 *
 * Optical ergonomics (640x480 per eye, 25 deg FOV):
 *   - Upper 62% of FOV is 100% pure black (transparent) so the road ahead is unobstructed.
 *   - Telemetry sits in the lower parafoveal zone (y = 66%..79%, centered at x = 50%),
 *     readable purely in peripheral vision without glancing down-left at the dashboard:
 *       1. Symmetrical expanding RPM light wings (center -> outward, cyan -> amber -> redline)
 *       2. Centered night-friendly monospace speed (km/h) + tucked gear badge
 *       3. Silent-unless-warning vitals (coolant/voltage stay hidden while normal; amber alert only on fault)
 */
class VoxDriveHudView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class Telemetry(
        val speedKmh: Float = 0f,
        val rpm: Float = 0f,
        val gear: String = "",
        val coolantC: Float? = null,
        val volt: Float? = null,
        val speedLimitKmh: Float? = null
    )

    companion object {
        private const val MAX_RPM = 7000f
        private const val STALE_MS = 2500L
        private const val WING_SEGMENTS_PER_SIDE = 14
        private const val COOLANT_WARN_C = 102f
        private const val VOLT_WARN_V = 12.2f
    }

    private var targetSpeed = 0f
    private var displaySpeed = 0f
    private var targetRpm = 0f
    private var displayRpm = 0f
    private var gearText = ""
    private var coolantC: Float? = null
    private var volt: Float? = null
    private var speedLimitKmh: Float? = null
    private var lastRxMs = 0L

    private val speedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        textSize = 54f
        color = 0xFFE8FBFF.toInt()
        textAlign = Paint.Align.CENTER
    }
    private val unitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.MONOSPACE
        textSize = 10.5f
        color = 0x997CE8FF.toInt()
        textAlign = Paint.Align.LEFT
    }
    private val gearPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        textSize = 15.5f
        color = 0xFF00F0FF.toInt()
        textAlign = Paint.Align.CENTER
    }
    private val gearBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.4f
        color = 0xAA00F0FF.toInt()
    }
    private val warnPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        textSize = 11.5f
        color = 0xFFFFC247.toInt()
        textAlign = Paint.Align.CENTER
    }
    private val segPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val centerTickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x7700E5FF
    }
    private val rectF = RectF()

    private val smoothAnim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 1000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            if (visibility != VISIBLE) return@addUpdateListener
            val prevSpd = displaySpeed
            val prevRpm = displayRpm
            displaySpeed += (targetSpeed - displaySpeed) * 0.28f
            displayRpm += (targetRpm - displayRpm) * 0.24f
            val redlining = displayRpm >= MAX_RPM * 0.80f
            val staleNow = lastRxMs > 0L && (SystemClock.uptimeMillis() - lastRxMs) > STALE_MS
            if (abs(displaySpeed - prevSpd) > 0.05f ||
                abs(displayRpm - prevRpm) > 5f ||
                redlining ||
                staleNow
            ) {
                invalidate()
            }
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        smoothAnim.start()
    }

    override fun onDetachedFromWindow() {
        smoothAnim.cancel()
        super.onDetachedFromWindow()
    }

    fun updateTelemetry(t: Telemetry) {
        targetSpeed = t.speedKmh.coerceIn(0f, 320f)
        targetRpm = t.rpm.coerceIn(0f, MAX_RPM)
        gearText = t.gear.trim().uppercase().take(3)
        coolantC = t.coolantC
        volt = t.volt
        speedLimitKmh = t.speedLimitKmh
        lastRxMs = SystemClock.uptimeMillis()
        invalidate()
    }

    fun resetTelemetry() {
        targetSpeed = 0f
        displaySpeed = 0f
        targetRpm = 0f
        displayRpm = 0f
        gearText = ""
        coolantC = null
        volt = null
        speedLimitKmh = null
        lastRxMs = 0L
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val now = SystemClock.uptimeMillis()
        val isFresh = lastRxMs > 0L && (now - lastRxMs) <= STALE_MS
        val alphaScale = if (isFresh) 1f else 0.40f

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w * 0.5f

        // 1. Parafoveal Symmetrical RPM Light Wings (y = 66% of height, expands outward from center)
        val wingY = h * 0.665f
        val halfSpan = (w * 0.24f).coerceIn(96f, 165f)
        val centerGap = 6f
        val segGap = 2.6f
        val segW = (halfSpan - centerGap - segGap * (WING_SEGMENTS_PER_SIDE - 1)) / WING_SEGMENTS_PER_SIDE
        val rpmRatio = (displayRpm / MAX_RPM).coerceIn(0f, 1f)
        val litPerSide = (rpmRatio * WING_SEGMENTS_PER_SIDE).roundToInt().coerceIn(0, WING_SEGMENTS_PER_SIDE)

        // Subtle pulse when pushing near redline (>= 80% RPM)
        val redlinePulse = if (rpmRatio >= 0.80f && isFresh) {
            (0.75f + 0.25f * sin(now * 0.018).toFloat())
        } else {
            1f
        }

        // Tiny center anchor dot between left & right light wings
        centerTickPaint.alpha = (110 * alphaScale).toInt()
        canvas.drawCircle(cx, wingY, 1.8f, centerTickPaint)

        for (i in 0 until WING_SEGMENTS_PER_SIDE) {
            val frac = (i + 1).toFloat() / WING_SEGMENTS_PER_SIDE
            val isLit = i < litPerSide && lastRxMs > 0L
            // Gentle taper: thin near center (3.4px), slightly bolder toward redline tips (5.4px)
            val segH = 3.4f + 2.0f * frac
            val halfH = segH * 0.5f

            val color = when {
                frac >= 0.80f -> 0xFFFF4D4F.toInt() // Redline
                frac >= 0.55f -> 0xFFFFC247.toInt() // High RPM amber
                else -> 0xFF00E5FF.toInt()           // Cruise cyan
            }
            segPaint.color = color
            segPaint.alpha = if (isLit) {
                (225 * alphaScale * (if (frac >= 0.80f) redlinePulse else 1f)).toInt().coerceIn(0, 255)
            } else {
                (26 * alphaScale).toInt()
            }

            val offset = centerGap + i * (segW + segGap)
            // Right wing
            rectF.set(cx + offset, wingY - halfH, cx + offset + segW, wingY + halfH)
            canvas.drawRoundRect(rectF, 1.5f, 1.5f, segPaint)
            // Left wing (mirrored)
            rectF.set(cx - offset - segW, wingY - halfH, cx - offset, wingY + halfH)
            canvas.drawRoundRect(rectF, 1.5f, 1.5f, segPaint)
        }

        // 2. Centered Digital Speed + Tucked Gear Badge (y = 78.5% of height)
        val speedBaselineY = h * 0.79f
        val spdInt = displaySpeed.roundToInt()
        val overLimit = (speedLimitKmh != null && displaySpeed > speedLimitKmh!! + 5f) || displaySpeed > 120f
        speedPaint.color = if (overLimit) 0xFFFFC247.toInt() else 0xFFE8FBFF.toInt()
        speedPaint.alpha = (225 * alphaScale).toInt()
        val spdStr = if (lastRxMs == 0L) "--" else spdInt.toString()
        canvas.drawText(spdStr, cx, speedBaselineY, speedPaint)

        val halfSpdW = speedPaint.measureText(spdStr) * 0.5f
        val rightEdgeX = cx + halfSpdW + 10f

        if (gearText.isNotEmpty()) {
            val gx = rightEdgeX + 13f
            val gy = speedBaselineY - 22f
            rectF.set(gx - 12f, gy - 13.5f, gx + 12f, gy + 4.5f)
            gearBoxPaint.color = if (overLimit) 0xAAFFC247.toInt() else 0xAA00F0FF.toInt()
            gearBoxPaint.alpha = (180 * alphaScale).toInt()
            canvas.drawRoundRect(rectF, 4f, 4f, gearBoxPaint)

            gearPaint.color = if (overLimit) 0xFFFFC247.toInt() else 0xFF00F0FF.toInt()
            gearPaint.alpha = (225 * alphaScale).toInt()
            canvas.drawText(gearText, gx, gy, gearPaint)
        }

        unitPaint.alpha = (145 * alphaScale).toInt()
        canvas.drawText("km/h", rightEdgeX, speedBaselineY - 1f, unitPaint)

        // 3. Silent-Unless-Warning Status Line (y = 85.5% of height)
        val warnings = mutableListOf<String>()
        if (lastRxMs == 0L) {
            warnings.add("OBD WAIT")
        } else if (!isFresh) {
            warnings.add("OBD SIGNAL LOST")
        }
        coolantC?.let {
            if (it >= COOLANT_WARN_C) warnings.add("TEMP ${it.roundToInt()}°C")
        }
        volt?.let {
            if (it < VOLT_WARN_V) warnings.add(String.format("VOLT %.1fV", it))
        }
        if (warnings.isNotEmpty()) {
            warnPaint.color = if (lastRxMs == 0L) 0xAA7CE8FF.toInt() else 0xFFFFC247.toInt()
            warnPaint.alpha = if (lastRxMs == 0L) 150 else 225
            canvas.drawText(warnings.joinToString(" · "), cx, h * 0.855f, warnPaint)
        }
    }
}