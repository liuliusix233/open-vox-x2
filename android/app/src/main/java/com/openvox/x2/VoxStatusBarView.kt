package com.openvox.x2

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.sin

/**
 * Minimal optical-waveguide top status strip for RayNeo X2 (pure black transparent background).
 * Displays pulsing network dot (LAN/WAN/OFF), mode tag, centered clock flanked by hairlines,
 * and vector battery gauge (+ charging bolt & thermal warning when >=38C).
 */
class VoxStatusBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val clockFmt = SimpleDateFormat("HH:mm", Locale.getDefault())

    private var connected = false
    private var lanMode = true
    private var modeLabel = "LIVE"
    private var batteryPct = -1
    private var charging = false
    private var tempC = 0f
    private var compactForNav = false
    private var pulsePhase = 0f

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.MONOSPACE
        textSize = 11.5f
    }
    private val clockPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        textSize = 13f
        color = 0xEECCF6FF.toInt()
        textAlign = Paint.Align.CENTER
    }
    private val battRect = RectF()

    private val pulseAnim = ValueAnimator.ofFloat(0f, 6.2832f).apply {
        duration = 2800L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            pulsePhase = it.animatedValue as Float
            if (visibility == VISIBLE) invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        pulseAnim.start()
    }

    override fun onDetachedFromWindow() {
        pulseAnim.cancel()
        super.onDetachedFromWindow()
    }

    fun setConnection(isConnected: Boolean, isLan: Boolean) {
        connected = isConnected
        lanMode = isLan
        invalidate()
    }

    fun setModeLabel(label: String) {
        modeLabel = label.uppercase()
        invalidate()
    }

    fun setTelemetry(pct: Int, isCharging: Boolean, tempCelsius: Float) {
        batteryPct = pct.coerceIn(0, 100)
        charging = isCharging
        tempC = tempCelsius
        invalidate()
    }

    /** Hide center clock when top AR compass bar is active so they never overlap. */
    fun setCompactForNav(compact: Boolean) {
        if (compactForNav != compact) {
            compactForNav = compact
            invalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val cy = height * 0.52f
        val leftX = 20f
        val rightX = width - 20f

        // 1. Left: Pulsing live status dot + LAN/WAN + Mode tag
        val breath = 0.65f + 0.35f * sin(pulsePhase)
        val (dotColor, netText) = when {
            !connected -> 0x99778899.toInt() to "OFF"
            lanMode -> 0xFF00F0FF.toInt() to "LAN"
            else -> 0xFF38F2A6.toInt() to "5G"
        }
        dotPaint.color = dotColor
        dotPaint.alpha = if (connected) (150 + 105 * breath).toInt() else 120
        canvas.drawCircle(leftX + 4f, cy, 3.6f, dotPaint)
        if (connected) {
            strokePaint.color = dotColor
            strokePaint.alpha = (65 * breath).toInt()
            strokePaint.strokeWidth = 1.2f
            canvas.drawCircle(leftX + 4f, cy, 6.8f, strokePaint)
        }

        textPaint.textAlign = Paint.Align.LEFT
        textPaint.color = if (connected) 0xCC99EEFF.toInt() else 0x88778899.toInt()
        val leftStr = if (modeLabel.isNotEmpty() && modeLabel != "LIVE") "$netText · $modeLabel" else netText
        canvas.drawText(leftStr, leftX + 15f, cy + 4f, textPaint)

        // 2. Center: Crisp clock flanked by hairline ticks (hidden when AR compass is active)
        if (!compactForNav) {
            val cx = width * 0.5f
            val timeStr = clockFmt.format(Date())
            canvas.drawText(timeStr, cx, cy + 4.5f, clockPaint)
            strokePaint.color = 0x5500E5FF
            strokePaint.strokeWidth = 1.2f
            canvas.drawLine(cx - 42f, cy, cx - 28f, cy, strokePaint)
            canvas.drawLine(cx + 28f, cy, cx + 42f, cy, strokePaint)
        }

        // 3. Right: Vector battery gauge + optional thermal readout (>= 38C)
        if (batteryPct >= 0) {
            val battW = 20f
            val battH = 10f
            val bx = rightX - battW - 2f
            val by = cy - battH * 0.5f

            val battColor = when {
                charging -> 0xFF38F2A6.toInt()
                batteryPct <= 15 -> 0xFFFF5A5F.toInt()
                batteryPct <= 28 -> 0xFFFFC247.toInt()
                else -> 0xCC99EEFF.toInt()
            }

            // Battery outline + terminal nub
            strokePaint.color = battColor
            strokePaint.alpha = 210
            strokePaint.strokeWidth = 1.3f
            battRect.set(bx, by, bx + battW, by + battH)
            canvas.drawRoundRect(battRect, 2.2f, 2.2f, strokePaint)
            dotPaint.color = battColor
            dotPaint.alpha = 210
            canvas.drawRect(bx + battW + 0.5f, cy - 2.5f, bx + battW + 2.5f, cy + 2.5f, dotPaint)

            // Dynamic fill bar
            val fillW = ((battW - 4f) * (batteryPct / 100f)).coerceAtLeast(1f)
            canvas.drawRoundRect(bx + 2f, by + 2f, bx + 2f + fillW, by + battH - 2f, 1f, 1f, dotPaint)

            // Percentage (+ bolt if charging) + thermal badge if warm
            textPaint.textAlign = Paint.Align.RIGHT
            textPaint.color = battColor
            val chargeMark = if (charging) "⚡" else ""
            val tempMark = if (tempC >= 38f) "${tempC.toInt()}° · " else ""
            canvas.drawText("$tempMark$chargeMark$batteryPct%", bx - 6f, cy + 4f, textPaint)
        }
    }
}
