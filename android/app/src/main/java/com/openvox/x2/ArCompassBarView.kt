package com.openvox.x2

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Game-style 60fps binocular AR top compass & waypoint bar for RayNeo X2 waveguide.
 * Inspired by tactical horizontal HUDs: pure black (#000000) transparent background,
 * fine degree ruler, circular target reticles with vertical drop-lines, km/m distance,
 * and left/right FOV edge clamping arrows that slide smoothly as the user turns their head.
 */
class ArCompassBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class Waypoint(
        val label: String,
        val kind: String,      // "dest" | "turn"
        val bearingDeg: Float, // 0..360 true north
        val distM: Int
    )

    companion object {
        private const val HALF_FOV_DEG = 42f
        private val CARDINALS = mapOf(
            0 to "N", 45 to "NE", 90 to "E", 135 to "SE",
            180 to "S", 225 to "SW", 270 to "W", 315 to "NW"
        )
    }

    private var active = false
    private var headingDeg = 0f
    private var nextInstruction = ""
    private var waypoints: List<Waypoint> = emptyList()

    private val rulerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x8899CCDD.toInt()
        strokeWidth = 1.5f
        style = Paint.Style.STROKE
    }
    private val dashPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x99FFFFFF.toInt()
        strokeWidth = 1.5f
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(4f, 4f), 0f)
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.2f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val cardinalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xAA88BBCC.toInt()
        textSize = 11f
        typeface = Typeface.MONOSPACE
        textAlign = Paint.Align.CENTER
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 12f
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
    }
    private val distPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00F0FF.toInt()
        textSize = 11f
        typeface = Typeface.MONOSPACE
        textAlign = Paint.Align.CENTER
    }
    private val turnBannerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xEEFFEEAA.toInt()
        textSize = 12.5f
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
    }

    fun updateNav(isActive: Boolean, instruction: String, newWaypoints: List<Waypoint>) {
        active = isActive
        nextInstruction = instruction
        waypoints = newWaypoints
        visibility = if (active) VISIBLE else GONE
        invalidate()
    }

    /** Smoothly update head azimuth (0..360 deg) at 60Hz with wrap-around protection. */
    fun setHeadingDeg(rawDeg: Float) {
        val target = ((rawDeg % 360f) + 360f) % 360f
        val diff = normalizeAngle(target - headingDeg)
        headingDeg = ((headingDeg + diff * 0.28f) % 360f + 360f) % 360f
        if (active) invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!active || width <= 0 || height <= 0) return

        val cx = width * 0.5f
        val barLeft = width * 0.08f
        val barRight = width * 0.92f
        val barSpan = barRight - barLeft
        val pxPerDeg = barSpan / (HALF_FOV_DEG * 2f)
        val rulerY = 42f

        // 1. Horizontal tactical baseline
        rulerPaint.color = 0x6688CCDD.toInt()
        rulerPaint.strokeWidth = 1.5f
        canvas.drawLine(barLeft, rulerY, barRight, rulerY, rulerPaint)

        // 2. Degree ticks & cardinal markers scrolling with head yaw
        val startDeg = (headingDeg - HALF_FOV_DEG).roundToInt()
        val endDeg = (headingDeg + HALF_FOV_DEG).roundToInt()
        for (d in startDeg..endDeg) {
            if (d % 5 != 0) continue
            val norm = ((d % 360) + 360) % 360
            val rel = normalizeAngle(d - headingDeg)
            val x = cx + rel * pxPerDeg
            if (x < barLeft || x > barRight) continue

            val isMajor = norm % 15 == 0
            val tickH = if (isMajor) 10f else 5f
            rulerPaint.color = if (isMajor) 0xAA99EEFF.toInt() else 0x557799AA.toInt()
            canvas.drawLine(x, rulerY - tickH, x, rulerY + 2f, rulerPaint)

            val card = CARDINALS[norm]
            if (card != null) {
                canvas.drawText(card, x, rulerY - 13f, cardinalPaint)
            }
        }

        // 3. Center gaze notch
        fillPaint.color = 0xDD00F0FF.toInt()
        val notch = Path().apply {
            moveTo(cx, rulerY - 2f)
            lineTo(cx - 5f, rulerY - 9f)
            lineTo(cx + 5f, rulerY - 9f)
            close()
        }
        canvas.drawPath(notch, fillPaint)

        // 4. Tactical circular waypoint nodes
        for (wp in waypoints) {
            val rel = normalizeAngle(wp.bearingDeg - headingDeg)
            val isDest = wp.kind == "dest"
            val accent = if (isDest) 0xFF00F0FF.toInt() else 0xFFFFB000.toInt()
            val clamped = rel.coerceIn(-HALF_FOV_DEG, HALF_FOV_DEG)
            val x = cx + clamped * pxPerDeg
            val offLeft = rel < -HALF_FOV_DEG
            val offRight = rel > HALF_FOV_DEG

            // Concentric circular reticle on ruler
            ringPaint.color = accent
            ringPaint.strokeWidth = if (isDest) 2.4f else 2.0f
            val r = if (isDest) 11f else 8.5f
            canvas.drawCircle(x, rulerY, r, ringPaint)
            if (isDest && abs(rel) < 8f) {
                // Subtle outer halo when looking straight toward destination
                ringPaint.color = 0x5500F0FF
                ringPaint.strokeWidth = 1.5f
                canvas.drawCircle(x, rulerY, r + 6f, ringPaint)
            }
            fillPaint.color = accent
            canvas.drawCircle(x, rulerY, 3.2f, fillPaint)

            // Vertical dashed drop-line & label/distance
            canvas.drawLine(x, rulerY - r - 2f, x, rulerY - r - 12f, dashPaint)
            val prefix = when {
                offLeft -> "◀ "
                offRight -> ""
                else -> ""
            }
            val suffix = if (offRight) " ▶" else ""
            labelPaint.color = accent
            canvas.drawText("$prefix${wp.label}$suffix", x, 16f, labelPaint)

            distPaint.color = if (isDest) 0xFFCCF8FF.toInt() else 0xFFFFE099.toInt()
            canvas.drawText(formatDist(wp.distM), x, rulerY + r + 13f, distPaint)
        }

        // 5. Compact next-turn banner below the bar
        if (nextInstruction.isNotEmpty()) {
            canvas.drawText(nextInstruction, cx, rulerY + 40f, turnBannerPaint)
        }
    }

    private fun normalizeAngle(deg: Float): Float {
        var a = deg % 360f
        if (a > 180f) a -= 360f
        if (a < -180f) a += 360f
        return a
    }

    private fun formatDist(distM: Int): String {
        if (distM < 0) return "--"
        return if (distM >= 1000) String.format("%.1f km", distM / 1000f) else "${distM} m"
    }
}
