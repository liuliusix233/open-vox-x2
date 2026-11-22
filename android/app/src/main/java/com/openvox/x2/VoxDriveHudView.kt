package com.openvox.x2

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * F-35 HMDS-inspired Hybrid-Stabilized "Light-Wing" Driving HUD for RayNeo X2 waveguide.
 *
 * Optical & Avionics Design (640x480 per eye, 25 deg FOV):
 *   1. Parafoveal Light-Wing Layout (zero center-road obstruction):
 *        - Upper 62% of FOV stays 100% pure black (transparent).
 *        - Symmetrical outward-expanding RPM light wings at y = 66.5% (cyan -> amber -> redline).
 *        - Centered night-friendly monospace speed + tucked gear badge at y = 79%.
 *        - Silent-unless-warning vitals (coolant/voltage stay hidden while normal).
 *   2. Hybrid 3DoF Micro-Stabilization (pure gyroscope, zero magnetometer / car-turn drift):
 *        - Absorbs road bumps and high-frequency head jitter via +/-36px (X) / +/-24px (Y)
 *          spring-damped counter-translation so telemetry floats steadily ahead of the hood.
 *   3. Off-Boresight Declutter (fighter-jet mirror/blind-spot clarity):
 *        - Quick head turns (> 18 deg off-boresight, e.g., checking mirrors or passenger)
 *          smoothly fade HUD alpha to 0 in ~150ms; turning back to the road restores it
 *          immediately. Slow highway curves (< 20 deg/s) never trigger false declutter.
 */
class VoxDriveHudView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs), SensorEventListener {

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

        // Hybrid 3DoF gyro stabilization & off-boresight declutter tuning
        private const val RAD_TO_DEG = 57.2957795f
        private const val PX_PER_DEG_X = 16.0f
        private const val PX_PER_DEG_Y = 14.0f
        private const val MAX_SHIFT_X_PX = 36f
        private const val MAX_SHIFT_Y_PX = 24f
        private const val SPRING_TAU_S = 0.65f          // Fast gentle recentering for bumps
        private const val DECLUTTER_YAW_DEG = 18.0f     // Fade out when glancing > 18 deg left/right
        private const val DECLUTTER_PITCH_DEG = 15.0f   // Fade out when looking down at dash/phone
        private const val HEAD_TURN_RATE_MIN = 0.32f    // ~18 deg/s; ignores slow highway curves
        private const val DECLUTTER_RECENTER_TAU_S = 2.6f
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

    // Gyroscope hybrid-3DoF state
    private val sensorManager: SensorManager? =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private var gyroRegistered = false
    private var lastGyroNs = 0L
    private var stabShiftX = 0f
    private var stabShiftY = 0f
    private var offAxisYawDeg = 0f
    private var offAxisPitchDeg = 0f
    private var targetDeclutterAlpha = 1f
    private var displayDeclutterAlpha = 1f

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
            val prevAlpha = displayDeclutterAlpha
            displaySpeed += (targetSpeed - displaySpeed) * 0.28f
            displayRpm += (targetRpm - displayRpm) * 0.24f
            // Fast 150ms fade-out / smooth fade-in for off-boresight declutter
            val alphaRate = if (targetDeclutterAlpha < displayDeclutterAlpha) 0.35f else 0.22f
            displayDeclutterAlpha += (targetDeclutterAlpha - displayDeclutterAlpha) * alphaRate

            val moving = abs(stabShiftX) > 0.2f || abs(stabShiftY) > 0.2f
            val redlining = displayRpm >= MAX_RPM * 0.80f
            val staleNow = lastRxMs > 0L && (SystemClock.uptimeMillis() - lastRxMs) > STALE_MS
            if (abs(displaySpeed - prevSpd) > 0.05f ||
                abs(displayRpm - prevRpm) > 5f ||
                abs(displayDeclutterAlpha - prevAlpha) > 0.01f ||
                moving ||
                redlining ||
                staleNow
            ) {
                invalidate()
            }
        }
    }

    private fun updateGyroSubscription() {
        val wantActive = isAttachedToWindow && visibility == VISIBLE
        if (wantActive && !gyroRegistered) {
            val gyro = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
            if (gyro != null) {
                lastGyroNs = 0L
                gyroRegistered = sensorManager?.registerListener(
                    this, gyro, SensorManager.SENSOR_DELAY_GAME
                ) ?: false
            }
        } else if (!wantActive && gyroRegistered) {
            sensorManager?.unregisterListener(this)
            gyroRegistered = false
            lastGyroNs = 0L
            stabShiftX = 0f
            stabShiftY = 0f
            offAxisYawDeg = 0f
            offAxisPitchDeg = 0f
            targetDeclutterAlpha = 1f
            displayDeclutterAlpha = 1f
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        smoothAnim.start()
        updateGyroSubscription()
    }

    override fun onDetachedFromWindow() {
        smoothAnim.cancel()
        if (gyroRegistered) {
            sensorManager?.unregisterListener(this)
            gyroRegistered = false
        }
        super.onDetachedFromWindow()
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        updateGyroSubscription()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        val ts = event.timestamp
        if (lastGyroNs == 0L) {
            lastGyroNs = ts
            return
        }
        val dt = ((ts - lastGyroNs) * 1e-9f).coerceIn(0.001f, 0.08f)
        lastGyroNs = ts

        // RayNeo X2 landscape glasses axes: values[0] = pitch (rad/s), values[1] = yaw (rad/s)
        val pitchRadS = if (abs(event.values[0]) < 0.015f) 0f else event.values[0]
        val yawRadS = if (abs(event.values[1]) < 0.015f) 0f else event.values[1]

        // 1. Micro-stabilization counter-shift (absorbs road bumps & head jitter)
        val springDecay = exp(-dt / SPRING_TAU_S)
        stabShiftX = (stabShiftX * springDecay + yawRadS * RAD_TO_DEG * dt * PX_PER_DEG_X)
            .coerceIn(-MAX_SHIFT_X_PX, MAX_SHIFT_X_PX)
        stabShiftY = (stabShiftY * springDecay + pitchRadS * RAD_TO_DEG * dt * PX_PER_DEG_Y)
            .coerceIn(-MAX_SHIFT_Y_PX, MAX_SHIFT_Y_PX)

        // 2. Off-boresight declutter (only rapid head turns accumulate; slow car turns ignored)
        val recenterDecay = exp(-dt / DECLUTTER_RECENTER_TAU_S)
        if (abs(yawRadS) >= HEAD_TURN_RATE_MIN) {
            offAxisYawDeg = (offAxisYawDeg + yawRadS * RAD_TO_DEG * dt).coerceIn(-65f, 65f)
        } else {
            offAxisYawDeg *= recenterDecay
        }
        if (abs(pitchRadS) >= HEAD_TURN_RATE_MIN) {
            offAxisPitchDeg = (offAxisPitchDeg + pitchRadS * RAD_TO_DEG * dt).coerceIn(-45f, 45f)
        } else {
            offAxisPitchDeg *= recenterDecay
        }

        val offAxis = abs(offAxisYawDeg) > DECLUTTER_YAW_DEG || abs(offAxisPitchDeg) > DECLUTTER_PITCH_DEG
        targetDeclutterAlpha = if (offAxis) 0f else 1f
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
        stabShiftX = 0f
        stabShiftY = 0f
        offAxisYawDeg = 0f
        offAxisPitchDeg = 0f
        targetDeclutterAlpha = 1f
        displayDeclutterAlpha = 1f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        if (displayDeclutterAlpha <= 0.02f) return

        val now = SystemClock.uptimeMillis()
        val isFresh = lastRxMs > 0L && (now - lastRxMs) <= STALE_MS
        val alphaScale = (if (isFresh) 1f else 0.40f) * displayDeclutterAlpha

        val w = width.toFloat()
        val h = height.toFloat()

        canvas.save()
        canvas.translate(stabShiftX, stabShiftY)

        val cx = w * 0.5f

        // 1. Parafoveal Symmetrical RPM Light Wings (y = 66.5%, expands outward from center)
        val wingY = h * 0.665f
        val halfSpan = (w * 0.24f).coerceIn(96f, 165f)
        val centerGap = 6f
        val segGap = 2.6f
        val segW = (halfSpan - centerGap - segGap * (WING_SEGMENTS_PER_SIDE - 1)) / WING_SEGMENTS_PER_SIDE
        val rpmRatio = (displayRpm / MAX_RPM).coerceIn(0f, 1f)
        val litPerSide = (rpmRatio * WING_SEGMENTS_PER_SIDE).roundToInt().coerceIn(0, WING_SEGMENTS_PER_SIDE)

        val redlinePulse = if (rpmRatio >= 0.80f && isFresh) {
            (0.75f + 0.25f * sin(now * 0.018).toFloat())
        } else {
            1f
        }

        centerTickPaint.alpha = (110 * alphaScale).toInt()
        canvas.drawCircle(cx, wingY, 1.8f, centerTickPaint)

        for (i in 0 until WING_SEGMENTS_PER_SIDE) {
            val frac = (i + 1).toFloat() / WING_SEGMENTS_PER_SIDE
            val isLit = i < litPerSide && lastRxMs > 0L
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
            rectF.set(cx + offset, wingY - halfH, cx + offset + segW, wingY + halfH)
            canvas.drawRoundRect(rectF, 1.5f, 1.5f, segPaint)
            rectF.set(cx - offset - segW, wingY - halfH, cx - offset, wingY + halfH)
            canvas.drawRoundRect(rectF, 1.5f, 1.5f, segPaint)
        }

        // 2. Centered Digital Speed + Tucked Gear Badge (y = 79%)
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

        // 3. Silent-Unless-Warning Status Line (y = 85.5%)
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
            warnPaint.alpha = ((if (lastRxMs == 0L) 150 else 225) * displayDeclutterAlpha).toInt()
            canvas.drawText(warnings.joinToString(" · "), cx, h * 0.855f, warnPaint)
        }

        canvas.restore()
    }
}