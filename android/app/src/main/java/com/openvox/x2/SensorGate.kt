package com.openvox.x2

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import android.os.SystemClock
import kotlin.math.sqrt

data class GeoSnapshot(val lat: Double, val lon: Double, val acc: Float)

/**
 * Single process-wide gyroscope (SENSOR_DELAY_UI) + passive location cache.
 * Eliminates dual-gyro battery drain, stop() deadlock, and blocking Thread.join() GPS fetches.
 */
class SensorGate(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val gyro: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    @Volatile var started = false; private set
    @Volatile var cachedGeo: GeoSnapshot? = null; private set
    @Volatile var isWalking = false; private set

    var onGazeLocked: (() -> Unit)? = null
    /** 连续静止 45s（确认已放桌面而非歪头）→ 进 SLEEP_LISTEN */
    var onLongStill: (() -> Unit)? = null

    private var stillStartMs = 0L
    private var stillAlerted = false
    private var lastGazeFireMs = 0L
    private var walkMotionCount = 0

    private val locListener = LocationListener { loc: Location ->
        cachedGeo = GeoSnapshot(loc.latitude, loc.longitude, loc.accuracy)
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (started) return
        started = true
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
        try {
            locManager.getLastKnownLocation(LocationManager.PASSIVE_PROVIDER)?.let {
                cachedGeo = GeoSnapshot(it.latitude, it.longitude, it.accuracy)
            }
            locManager.requestLocationUpdates(
                LocationManager.PASSIVE_PROVIDER, 15_000L, 10f, locListener, Looper.getMainLooper()
            )
        } catch (_: Throwable) {}
    }

    fun updateGeo(lat: Double, lon: Double, acc: Float) {
        cachedGeo = GeoSnapshot(lat, lon, acc)
    }

    fun stop() {
        if (!started) return
        started = false
        try { sensorManager.unregisterListener(this) } catch (_: Throwable) {}
        try { locManager.removeUpdates(locListener) } catch (_: Throwable) {}
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        val degPerSec = Math.toDegrees(
            sqrt(
                (event.values[0] * event.values[0] +
                 event.values[1] * event.values[1] +
                 event.values[2] * event.values[2]).toDouble()
            )
        )
        val now = SystemClock.uptimeMillis()
        if (degPerSec > 18.0) {
            walkMotionCount = (walkMotionCount + 1).coerceAtMost(30)
            stillStartMs = 0L
            stillAlerted = false
        } else {
            walkMotionCount = (walkMotionCount - 1).coerceAtLeast(0)
            if (degPerSec < 8.0) {
                if (stillStartMs == 0L) stillStartMs = now
                if (!stillAlerted && now - stillStartMs >= 45_000L) {
                    stillAlerted = true
                    onLongStill?.invoke()
                }
                if (now - stillStartMs >= 1200L && now - lastGazeFireMs >= 30_000L) {
                    lastGazeFireMs = now
                    onGazeLocked?.invoke()
                }
            } else {
                stillStartMs = 0L
            }
        }
        isWalking = walkMotionCount >= 12
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
