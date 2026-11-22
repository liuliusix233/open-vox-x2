package com.openvox.x2

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.PI

/**
 * 60Hz head-yaw azimuth tracker (0..360 deg true/magnetic north) using Android's fused
 * TYPE_ROTATION_VECTOR sensor. Feeds ArCompassBarView with zero network latency.
 */
class HeadHeadingTracker(
    context: Context,
    private val onHeadingDeg: (Float) -> Unit
) : SensorEventListener {

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotSensor: Sensor? =
        sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sm.getDefaultSensor(Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR)

    private val rotMatrix = FloatArray(9)
    private val remapped = FloatArray(9)
    private val orientation = FloatArray(3)
    private var running = false

    fun start() {
        if (running || rotSensor == null) return
        running = true
        sm.registerListener(this, rotSensor, SensorManager.SENSOR_DELAY_GAME)
    }

    fun stop() {
        if (!running) return
        running = false
        sm.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR &&
            event.sensor.type != Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR
        ) return
        SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
        // RayNeo X2 glasses are landscape-mounted; remap X->X, Z->Y so forward gaze gives yaw
        SensorManager.remapCoordinateSystem(
            rotMatrix,
            SensorManager.AXIS_X,
            SensorManager.AXIS_Z,
            remapped
        )
        SensorManager.getOrientation(remapped, orientation)
        // 2026-11-22：kotlin.math.toDegrees 是 1.6+ 的实验性 API，本工程 Kotlin 1.8 编译器下
        // 解析不到；直接乘 180/PI 等价替换。
        val azimuth = ((orientation[0] * 180f / PI.toFloat()) % 360f + 360f) % 360f
        onHeadingDeg(azimuth)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
