package com.openvox.x2

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log

/**
 * 2x2 Adaptive Power & True-Sleep Policy Controller for RayNeo X2.
 *
 * Orthogonal dimensions:
 *   1. UserMode (manual double-tap): LIVE <-> STANDBY, plus system SLEEP on screen-off.
 *   2. PowerProfile (auto-detected): AC (plugged into power bank) vs BAT (unplugged).
 *
 * Aligns with RayNeo X2 firmware behavior (which turns off Wi-Fi/BT on screen-off):
 *   - LIVE    + AC  -> Full continuous stream (1080p LAN / 720p WAN 1-3fps), full HUD, full agency
 *   - LIVE    + BAT -> Event/periodic snapshot only (vad + every 30s 960x720 q75), full HUD, 1.8x agency interval
 *   - STANDBY + AC  -> Black transparent HUD (APL=0), silent low-rate 720p archive (every 15s), quiet agency
 *   - STANDBY + BAT -> True deep sleep (stop camera, stop audio + audio_source_record=off, close WS, release wake lock)
 *   - SLEEP (screen off) -> Always true deep sleep; auto-resumes cleanly on ACTION_SCREEN_ON + Wi-Fi available
 */
class VoxPowerPolicy(
    private val context: Context,
    private val onPolicyChanged: (EffectivePolicy) -> Unit
) {
    companion object {
        private const val TAG = "VoxPower"
    }

    enum class UserMode { LIVE, STANDBY, SLEEP }
    enum class PowerProfile { AC, BAT }
    enum class CameraMode { CONTINUOUS_STREAM, EVENT_AND_30S, SILENT_ARCHIVE_15S, OFF }

    data class EffectivePolicy(
        val userMode: UserMode,
        val power: PowerProfile,
        val cameraMode: CameraMode,
        val showHud: Boolean,
        val keepAudioAndWs: Boolean,
        val holdWakeLock: Boolean,
        val statusBadge: String,
        val heartbeatMult: Float
    )

    @Volatile private var userMode = UserMode.LIVE
    @Volatile private var lastAwakeMode = UserMode.LIVE
    @Volatile private var powerProfile = PowerProfile.BAT
    @Volatile private var registered = false

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_POWER_CONNECTED -> updatePower(PowerProfile.AC)
                Intent.ACTION_POWER_DISCONNECTED -> updatePower(PowerProfile.BAT)
                Intent.ACTION_SCREEN_OFF -> enterSystemSleep("screen_off")
                Intent.ACTION_SCREEN_ON -> wakeFromSystemSleep("screen_on")
            }
        }
    }

    fun start() {
        if (registered) return
        registered = true
        powerProfile = readInitialPowerProfile()
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_POWER_CONNECTED)
            addAction(Intent.ACTION_POWER_DISCONNECTED)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        context.registerReceiver(receiver, filter)
        emitCurrent("init")
    }

    fun stop() {
        if (!registered) return
        registered = false
        try { context.unregisterReceiver(receiver) } catch (_: Throwable) {}
    }

    /** Called on temple double-tap: toggles LIVE <-> STANDBY. */
    fun toggleLiveStandby(): EffectivePolicy {
        val next = if (userMode == UserMode.LIVE) UserMode.STANDBY else UserMode.LIVE
        userMode = next
        lastAwakeMode = next
        return emitCurrent("double_tap")
    }

    fun setUserMode(mode: UserMode): EffectivePolicy {
        if (mode != UserMode.SLEEP) lastAwakeMode = mode
        userMode = mode
        return emitCurrent("explicit_set")
    }

    fun enterSystemSleep(reason: String) {
        if (userMode == UserMode.SLEEP) return
        userMode = UserMode.SLEEP
        emitCurrent(reason)
    }

    fun wakeFromSystemSleep(reason: String) {
        if (userMode != UserMode.SLEEP) return
        userMode = lastAwakeMode
        emitCurrent(reason)
    }

    fun currentPolicy(): EffectivePolicy = computePolicy(userMode, powerProfile)

    private fun updatePower(newPower: PowerProfile) {
        if (powerProfile == newPower) return
        powerProfile = newPower
        emitCurrent("power_changed")
    }

    private fun readInitialPowerProfile(): PowerProfile {
        val sticky = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val plugged = sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        return if (plugged != 0) PowerProfile.AC else PowerProfile.BAT
    }

    private fun computePolicy(mode: UserMode, power: PowerProfile): EffectivePolicy {
        return when {
            mode == UserMode.SLEEP -> EffectivePolicy(
                userMode = UserMode.SLEEP,
                power = power,
                cameraMode = CameraMode.OFF,
                showHud = false,
                keepAudioAndWs = false,
                holdWakeLock = false,
                statusBadge = "SLEEP",
                heartbeatMult = 99f
            )
            mode == UserMode.STANDBY && power == PowerProfile.BAT -> EffectivePolicy(
                userMode = UserMode.STANDBY,
                power = PowerProfile.BAT,
                cameraMode = CameraMode.OFF,
                showHud = false,
                keepAudioAndWs = false, // Unplugged standby -> true deep sleep to save battery
                holdWakeLock = false,
                statusBadge = "SLEEP",
                heartbeatMult = 99f
            )
            mode == UserMode.STANDBY && power == PowerProfile.AC -> EffectivePolicy(
                userMode = UserMode.STANDBY,
                power = PowerProfile.AC,
                cameraMode = CameraMode.SILENT_ARCHIVE_15S,
                showHud = false,
                keepAudioAndWs = true,
                holdWakeLock = true,
                statusBadge = "REC",
                heartbeatMult = 99f
            )
            mode == UserMode.LIVE && power == PowerProfile.BAT -> EffectivePolicy(
                userMode = UserMode.LIVE,
                power = PowerProfile.BAT,
                cameraMode = CameraMode.EVENT_AND_30S,
                showHud = true,
                keepAudioAndWs = true,
                holdWakeLock = false,
                statusBadge = "ECO",
                heartbeatMult = 1.8f
            )
            else -> EffectivePolicy(
                userMode = UserMode.LIVE,
                power = PowerProfile.AC,
                cameraMode = CameraMode.CONTINUOUS_STREAM,
                showHud = true,
                keepAudioAndWs = true,
                holdWakeLock = true,
                statusBadge = "LIVE",
                heartbeatMult = 1.0f
            )
        }
    }

    private fun emitCurrent(reason: String): EffectivePolicy {
        val p = computePolicy(userMode, powerProfile)
        Log.i(
            TAG,
            "POLICY($reason) mode=${p.userMode} power=${p.power} cam=${p.cameraMode} " +
                "hud=${p.showHud} audioWs=${p.keepAudioAndWs} badge=${p.statusBadge}"
        )
        onPolicyChanged(p)
        return p
    }
}
