package com.openvox.x2

/**
 * 2026-11-22 三模式状态机（LIVE / STANDBY / SLEEP_LISTEN）。
 *
 * 持有点放在进程级单例里，而不是搬到 Service：
 * 本工程 Activity 常驻（launchMode=singleTask）+ 前台服务保活，对象本来就不会因熄屏销毁，
 * 迁移 Binder 只会增加断链风险。Service 只负责 WAKE_LOCK 与亮/灭屏广播。
 */
enum class VoxPowerMode { LIVE, STANDBY, SLEEP_LISTEN }

object VoxModeController {
    @Volatile
    var current: VoxPowerMode = VoxPowerMode.LIVE
        private set

    /** 只记录 LIVE / STANDBY，用于亮屏后恢复 */
    @Volatile
    var lastActive: VoxPowerMode = VoxPowerMode.LIVE
        private set

    /** (new, old) */
    @Volatile
    var listener: ((VoxPowerMode, VoxPowerMode) -> Unit)? = null

    fun setMode(newMode: VoxPowerMode) {
        val old = current
        if (old == newMode) return
        current = newMode
        if (newMode != VoxPowerMode.SLEEP_LISTEN) lastActive = newMode
        listener?.invoke(newMode, old)
    }

    fun toggleStandby() {
        setMode(if (current == VoxPowerMode.LIVE) VoxPowerMode.STANDBY else VoxPowerMode.LIVE)
    }

    /** 亮屏 / 拿起：回到进入休眠前的模式 */
    fun restore() = setMode(lastActive)
}
