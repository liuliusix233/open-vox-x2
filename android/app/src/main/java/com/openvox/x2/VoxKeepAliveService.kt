package com.openvox.x2

import android.app.*
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.PowerManager
import android.os.Build
import android.os.IBinder

/**
 * 2026-11-22: NotificationCompat 换成平台 Notification.Builder —— 本机只有 platform-32，
 * androidx 那套要 compileSdk 33+。渠道 ID 与前台服务类型不变。
 */
class VoxKeepAliveService : Service() {
    companion object {
        private const val CH_ID = "vox_x2_live"
    }

    private var wake: PowerManager.WakeLock? = null

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_OFF -> VoxModeController.setMode(VoxPowerMode.SLEEP_LISTEN)
                Intent.ACTION_SCREEN_ON -> VoxModeController.restore()
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "vox:duplex").apply {
                setReferenceCounted(false)
                acquire()
            }
        } catch (_: Throwable) {}
        val f = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        try { registerReceiver(screenReceiver, f) } catch (_: Throwable) {}
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CH_ID, "Vox X2 Live", NotificationManager.IMPORTANCE_LOW)
        )
        val n = Notification.Builder(this, CH_ID)
            .setContentTitle("Vox X2")
            .setContentText("Full-duplex voice & vision active")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .build()
        startForeground(1, n)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        try { unregisterReceiver(screenReceiver) } catch (_: Throwable) {}
        try { wake?.release() } catch (_: Throwable) {}
        wake = null
        try {
            val am = getSystemService(Context.AUDIO_SERVICE) as android.media.AudioManager
            am.setParameters("audio_source_record=off")
        } catch (_: Throwable) {}
        super.onDestroy()
    }
}
