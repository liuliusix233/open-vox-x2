package com.openvox.x2

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress

/**
 * Detects connection to the in-car Wi-Fi SSID ("") and listens
 * for Raspberry Pi LAN UDP telemetry broadcasts on port 18766.
 */
class DriveModeDetector(
    private val context: Context,
    private val targetSsid: String = "",
    private val udpPort: Int = 18766,
    private val onDriveModeChanged: (Boolean) -> Unit,
    private val onTelemetry: (VoxDriveHudView.Telemetry) -> Unit
) {
    companion object {
        private const val TAG = "VoxDrive"
    }

    private val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private val wm = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

    @Volatile private var running = false
    @Volatile private var driveActive = false
    private var unknownStreak = 0
    private var udpSocket: DatagramSocket? = null
    private var udpThread: Thread? = null

    private val netCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            checkSsid()
        }

        override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
            val ssid = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                (caps.transportInfo as? WifiInfo)?.ssid
            } else null
            checkSsid(ssid)
        }

        override fun onLost(network: Network) {
            checkSsid("")
        }
    }

    fun start() {
        if (running) return
        running = true
        val req = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .build()
        try {
            cm.registerNetworkCallback(req, netCallback)
        } catch (t: Throwable) {
            Log.w(TAG, "registerNetworkCallback failed: ${t.message}")
        }
        checkSsid()
    }

    fun stop() {
        running = false
        try { cm.unregisterNetworkCallback(netCallback) } catch (_: Throwable) {}
        stopUdpListener()
        setDriveActive(false)
    }

    fun isDriveActive(): Boolean = driveActive

    private fun checkSsid(hintSsid: String? = null) {
        val raw = when {
            !hintSsid.isNullOrBlank() -> hintSsid
            else -> try {
                @Suppress("DEPRECATION")
                wm.connectionInfo?.ssid ?: ""
            } catch (_: Throwable) { "" }
        }
        val clean = raw.trim().removePrefix("\"").removeSuffix("\"")
        // Android 在扫描/重连瞬间会返回 <unknown ssid> 或空串；这时不能判定为"离开车载热点"，
        // 否则驾驶 HUD 会反复闪。只有稳定读到"另一个已知 SSID"时才退出。
        if (clean.isEmpty() || clean.equals("<unknown ssid>", ignoreCase = true)) {
            Log.d(TAG, "Wi-Fi SSID unknown -> keep drive=$driveActive")
            return
        }
        val matched = clean.equals(targetSsid, ignoreCase = true)
        Log.d(TAG, "Wi-Fi SSID check: '$clean' -> drive=$matched")
        if (!matched) {
            unknownStreak++
            if (unknownStreak < 2) {   // 需要连续两次读到别的 SSID 才退出
                return
            }
        }
        unknownStreak = 0
        setDriveActive(matched)
    }

    private fun setDriveActive(active: Boolean) {
        if (driveActive == active) return
        driveActive = active
        Log.i(TAG, "DRIVE_MODE -> $active (ssid=$targetSsid)")
        if (active) startUdpListener() else stopUdpListener()
        onDriveModeChanged(active)
    }

    private fun startUdpListener() {
        if (udpThread?.isAlive == true) return
        udpThread = Thread({
            val buf = ByteArray(1024)
            try {
                val sock = DatagramSocket(null).apply {
                    reuseAddress = true
                    broadcast = true
                    bind(InetSocketAddress(udpPort))
                }
                udpSocket = sock
                Log.i(TAG, "UDP telemetry listening on :$udpPort")
                while (running && driveActive) {
                    val pkt = DatagramPacket(buf, buf.size)
                    sock.receive(pkt)
                    val jsonStr = String(pkt.data, pkt.offset, pkt.length, Charsets.UTF_8).trim()
                    parseTelemetry(jsonStr)?.let { onTelemetry(it) }
                }
            } catch (t: Throwable) {
                if (running && driveActive) Log.w(TAG, "UDP listener stopped: ${t.message}")
            } finally {
                try { udpSocket?.close() } catch (_: Throwable) {}
                udpSocket = null
            }
        }, "vox-drive-udp").apply { isDaemon = true; start() }
    }

    private fun stopUdpListener() {
        try { udpSocket?.close() } catch (_: Throwable) {}
        udpSocket = null
        udpThread?.interrupt()
        udpThread = null
    }

    private fun parseTelemetry(jsonStr: String): VoxDriveHudView.Telemetry? {
        return try {
            val o = JSONObject(jsonStr)
            val spd = when {
                o.has("speed_kmh") -> o.optDouble("speed_kmh", 0.0).toFloat()
                o.has("speed") -> o.optDouble("speed", 0.0).toFloat()
                else -> 0f
            }
            val rpm = o.optDouble("rpm", 0.0).toFloat()
            val gear = o.optString("gear", "")
            val cool = if (o.has("coolant_c")) o.optDouble("coolant_c").toFloat() else null
            val volt = if (o.has("volt")) o.optDouble("volt").toFloat() else null
            val limit = if (o.has("speed_limit")) o.optDouble("speed_limit").toFloat() else null
            VoxDriveHudView.Telemetry(
                speedKmh = spd,
                rpm = rpm,
                gear = gear,
                coolantC = cool,
                volt = volt,
                speedLimitKmh = limit
            )
        } catch (_: Throwable) {
            null
        }
    }
}
