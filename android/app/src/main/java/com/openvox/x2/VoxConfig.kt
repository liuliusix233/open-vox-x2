package com.openvox.x2

import java.net.InetSocketAddress
import java.net.Socket

data class EndpointSelection(
    val wsUrl: String,
    val isLan: Boolean
)

class VoxConfig(
    val clientId: String = "voxx2",
    val lanHost: String = BuildConfig.LAN_HOST,
    val lanPort: Int = BuildConfig.LAN_PORT,
    val wanWsUrl: String = BuildConfig.WAN_WS_URL,
    val token: String = BuildConfig.VOX_TOKEN
) {
    /**
     * 2026-11-22 出门体验：LAN 快探 600ms；连续失败 2 次立刻切公网 WSS
     * （旧的 5 次 x 2s 重连会把热点出门拖到十几秒才出网）。
     */
    fun pickBestEndpoint(forceLan: Boolean = false): EndpointSelection {
        // forceLan 分支：调用方已判定"内网还值得再试"，直接给内网地址（之后仍会回退）。
        if (forceLan && lanHost.isNotBlank()) {
            return EndpointSelection(
                wsUrl = "ws://$lanHost:$lanPort/ws/v1/duplex?client=$clientId",
                isLan = true
            )
        }
        if (lanHost.isNotBlank() && tcpReachable(lanHost, lanPort, 600)) {
            return EndpointSelection(
                wsUrl = "ws://$lanHost:$lanPort/ws/v1/duplex?client=$clientId",
                isLan = true
            )
        }
        val sep = if (wanWsUrl.contains("?")) "&" else "?"
        return EndpointSelection(
            wsUrl = "$wanWsUrl${sep}client=$clientId",
            isLan = false
        )
    }

    private fun tcpReachable(host: String, port: Int, timeoutMs: Int): Boolean {
        return try {
            Socket().use { s ->
                s.connect(InetSocketAddress(host, port), timeoutMs)
                true
            }
        } catch (_: Throwable) {
            false
        }
    }
}
