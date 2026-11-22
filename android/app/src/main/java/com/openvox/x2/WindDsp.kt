package com.openvox.x2

import kotlin.math.abs

/**
 * Continuous-state 2nd-order high-pass wind filter with smooth wet/dry crossfade.
 * Never freezes filter state on quiet frames (prevents click/pop on wind onset).
 */
class WindDsp {
    private class BiquadHp(
        private val b0: Float,
        private val b1: Float,
        private val b2: Float,
        private val a1: Float,
        private val a2: Float
    ) {
        private var x1 = 0f
        private var x2 = 0f
        private var y1 = 0f
        private var y2 = 0f

        fun step(x: Float): Float {
            val y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x
            y2 = y1
            y1 = y
            return y
        }
    }

    // ~180Hz Butterworth high-pass @ 16kHz
    private val hp1 = BiquadHp(0.9512f, -1.9024f, 0.9512f, -1.8999f, 0.9049f)
    private val hp2 = BiquadHp(0.9512f, -1.9024f, 0.9512f, -1.8999f, 0.9049f)
    private var smoothWet = 0f

    fun process(pcm: ShortArray, len: Int = pcm.size): Float {
        if (len <= 0) return 0f
        var zcr = 0
        var energy = 0.0
        for (i in 1 until len) {
            if ((pcm[i] >= 0 && pcm[i - 1] < 0) || (pcm[i] < 0 && pcm[i - 1] >= 0)) zcr++
            energy += abs(pcm[i].toInt())
        }
        val avgAmp = (energy / len).toFloat()
        val zcrRatio = zcr.toFloat() / len
        // Wind noise has high low-frequency energy with very low zero-crossing rate
        val windScore = if (avgAmp > 900f && zcrRatio < 0.06f) {
            ((avgAmp - 900f) / 2500f).coerceIn(0f, 1f)
        } else 0f

        val targetWet = ((windScore - 0.25f) / 0.35f).coerceIn(0f, 1f)
        for (i in 0 until len) {
            smoothWet += 0.006f * (targetWet - smoothWet)
            val dry = pcm[i].toFloat()
            // Always advance filter state to avoid transient step clicks
            val wet = hp2.step(hp1.step(dry))
            val out = dry * (1f - smoothWet) + wet * smoothWet
            pcm[i] = out.toInt().coerceIn(-32768, 32767).toShort()
        }
        return windScore
    }
}
