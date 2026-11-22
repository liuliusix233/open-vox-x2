package com.openvox.x2

import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.AttributeSet
import android.view.Gravity
import android.view.animation.DecelerateInterpolator
import android.widget.TextView

/**
 * Animated waveguide bottom subtitle strip with smooth glide-in and auto-fadeout.
 * Distinguishes user utterances (ice cyan) from assistant replies (crisp white)
 * with subtle drop shadow for outdoor legibility and zero burn-in.
 */
class VoxSubtitleView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : TextView(context, attrs) {

    private val handler = Handler(Looper.getMainLooper())
    private val autoFadeTask = Runnable {
        animate().cancel()
        animate().alpha(0f).setDuration(450L).start()
    }

    init {
        textSize = 15f
        maxLines = 2
        ellipsize = TextUtils.TruncateAt.END
        gravity = Gravity.CENTER
        setPadding(26, 8, 26, 18)
        setShadowLayer(4f, 0f, 1f, Color.BLACK)
        alpha = 0f
    }

    fun showTranscript(role: String, content: String, autoFadeMs: Long = 6500L) {
        val clean = content.trim()
        if (clean.isEmpty()) return
        handler.removeCallbacks(autoFadeTask)

        val isUser = role.equals("user", ignoreCase = true)
        setTextColor(if (isUser) 0xFF7CE8FF.toInt() else 0xFFFFFFFF.toInt())
        text = if (isUser) "› $clean" else clean

        animate().cancel()
        if (alpha < 0.2f) {
            translationY = 10f
            animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(220L)
                .setInterpolator(DecelerateInterpolator())
                .start()
        } else {
            alpha = 1f
            translationY = 0f
        }
        handler.postDelayed(autoFadeTask, autoFadeMs)
    }

    fun clearImmediate() {
        handler.removeCallbacks(autoFadeTask)
        animate().cancel()
        alpha = 0f
        text = ""
    }

    override fun onDetachedFromWindow() {
        handler.removeCallbacks(autoFadeTask)
        super.onDetachedFromWindow()
    }
}
