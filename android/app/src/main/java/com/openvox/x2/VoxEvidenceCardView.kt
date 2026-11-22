package com.openvox.x2

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator

/**
 * Floating optical-waveguide evidence card for RayNeo X2.
 * Pure black (#000000) background with glowing L-corner brackets (zero grey fog),
 * rounded-clip evidence thumbnail, crossfade carousel, and live TTL countdown bar.
 */
class VoxEvidenceCardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var headerText = ""
    private var bodyLines: List<String> = emptyList()
    private var bitmaps: List<Pair<Bitmap, String>> = emptyList()
    private var activeIdx = 0
    private var ttlProgress = 1f // 1.0 -> 0.0 countdown bar
    private var imgAlpha = 255

    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xDD00F0FF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 2.0f
        strokeCap = Paint.Cap.SQUARE
    }
    private val hairlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x5500E5FF
        style = Paint.Style.STROKE
        strokeWidth = 1.0f
    }
    private val ttlBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x3300E5FF
        style = Paint.Style.FILL
    }
    private val ttlFgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCC00F0FF.toInt()
        style = Paint.Style.FILL
    }
    private val headerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00F0FF.toInt()
        textSize = 11.5f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    }
    private val badgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCC99EEFF.toInt()
        textSize = 10.5f
        typeface = Typeface.MONOSPACE
        textAlign = Paint.Align.RIGHT
    }
    private val captionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF99EEFF.toInt()
        textSize = 10.5f
        typeface = Typeface.MONOSPACE
    }
    private val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFEEEEEE.toInt()
        textSize = 12f
    }
    private val imgPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val imgRect = RectF()
    private val clipPath = Path()

    private var ttlAnimator: ValueAnimator? = null
    private var fadeAnimator: ValueAnimator? = null

    fun showEvidence(
        header: String,
        lines: List<String>,
        decodedImages: List<Pair<Bitmap, String>>,
        ttlMs: Long
    ) {
        recycleBitmaps()
        headerText = header
        bodyLines = lines.take(3)
        bitmaps = decodedImages
        activeIdx = 0
        imgAlpha = 255
        ttlProgress = 1f

        // Smooth slide-in + fade-in from right edge
        animate().cancel()
        visibility = VISIBLE
        alpha = 0f
        translationX = 24f
        animate()
            .alpha(1f)
            .translationX(0f)
            .setDuration(260L)
            .setInterpolator(DecelerateInterpolator())
            .start()

        // Smooth bottom TTL countdown bar
        ttlAnimator?.cancel()
        ttlAnimator = ValueAnimator.ofFloat(1f, 0f).apply {
            duration = ttlMs.coerceAtLeast(2000L)
            interpolator = LinearInterpolator()
            addUpdateListener {
                ttlProgress = it.animatedValue as Float
                invalidate()
            }
            start()
        }
        requestLayout()
        invalidate()
    }

    fun stepImage(delta: Int): Boolean {
        if (bitmaps.size <= 1 || visibility != VISIBLE) return false
        activeIdx = (activeIdx + delta + bitmaps.size) % bitmaps.size
        fadeAnimator?.cancel()
        fadeAnimator = ValueAnimator.ofInt(90, 255).apply {
            duration = 200L
            addUpdateListener {
                imgAlpha = it.animatedValue as Int
                invalidate()
            }
            start()
        }
        return true
    }

    fun imageCount(): Int = bitmaps.size

    fun dismiss() {
        ttlAnimator?.cancel()
        fadeAnimator?.cancel()
        if (visibility != VISIBLE) {
            recycleBitmaps()
            return
        }
        animate().cancel()
        animate()
            .alpha(0f)
            .translationX(16f)
            .setDuration(200L)
            .withEndAction {
                visibility = GONE
                translationX = 0f
                recycleBitmaps()
            }
            .start()
    }

    fun releaseImmediate() {
        ttlAnimator?.cancel()
        fadeAnimator?.cancel()
        animate().cancel()
        visibility = GONE
        recycleBitmaps()
    }

    private fun recycleBitmaps() {
        for ((bmp, _) in bitmaps) {
            if (!bmp.isRecycled) bmp.recycle()
        }
        bitmaps = emptyList()
        activeIdx = 0
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec).coerceAtLeast(240)
        val hasImg = bitmaps.isNotEmpty()
        val imgH = if (hasImg) 148 else 0
        val capH = if (hasImg) 18 else 0
        val textH = bodyLines.size * 18
        val totalH = (42 + imgH + capH + textH + 14).coerceIn(64, 290)
        setMeasuredDimension(w, totalH)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val pad = 12f
        val w = width.toFloat()
        val h = height.toFloat()

        // 1. Tactical floating L-corner brackets (zero solid grey background fog)
        val cLen = 12f
        // Top-left
        canvas.drawLine(2f, 2f, 2f + cLen, 2f, bracketPaint)
        canvas.drawLine(2f, 2f, 2f, 2f + cLen, bracketPaint)
        // Top-right
        canvas.drawLine(w - 2f, 2f, w - 2f - cLen, 2f, bracketPaint)
        canvas.drawLine(w - 2f, 2f, w - 2f, 2f + cLen, bracketPaint)
        // Bottom-left
        canvas.drawLine(2f, h - 2f, 2f + cLen, h - 2f, bracketPaint)
        canvas.drawLine(2f, h - 2f, 2f, h - 2f - cLen, bracketPaint)
        // Bottom-right
        canvas.drawLine(w - 2f, h - 2f, w - 2f - cLen, h - 2f, bracketPaint)
        canvas.drawLine(w - 2f, h - 2f, w - 2f, h - 2f - cLen, bracketPaint)

        // Subtle side hairlines
        canvas.drawLine(2f, cLen + 4f, 2f, h - cLen - 4f, hairlinePaint)
        canvas.drawLine(w - 2f, cLen + 4f, w - 2f, h - cLen - 4f, hairlinePaint)

        // 2. Header + page badge
        var y = 22f
        canvas.drawText(ellipsize(headerText, headerPaint, w - pad * 2 - 36f), pad, y, headerPaint)
        if (bitmaps.size > 1) {
            canvas.drawText("${activeIdx + 1}/${bitmaps.size}", w - pad, y, badgePaint)
        }
        y += 7f
        canvas.drawLine(pad, y, w - pad, y, hairlinePaint)
        y += 8f

        // 3. Rounded-clip evidence photo thumbnail
        if (bitmaps.isNotEmpty()) {
            val (bmp, label) = bitmaps[activeIdx.coerceIn(0, bitmaps.lastIndex)]
            if (!bmp.isRecycled) {
                val maxW = w - pad * 2
                val maxH = 142f
                val scale = minOf(maxW / bmp.width.toFloat(), maxH / bmp.height.toFloat())
                val drawW = bmp.width * scale
                val drawH = bmp.height * scale
                val left = (w - drawW) * 0.5f
                imgRect.set(left, y, left + drawW, y + drawH)

                clipPath.reset()
                clipPath.addRoundRect(imgRect, 6f, 6f, Path.Direction.CW)
                canvas.save()
                canvas.clipPath(clipPath)
                imgPaint.alpha = imgAlpha
                canvas.drawBitmap(bmp, null, imgRect, imgPaint)
                canvas.restore()

                canvas.drawRoundRect(imgRect, 6f, 6f, hairlinePaint)
                y += drawH + 14f
            }
            if (label.isNotEmpty()) {
                canvas.drawText(ellipsize(label, captionPaint, w - pad * 2), pad, y, captionPaint)
                y += 16f
            }
        }

        // 4. Concise evidence text lines
        for (line in bodyLines) {
            if (y > h - 12f) break
            canvas.drawText(ellipsize(line, bodyPaint, w - pad * 2), pad, y, bodyPaint)
            y += 17f
        }

        // 5. Bottom TTL countdown bar
        val barY = h - 5f
        canvas.drawRect(pad, barY, w - pad, barY + 2f, ttlBgPaint)
        val fgW = (w - pad * 2) * ttlProgress.coerceIn(0f, 1f)
        canvas.drawRect(pad, barY, pad + fgW, barY + 2f, ttlFgPaint)
    }

    private fun ellipsize(text: String, paint: Paint, maxW: Float): String {
        if (paint.measureText(text) <= maxW) return text
        var end = text.length
        while (end > 1 && paint.measureText(text.substring(0, end) + "…") > maxW) end--
        return text.substring(0, end) + "…"
    }
}
