package top.bogey.ocr;

import android.graphics.RectF;

public class OcrNativeResult {
    private final String text;
    private final float similar;
    private final float centerX;
    private final float centerY;
    private final float width;
    private final float height;
    private final float angle;

    public OcrNativeResult(String text, float similar, float centerX, float centerY, float width, float height, float angle) {
        this.text = text;
        this.similar = similar;
        this.centerX = centerX;
        this.centerY = centerY;
        this.width = width;
        this.height = height;
        this.angle = angle;
    }

    public String getText() {
        return text;
    }

    public float getSimilar() {
        return similar;
    }

    public RectF getArea() {
        // 半宽半高
        float hw = width / 2f;
        float hh = height / 2f;

        // 角度转弧度
        double rad = Math.toRadians(angle);
        float cos = (float) Math.cos(rad);
        float sin = (float) Math.sin(rad);

        // 四个角点（相对于中心）
        float[] xs = new float[]{-hw, hw, hw, -hw};
        float[] ys = new float[]{-hh, -hh, hh, hh};

        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE;
        float maxY = Float.MIN_VALUE;

        for (int i = 0; i < 4; i++) {
            // 旋转
            float rx = xs[i] * cos - ys[i] * sin;
            float ry = xs[i] * sin + ys[i] * cos;

            // 平移到世界坐标
            float x = centerX + rx;
            float y = centerY + ry;

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        return new RectF(minX, minY, maxX, maxY);
    }
}
