package top.bogey.ocr.baidu.paddle;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import top.bogey.ocr.OcrNativeResult;
import top.bogey.ocr.OcrResult;

public class Ocr {
    private final static String KEYS = "keys_v5.txt";
    private final static String DET = "ocr_det_fp16.tflite";     // 文字区域检测
    private final static String REC = "ocr_rec_fp16.tflite";     // 文字识别
    private final static String NPU_CACHE_DIR = "npu_cache";
    private boolean npuCacheInited = false;

    private long module = 0;

    private static Ocr instance;

    public static Ocr getInstance(Context context) {
        if (instance == null) instance = new Ocr(context);
        return instance;
    }

    public Ocr(Context context) {
        initNpuCache(context);

        String detPath = copyAssetToCache(context, DET);
        String recPath = copyAssetToCache(context, REC);
        String keysPath = copyAssetToCache(context, KEYS);
        module = nativeCreate(detPath, recPath, keysPath);
    }

    private void initNpuCache(Context context) {
        if (npuCacheInited) return;
        File cacheDir = new File(context.getCacheDir(), NPU_CACHE_DIR);
        if (!cacheDir.exists()) cacheDir.mkdirs();
        nativeSetCacheDir(cacheDir.getAbsolutePath());
        npuCacheInited = true;
    }

    public List<OcrResult> runOcr(Bitmap bitmap) {
        if (module != 0) {
            List<OcrResult> results = new ArrayList<>();
            LetterBox letterBox = null;
            if (bitmap.getWidth() < 640 || bitmap.getHeight() < 640) {
                letterBox = LetterBox.create(bitmap, 640, 640);
                bitmap = letterBox.bitmap();
            }
            for (OcrNativeResult nativeResult : nativeRunOcr(module, bitmap)) {
                RectF area = nativeResult.getArea();
                Rect rect = new Rect();
                if (letterBox != null) {
                    float offsetX = letterBox.offsetX();
                    float offsetY = letterBox.offsetY();
                    float scale = letterBox.scale();

                    float left = (area.left - offsetX) / scale;
                    float right = (area.right - offsetX) / scale;
                    float top = (area.top - offsetY) / scale;
                    float bottom = (area.bottom - offsetY) / scale;
                    rect.set((int) left, (int) top, (int) right, (int) bottom);
                } else {
                    rect.set((int) area.left, (int) area.top, (int) area.right, (int) area.bottom);
                }

                OcrResult ocrResult = new OcrResult(rect, nativeResult.getText(), (int) (nativeResult.getSimilar() * 100));
                results.add(ocrResult);
            }
            return results;
        }
        return new ArrayList<>();
    }

    public void releaseModule() {
        if (module != 0) {
            nativeRelease(module);
            module = 0;
        }
        nativeShutdown();
    }

    public static String copyAssetToCache(Context context, String fileName) {
        File cacheDir = context.getCacheDir();
        File cacheFile = new File(cacheDir, fileName);
        if (cacheFile.exists()) return cacheFile.getAbsolutePath();

        try (InputStream inputStream = new BufferedInputStream(context.getAssets().open(fileName));
             OutputStream outputStream = new BufferedOutputStream(new FileOutputStream(cacheFile))) {
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return cacheFile.getAbsolutePath();
    }

    private static native void nativeSetCacheDir(String cacheDir);

    private static native long nativeCreate(String detPath, String recPath, String keysPath);

    private static native void nativeRelease(long module);

    private static native void nativeShutdown();

    private static native OcrNativeResult[] nativeRunOcr(long module, Bitmap bitmap);
}
