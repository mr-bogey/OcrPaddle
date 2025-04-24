package top.bogey.ocr.baidu.paddle;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import top.bogey.ocr.OcrResult;

public class Ocr {
    private final static String KEYS = "keys.txt";
    private final static String DET = "det.nb";     // 文字区域检测
    private final static String CLS = "cls.nb";     // 文字方向分类
    private final static String REC = "rec.nb";     // 文字识别
    private final static List<String> labels = new ArrayList<>();
    private long module = 0;

    private static Ocr instance;

    public static Ocr getInstance(Context context) {
        if (instance == null) instance = new Ocr(context);
        return instance;
    }

    public Ocr(Context context) {
        loadModule(context);
    }

    private void loadModule(Context context) {
        String path = context.getCacheDir().getAbsolutePath();
        copyFileFromAssets(context, DET, path + File.separator + DET);
        copyFileFromAssets(context, CLS, path + File.separator + CLS);
        copyFileFromAssets(context, REC, path + File.separator + REC);

        module = initModule(path + File.separator + DET, path + File.separator + REC, path + File.separator + CLS);

        labels.add("black");
        try (InputStream inputStream = context.getAssets().open(KEYS)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        labels.add(" ");
    }

    public List<OcrResult> runOcr(Bitmap image) {
        List<OcrResult> results = new ArrayList<>();
        if (module == 0) return results;
        if (image == null) return results;

        float[] ocrData = forward(module, image, Math.max(image.getWidth(), image.getHeight()), 1, 0, 1);
        int begin = 0;
        while (begin < ocrData.length) {
            int pointNum = Math.round(ocrData[begin]);
            int wordNum = Math.round(ocrData[begin + 1]);
            int similar = Math.round(ocrData[begin + 2] * 100);

            int current = begin + 3;
            Rect rect = new Rect();
            boolean init = false;
            for (int i = 0; i < pointNum; i++) {
                int x = Math.round(ocrData[current + i * 2]);
                int y = Math.round(ocrData[current + i * 2 + 1]);
                if (init) {
                    rect.left = Math.min(x, rect.left);
                    rect.right = Math.max(x, rect.right);
                    rect.top = Math.min(y, rect.top);
                    rect.bottom = Math.max(y, rect.bottom);
                } else {
                    rect.set(x, y, x, y);
                    init = true;
                }
            }

            StringBuilder builder = new StringBuilder();
            current += (pointNum * 2);
            for (int i = 0; i < wordNum; i++) {
                int index = Math.round(ocrData[current + i]);
                builder.append(labels.get(index));
            }

            results.add(new OcrResult(rect, builder.toString(), similar));

            begin += (3 + pointNum * 2 + wordNum + 2);
        }

        results.sort((o1, o2) -> {
            int topOffset = -(o1.getArea().top - o2.getArea().top);
            if (Math.abs(topOffset) <= 10) {
                return -(o1.getArea().left - o2.getArea().left);
            } else {
                return topOffset;
            }
        });

        return results;
    }

    public void releaseModule() {
        if (module != 0) {
            releaseModule(module);
        }
    }

    public static void copyFileFromAssets(Context context, String from, String to) {
        if (from == null || from.isEmpty() || to == null || to.isEmpty()) return;
        try (InputStream inputStream = new BufferedInputStream(context.getAssets().open(from));
             OutputStream outputStream = new BufferedOutputStream(new FileOutputStream(to))) {
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static native long initModule(String det, String rec, String cls);

    private static native float[] forward(long module, Bitmap image, int size, int det, int cls, int rec);

    private static native void releaseModule(long module);
}
