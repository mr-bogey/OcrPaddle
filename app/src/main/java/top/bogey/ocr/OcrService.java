package top.bogey.ocr;

import android.app.Service;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.IBinder;
import android.os.RemoteException;

import java.util.ArrayList;
import java.util.List;

public class OcrService extends Service {
    static {
        System.loadLibrary("native");
    }

    private final IOcr.Stub binder = new IOcr.Stub() {
        @Override
        public void runOcr(Bitmap bitmap, IOcrCallback callback) throws RemoteException {
            OcrNativeResult[] nativeResults = runModel(bitmap);
            List<OcrResult> results = new ArrayList<>();
            if (nativeResults == null || nativeResults.length == 0) {
                callback.onResult(results);
                return;
            }

            for (OcrNativeResult nativeResult : nativeResults) {
                RectF area = nativeResult.getArea();
                Rect rect = new Rect();
                rect.set((int) area.left, (int) area.top, (int) area.right, (int) area.bottom);
                OcrResult ocrResult = new OcrResult(rect, nativeResult.getText(), (int) (nativeResult.getSimilar() * 100));
                results.add(ocrResult);
            }
            callback.onResult(results);
        }
    };

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        loadModel(getAssets());
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        releaseModel();
    }

    private native boolean loadModel(AssetManager manager);

    private native void releaseModel();

    private native OcrNativeResult[] runModel(Bitmap bitmap);
}
