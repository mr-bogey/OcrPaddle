package top.bogey.ocr;

import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.IBinder;
import android.os.RemoteException;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import top.bogey.ocr.baidu.paddle.Ocr;

public class OcrService extends Service {
    static {
        System.loadLibrary("native");
    }

    private final IOcr.Stub binder = new IOcr.Stub() {
        @Override
        public void runOcr(Bitmap bitmap, IOcrCallback callback) {
            try(ExecutorService executor = Executors.newSingleThreadExecutor()) {
                executor.execute(() -> {
                    List<OcrResult> results = Ocr.getInstance(OcrService.this).runOcr(bitmap);
                    try {
                        callback.onResult(results);
                    } catch (RemoteException e) {
                        throw new RuntimeException(e);
                    }
                });
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    };

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Ocr.getInstance(this).releaseModule();
    }
}
