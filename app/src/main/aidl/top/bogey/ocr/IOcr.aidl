// IOcr.aidl
package top.bogey.ocr;

import android.graphics.Bitmap;

import top.bogey.ocr.IOcrCallback;

interface IOcr {
    void runOcr(in Bitmap bitmap, in IOcrCallback callback);
}