// IOcrCallback.aidl
package top.bogey.ocr;

parcelable OcrResult;

interface IOcrCallback {
    void onResult(in List<OcrResult> result);
}