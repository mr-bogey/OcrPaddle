//
// Created by Bogey on 2026/1/21.
//

#ifndef OCRPADDLE_NATIVE_H
#define OCRPADDLE_NATIVE_H

#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

inline cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap) {
    AndroidBitmapInfo info;
    int result = AndroidBitmap_getInfo(env, bitmap, &info);
    if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
        return cv::Mat{};
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return cv::Mat{};
    }
    unsigned char *srcData = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, (void **) &srcData);
    cv::Mat mat = cv::Mat::zeros(info.height, info.width, CV_8UC4);
    memcpy(mat.data, srcData, info.height * info.width * 4);
    AndroidBitmap_unlockPixels(env, bitmap);
    cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
    return mat;
}

#endif //OCRPADDLE_NATIVE_H
