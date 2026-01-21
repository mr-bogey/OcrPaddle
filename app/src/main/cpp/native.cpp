// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <sstream>

#include <platform.h>
#include <benchmark.h>

#include "ppocr.h"
#include "native.h"
#include "ppocr_dict.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON

#include <arm_neon.h>

#endif // __ARM_NEON

static PPOCR *g_ppocr = nullptr;
static ncnn::Mutex lock;

extern "C" {
JNIEXPORT jboolean JNICALL
Java_top_bogey_ocr_OcrService_loadModel(JNIEnv *env, jobject thiz, jobject manager) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel");

    AAssetManager *mgr = AAssetManager_fromJava(env, manager);
    ncnn::create_gpu_instance();

    std::string det_param_path = std::string("ppocr_v5_mobile_det.ncnn.param");
    std::string det_model_path = std::string("ppocr_v5_mobile_det.ncnn.bin");
    std::string rec_param_path = std::string("ppocr_v5_mobile_rec.ncnn.param");
    std::string rec_model_path = std::string("ppocr_v5_mobile_rec.ncnn.bin");

    {
        ncnn::MutexLockGuard g(lock);
        {
            if (!g_ppocr) {
                g_ppocr = new PPOCR;
                g_ppocr->load(mgr, det_param_path.c_str(), det_model_path.c_str(), rec_param_path.c_str(), rec_model_path.c_str(), false, false);
            }
        }
    }

    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_top_bogey_ocr_OcrService_releaseModel(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "releaseModel");
    {
        ncnn::destroy_gpu_instance();
        delete g_ppocr;
        g_ppocr = nullptr;
    }
}

JNIEXPORT jobjectArray JNICALL
Java_top_bogey_ocr_OcrService_runModel(JNIEnv *env, jobject thiz, jobject bitmap) {
    cv::Mat bgr = bitmap_to_cv_mat(env, bitmap);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<Object> objects;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "start detect");
    g_ppocr->detect(rgb, objects);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "end detect");
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "start recognize");
#pragma omp parallel for num_threads(ncnn::get_big_cpu_count()) schedule(dynamic)
    for (auto &item: objects) {
        g_ppocr->recognize(rgb, item);
    }
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "end recognize");

    // 获取OcrNativeResult类
    jclass cls_OcrNativeResult = env->FindClass("top/bogey/ocr/OcrNativeResult");
    jmethodID constructor = env->GetMethodID(cls_OcrNativeResult, "<init>", "(Ljava/lang/String;FFFFFF)V");

    // 创建结果数组
    jobjectArray resultArray = env->NewObjectArray(static_cast<jsize>(objects.size()), cls_OcrNativeResult, nullptr);

    for (size_t i = 0; i < objects.size(); ++i) {
        const Object &obj = objects[i];

        std::string text;
        for (auto ch: obj.text) {
            if (ch.id > character_dict_size) {
                if (!text.empty() && text.back() != ' ') {
                    text += ' ';
                    continue;
                }
            }
            text += character_dict[ch.id];
        }
        jstring jtext = env->NewStringUTF(text.c_str());

        // 获取旋转矩形参数
        cv::Point2f center = obj.rotatedRect.center;
        float width = obj.rotatedRect.size.width;
        float height = obj.rotatedRect.size.height;
        float angle = obj.rotatedRect.angle;

        // 创建OcrNativeResult实例
        jobject ocr_result = env->NewObject(cls_OcrNativeResult, constructor, jtext, obj.prob, center.x, center.y, width, height, angle);

        // 将实例添加到数组
        env->SetObjectArrayElement(resultArray, static_cast<jsize>(i), ocr_result);

        // 清理局部引用
        env->DeleteLocalRef(jtext);
        env->DeleteLocalRef(ocr_result);
    }

    env->DeleteLocalRef(cls_OcrNativeResult);

    return resultArray;
}
}
