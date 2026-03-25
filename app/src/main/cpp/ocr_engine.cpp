/*
 * Copyright (C) 2025 Fleey
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ocr_engine.h"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "litert_config.h"
#include "logging.h"

#define TAG "OcrEngine"

namespace ppocrv5 {

    namespace {

        constexpr AcceleratorType kFallbackChain[] = {
                AcceleratorType::kGpu,
                AcceleratorType::kCpu,
        };
        constexpr int kFallbackChainSize = 2;
        constexpr int kWarmupIterations = 3;
        constexpr int kWarmupImageSize = 128;

        constexpr float kMinBoxArea = 50.0f;
        constexpr float kMinConfidenceThreshold = 0.0f;
        constexpr int kMaxBoxesPerFrame = 50;

        constexpr int kRecInputHeight = 48;

        inline int CalculateTargetWidth(const RotatedRect &box) {
            float src_width = box.width;
            float src_height = box.height;
            if (src_width < src_height) {
                std::swap(src_width, src_height);
            }
            const float aspect_ratio = src_width / std::max(src_height, 1.0f);
            return static_cast<int>(kRecInputHeight * aspect_ratio);
        }

        int GetFallbackStartIndex(AcceleratorType requested) {
            switch (requested) {
                case AcceleratorType::kGpu:
                case AcceleratorType::kNpu:
                    return 0;
                case AcceleratorType::kCpu:
                default:
                    return 1;
            }
        }

        const char *AcceleratorName(AcceleratorType type) {
            switch (type) {
                case AcceleratorType::kNpu:
                    return "NPU";
                case AcceleratorType::kGpu:
                    return "GPU";
                case AcceleratorType::kCpu:
                default:
                    return "CPU";
            }
        }

        inline void SortBoxesByArea(std::vector<RotatedRect> &boxes, std::vector<size_t> &indices) {
            indices.resize(boxes.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::sort(indices.begin(), indices.end(), [&boxes](size_t a, size_t b) {
                return boxes[a].width * boxes[a].height > boxes[b].width * boxes[b].height;
            });
        }

    }  // namespace

    std::unique_ptr<OcrEngine> OcrEngine::Create(
            const std::string &det_model_path,
            const std::string &rec_model_path_small,
            const std::string &rec_model_path_large,
            const std::string &keys_path,
            AcceleratorType accelerator_type) {

        auto engine = std::unique_ptr<OcrEngine>(new OcrEngine());
        int start_index = GetFallbackStartIndex(accelerator_type);

        for (int i = start_index; i < kFallbackChainSize; ++i) {
            AcceleratorType current_accelerator = kFallbackChain[i];
            LOGD(TAG, "Attempting to initialize with %s accelerator",
                 AcceleratorName(current_accelerator));

            auto detector = TextDetector::Create(det_model_path, current_accelerator);
            if (!detector) {
                LOGD(TAG, "TextDetector failed with %s, trying next",
                     AcceleratorName(current_accelerator));
                continue;
            }

            auto recognizer_small = TextRecognizer::Create(
                    rec_model_path_small, keys_path, current_accelerator,
                    RecInputWidth::kW320);
            if (!recognizer_small) {
                LOGD(TAG, "TextRecognizer (320) failed with %s, trying next",
                     AcceleratorName(current_accelerator));
                continue;
            }

            auto recognizer_large = TextRecognizer::Create(
                    rec_model_path_large, keys_path, current_accelerator,
                    RecInputWidth::kW640);
            if (!recognizer_large) {
                LOGD(TAG, "TextRecognizer (640) failed with %s, trying next",
                     AcceleratorName(current_accelerator));
                continue;
            }

            engine->detector_ = std::move(detector);
            engine->recognizer_small_ = std::move(recognizer_small);
            engine->recognizer_large_ = std::move(recognizer_large);
            engine->active_accelerator_ = current_accelerator;

            LOGD(TAG, "OcrEngine initialized with %s accelerator",
                 AcceleratorName(current_accelerator));

            engine->WarmUp();
            return engine;
        }

        LOGE(TAG, "Failed to initialize OcrEngine with any accelerator");
        return nullptr;
    }

    std::vector<OcrResult> OcrEngine::Process(const uint8_t *image_data,
                                              int width, int height, int stride) {
        if (!detector_ || !recognizer_small_ || !recognizer_large_) {
            LOGE(TAG, "OcrEngine not properly initialized");
            return {};
        }

        auto total_start = std::chrono::high_resolution_clock::now();

        float detection_time_ms = 0.0f;
        auto boxes = detector_->Detect(image_data, width, height, stride, &detection_time_ms);
        benchmark_.detection_time_ms = detection_time_ms;

        if (boxes.empty()) {
            auto total_end = std::chrono::high_resolution_clock::now();
            benchmark_.total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    total_end - total_start).count() / 1000.0f;
            benchmark_.recognition_time_ms = 0.0f;
            benchmark_.fps = (benchmark_.total_time_ms > 0.0f) ? (1000.0f / benchmark_.total_time_ms) : 0.0f;
            return {};
        }

        std::vector<RotatedRect> filtered_boxes;
        filtered_boxes.reserve(std::min(boxes.size(), static_cast<size_t>(kMaxBoxesPerFrame)));

        for (const auto &box: boxes) {
            if (box.width * box.height >= kMinBoxArea) {
                filtered_boxes.push_back(box);
                if (filtered_boxes.size() >= kMaxBoxesPerFrame) break;
            }
        }

        if (filtered_boxes.empty()) {
            LOGD(TAG, "After filtering, no boxes left");
            auto total_end = std::chrono::high_resolution_clock::now();
            benchmark_.total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    total_end - total_start).count() / 1000.0f;
            benchmark_.recognition_time_ms = 0.0f;
            benchmark_.fps = (benchmark_.total_time_ms > 0.0f) ? (1000.0f / benchmark_.total_time_ms) : 0.0f;
            return {};
        }

        std::vector<size_t> sorted_indices;
        SortBoxesByArea(filtered_boxes, sorted_indices);

        std::vector<OcrResult> results;
        results.reserve(filtered_boxes.size());

        auto rec_start = std::chrono::high_resolution_clock::now();

        for (size_t idx: sorted_indices) {
            const auto &box = filtered_boxes[idx];
            float rec_time_ms = 0.0f;

            int target_width = CalculateTargetWidth(box);
            TextRecognizer *recognizer = nullptr;
            if (target_width <= 320) {
                recognizer = recognizer_small_.get();
                LOGD(TAG, "Using 320 model: box_width=%.1f, target_width=%d", box.width, target_width);
            } else {
                recognizer = recognizer_large_.get();
                LOGD(TAG, "Using 640 model: box_width=%.1f, target_width=%d", box.width, target_width);
            }

            auto rec_result = recognizer->Recognize(image_data, width, height, stride,
                                                     box, &rec_time_ms);

            if (!rec_result.text.empty() && rec_result.confidence >= kMinConfidenceThreshold) {
                OcrResult result;
                result.text = std::move(rec_result.text);
                result.confidence = rec_result.confidence;
                result.box = box;
                results.push_back(std::move(result));
            }
        }

        auto rec_end = std::chrono::high_resolution_clock::now();
        benchmark_.recognition_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                rec_end - rec_start).count() / 1000.0f;

        auto total_end = std::chrono::high_resolution_clock::now();
        benchmark_.total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                total_end - total_start).count() / 1000.0f;
        benchmark_.fps = (benchmark_.total_time_ms > 0.0f) ? (1000.0f / benchmark_.total_time_ms) : 0.0f;

        LOGD(TAG, "OCR: %zu/%zu results, det=%.1fms, rec=%.1fms (%.1fms/box), total=%.1fms",
             results.size(), filtered_boxes.size(),
             benchmark_.detection_time_ms, benchmark_.recognition_time_ms,
             filtered_boxes.size() > 0 ? benchmark_.recognition_time_ms / filtered_boxes.size() : 0.0f,
             benchmark_.total_time_ms);

        std::sort(results.begin(), results.end(), [](const OcrResult &a, const OcrResult &b) {
            constexpr float kLineThreshold = 20.0f;
            if (std::abs(a.box.center_y - b.box.center_y) < kLineThreshold) {
                return a.box.center_x < b.box.center_x;
            }
            return a.box.center_y < b.box.center_y;
        });

        return results;
    }

    Benchmark OcrEngine::GetBenchmark() const {
        return benchmark_;
    }

    AcceleratorType OcrEngine::GetActiveAccelerator() const {
        return active_accelerator_;
    }

    void OcrEngine::WarmUp() {
        LOGD(TAG, "Starting warm-up (%d iterations)...", kWarmupIterations);

        std::vector<uint8_t> dummy_image(kWarmupImageSize * kWarmupImageSize * 4, 128);
        for (int i = 0; i < kWarmupImageSize * kWarmupImageSize; ++i) {
            dummy_image[i * 4 + 0] = static_cast<uint8_t>((i * 7) % 256);
            dummy_image[i * 4 + 1] = static_cast<uint8_t>((i * 11) % 256);
            dummy_image[i * 4 + 2] = static_cast<uint8_t>((i * 13) % 256);
            dummy_image[i * 4 + 3] = 255;
        }

        for (int iter = 0; iter < kWarmupIterations; ++iter) {
            float detection_time_ms = 0.0f;
            detector_->Detect(dummy_image.data(), kWarmupImageSize, kWarmupImageSize,
                              kWarmupImageSize * 4, &detection_time_ms);
        }

        LOGD(TAG, "Warm-up completed (accelerator: %s)", AcceleratorName(active_accelerator_));
    }

}  // namespace ppocrv5
