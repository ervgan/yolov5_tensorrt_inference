#pragma once

#include <opencv2/opencv.hpp>

#include "types.h"

cv::Rect CreateRectangle(cv::Mat& image, float bounding_box[4]);

void ApplyNonMaxSuppresion(std::vector<Detection>& result, float* output,
                           float confidence_tresh, float nms_thresh);

void ApplyBatchNonMaxSuppression(
    std::vector<std::vector<Detection>>& batch_result, float* output,
    int batch_size, int output_size, float confidence_tresh, float nms_thresh);

void DrawBox(std::vector<cv::Mat>& image_batch,
             std::vector<std::vector<Detection>>& result_batch);
