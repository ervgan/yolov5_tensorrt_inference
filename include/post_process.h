#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "../include/types.h"

cv::Rect CreateRectangle(const cv::Mat &image, float bounding_box[4]);

void ApplyNonMaxSuppresion(std::vector<Detection> *result, float *output,
                           float confidence_tresh, float nms_thresh);

void ApplyBatchNonMaxSuppression(
    std::vector<std::vector<Detection>> *batch_result, float *output,
    int batch_size, int output_size, float confidence_tresh, float nms_thresh);

Detection GetMaxDetection(std::vector<Detection> *results);

void DrawBox(const cv::Mat &image_batch, Detection *detection);
