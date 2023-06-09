#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "../include/types.h"

cv::Rect CreateRectangle(const cv::Mat &image, float bounding_box[4]);

void ApplyNonMaxSuppresion(float *cpu_input, float confidence_tresh,
                           float nms_thresh, std::vector<Detection> *result);

void ApplyBatchNonMaxSuppression(
    float *cpu_input, int batch_size, int output_size, float confidence_tresh,
    float nms_thresh, std::vector<std::vector<Detection>> *result_batch);

void DrawBox(const std::vector<cv::Mat> &image_batch,
             std::vector<std::vector<Detection>> *result_batch);
