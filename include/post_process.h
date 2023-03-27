#ifndef YOLOV5_INFERENCE_POST_PROCESS_H_
#define YOLOV5_INFERENCE_POST_PROCESS_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include "../include/types.h"

namespace yolov5_inference {

// uses IntersectionOverUnion to delete duplicate bounding boxes
// for the same detection
void ApplyBatchNonMaxSuppression(
    float* cpu_buffer, int batch_size, int output_size, float confidence_tresh,
    float nms_thresh, std::vector<std::vector<Detection>>* batch_result);

Detection GetMaxDetection(std::vector<Detection>* results);

void DrawBox(const cv::Mat& image_batch, Detection* detection);

}  // namespace yolov5_inference
