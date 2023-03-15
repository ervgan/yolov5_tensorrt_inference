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

void DrawBox(const std::vector<cv::Mat>& image_batch,
             std::vector<std::vector<Detection>>* result_batch);

}  // namespace yolov5_inference

#endif  // YOLOV5_INFERENCE_POST_PROCESS_H_
