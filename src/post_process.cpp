#include "../include/post_process.h"

#include <sstream>

namespace yolov5_inference {

namespace {
// determines overlap between two boxes
float IntersectionOverUnion(float first_box[4], float second_box[4]) {
  float intersection_box[] = {
      (std::max)(first_box[0] - first_box[2] / 2.f,
                 second_box[0] - second_box[2] / 2.f),  // left
      (std::min)(first_box[0] + first_box[2] / 2.f,
                 second_box[0] + second_box[2] / 2.f),  // right
      (std::max)(first_box[1] - first_box[3] / 2.f,
                 second_box[1] - second_box[3] / 2.f),  // top
      (std::min)(first_box[1] + first_box[3] / 2.f,
                 second_box[1] + second_box[3] / 2.f),  // bottom
  };

  if (intersection_box[2] > intersection_box[3] ||
      intersection_box[0] > intersection_box[1])
    return 0.0f;

  float intersection_box_area = (intersection_box[1] - intersection_box[0]) *
                                (intersection_box[3] - intersection_box[2]);
  return intersection_box_area /
         (first_box[2] * first_box[3] + second_box[2] * second_box[3] -
          intersection_box_area);
}

bool compare(const Detection& first_detection,
             const Detection& second_detection) {
  return first_detection.confidence > second_detection.confidence;
}

cv::Rect ScaleRectangle(float bounding_box_px[4], float scale) {
  float left = bounding_box_px[0] - bounding_box_px[2] / 2;
  float top = bounding_box_px[1] - bounding_box_px[3] / 2;
  float right = bounding_box_px[0] + bounding_box_px[2] / 2;
  float bottom = bounding_box_px[1] + bounding_box_px[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left),
                  round(bottom - top));
}

// Creates an openCV rectangle object resized to the appropriate dimensions
// and will be drawn on the image later on
cv::Rect CreateRectangle(const cv::Mat& image, float bounding_box_px[4]) {
  float rectangle_bottom_left_x, rectangle_top_right_x, rectangle_bottom_left_y,
      rectangle_top_right_y;
  const float width_ratio = kInputW / (image.cols * 1.0);
  const float height_ratio = kInputH / (image.rows * 1.0);
  const float height_adjustment = (kInputH - width_ratio * image.rows) / 2;
  const float width_adjustment = (kInputW - height_ratio * image.cols) / 2;

  if (height_ratio > width_ratio) {
    rectangle_bottom_left_x = bounding_box_px[0] - bounding_box_px[2] / 2.f;
    rectangle_top_right_x = bounding_box_px[0] + bounding_box_px[2] / 2.f;
    rectangle_bottom_left_y =
        bounding_box_px[1] - bounding_box_px[3] / 2.f - height_adjustment;
    rectangle_top_right_y =
        bounding_box_px[1] + bounding_box_px[3] / 2.f - height_adjustment;
    rectangle_bottom_left_x = rectangle_bottom_left_x / width_ratio;
    rectangle_top_right_x = rectangle_top_right_x / width_ratio;
    rectangle_bottom_left_y = rectangle_bottom_left_y / width_ratio;
    rectangle_top_right_y = rectangle_top_right_y / width_ratio;
  } else {
    rectangle_bottom_left_x =
        bounding_box_px[0] - bounding_box_px[2] / 2.f - width_adjustment;
    rectangle_top_right_x =
        bounding_box_px[0] + bounding_box_px[2] / 2.f - width_adjustment;
    rectangle_bottom_left_y = bounding_box_px[1] - bounding_box_px[3] / 2.f;
    rectangle_top_right_y = bounding_box_px[1] + bounding_box_px[3] / 2.f;
    rectangle_bottom_left_x = rectangle_bottom_left_x / height_ratio;
    rectangle_top_right_x = rectangle_top_right_x / height_ratio;
    rectangle_bottom_left_y = rectangle_bottom_left_y / height_ratio;
    rectangle_top_right_y = rectangle_top_right_y / height_ratio;
  }

  return cv::Rect(round(rectangle_bottom_left_x),
                  round(rectangle_bottom_left_y),
                  round(rectangle_top_right_x - rectangle_bottom_left_x),
                  round(rectangle_top_right_y - rectangle_bottom_left_y));
}

// uses IntersectionOverUnion to delete duplicate bounding boxes
// for the same detection
void ApplyNonMaxSuppresion(float* cpu_buffer, float confidence_thresh,
                           float nms_thresh, std::vector<Detection>* results) {
  CHECK_NOTNULL(cpu_buffer);
  int detection_size = sizeof(Detection) / sizeof(float);
  std::unordered_map<float, std::vector<Detection>> detection_map;

  for (int i = 0; i < cpu_buffer[0] && i < kMaxNumOutputBbox; ++i) {
    // deletes all boxes with confidence less than confidence threshold
    // set to 0.1 in config.h
    if (cpu_buffer[1 + detection_size * i + 4] <= confidence_thresh) continue;

    Detection detection;
    memcpy(&detection, &cpu_buffer[1 + detection_size * i],
           detection_size * sizeof(float));

    if (detection_map.count(detection.class_id) == 0)
      detection_map.emplace(detection.class_id, std::vector<Detection>());

    detection_map[detection.class_id].push_back(detection);
  }
  for (auto it = detection_map.begin(); it != detection_map.end(); ++it) {
    auto& detections = it->second;
    std::sort(detections.begin(), detections.end(), compare);

    for (size_t i = 0; i < detections.size(); ++i) {
      auto& item = detections[i];
      results->push_back(item);

      for (size_t n = i + 1; n < detections.size(); ++n) {
        if (IntersectionOverUnion(item.bounding_box_px,
                                  detections[n].bounding_box_px) > nms_thresh) {
          detections.erase(detections.begin() + n);
          --n;
        }
      }
    }
  }
}
}  // namespace

void ApplyBatchNonMaxSuppression(
    float* cpu_buffer, int batch_size, int output_size, float confidence_thresh,
    float nms_thresh, std::vector<std::vector<Detection>>* result_batch) {
  CHECK_NOTNULL(cpu_buffer);
  result_batch->resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ApplyNonMaxSuppresion(&(*result_batch)[i], &cpu_buffer[i * output_size],
                          confidence_thresh, nms_thresh);
  }
}

void DrawBox(const std::vector<cv::Mat>& image_batch,
             std::vector<std::vector<Detection>>* result_batch) {
  for (size_t i = 0; i < image_batch.size(); ++i) {
    std::vector<Detection>& result = (*result_batch)[i];
    cv::Mat image = image_batch[i];

    for (size_t j = 0; j < result.size(); ++j) {
      cv::Rect rectangle = CreateRectangle(image, result[j].bounding_box_px);
      // (0x27, 0xC1, 0x36) represent RGB code for green
      cv::rectangle(image, rectangle, cv::Scalar(0x27, 0xC1, 0x36), 2);
      int rounded_confidence =
          static_cast<int>(std::round(result[j].confidence * 100));
      float result_confidence = static_cast<float>(rounded_confidence) / 100.0f;
      // (0xFF, 0xFF, 0xFF) represent RGB color for white
      cv::putText(image, std::to_string(result_confidence).substr(0, 4),
                  cv::Point(rectangle.x, rectangle.y - 1),
                  cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
  }
}

}  // namespace yolov5_inference
