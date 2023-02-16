#include "../include/post_process.h"

#include <sstream>

namespace {
// determines overlap between two boxes
float intersection_over_union(float first_box[4], float second_box[4]) {
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

cv::Rect ScaleRectangle(float bounding_box[4], float scale) {
  float left = bounding_box[0] - bounding_box[2] / 2;
  float top = bounding_box[1] - bounding_box[3] / 2;
  float right = bounding_box[0] + bounding_box[2] / 2;
  float bottom = bounding_box[1] + bounding_box[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left),
                  round(bottom - top));
}

}  // namespace

cv::Rect CreateRectangle(cv::Mat& image, float bounding_box[4]) {
  float rectangle_top_left_x, rectangle_bottom_right_x, rectangle_top_left_y,
      rectangle_bottom_right_y;
  const float width_ratio = kInputW / (image.cols * 1.0);
  const float height_ratio = kInputH / (image.rows * 1.0);

  if (height_ratio > width_ratio) {
    rectangle_top_left_x = bounding_box[0] - bounding_box[2] / 2.f;
    rectangle_bottom_right_x = bounding_box[0] + bounding_box[2] / 2.f;
    rectangle_top_left_y = bounding_box[1] - bounding_box[3] / 2.f -
                           (kInputH - width_ratio * image.rows) / 2;
    rectangle_bottom_right_y = bounding_box[1] + bounding_box[3] / 2.f -
                               (kInputH - width_ratio * image.rows) / 2;
    rectangle_top_left_x = rectangle_top_left_x / width_ratio;
    rectangle_bottom_right_x = rectangle_bottom_right_x / width_ratio;
    rectangle_top_left_y = rectangle_top_left_y / width_ratio;
    rectangle_bottom_right_y = rectangle_bottom_right_y / width_ratio;
  } else {
    rectangle_top_left_x = bounding_box[0] - bounding_box[2] / 2.f -
                           (kInputW - height_ratio * image.cols) / 2;
    rectangle_bottom_right_x = bounding_box[0] + bounding_box[2] / 2.f -
                               (kInputW - height_ratio * image.cols) / 2;
    rectangle_top_left_y = bounding_box[1] - bounding_box[3] / 2.f;
    rectangle_bottom_right_y = bounding_box[1] + bounding_box[3] / 2.f;
    rectangle_top_left_x = rectangle_top_left_x / height_ratio;
    rectangle_bottom_right_x = rectangle_bottom_right_x / height_ratio;
    rectangle_top_left_y = rectangle_top_left_y / height_ratio;
    rectangle_bottom_right_y = rectangle_bottom_right_y / height_ratio;
  }

  return cv::Rect(round(rectangle_top_left_x), round(rectangle_top_left_y),
                  round(rectangle_bottom_right_x - rectangle_top_left_x),
                  round(rectangle_bottom_right_y - rectangle_top_left_y));
}

// uses intersection_over_union to delete duplicate bounding boxes
// for the same detection
void ApplyNonMaxSuppresion(std::vector<Detection>* results, float* output,
                           float confidence_thresh, float nms_thresh) {
  int detection_size = sizeof(Detection) / sizeof(float);
  std::map<float, std::vector<Detection>> detection_map;

  for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) {
    // deletes all boxes with confidence less than confidence threshold
    // set to 0.1 in config.h
    if (output[1 + detection_size * i + 4] <= confidence_thresh) continue;

    Detection detection;
    memcpy(&detection, &output[1 + detection_size * i],
           detection_size * sizeof(float));

    if (detection_map.count(detection.class_id) == 0)
      detection_map.emplace(detection.class_id, std::vector<Detection>());

    detection_map[detection.class_id].push_back(detection);
  }
  for (auto it = detection_map.begin(); it != detection_map.end(); it++) {
    auto& detections = it->second;
    std::sort(detections.begin(), detections.end(), compare);

    for (size_t i = 0; i < detections.size(); ++i) {
      auto& item = detections[i];
      results->push_back(item);

      for (size_t n = i + 1; n < detections.size(); ++n) {
        if (intersection_over_union(item.bounding_box,
                                    detections[n].bounding_box) > nms_thresh) {
          detections.erase(detections.begin() + n);
          --n;
        }
      }
    }
  }
}

void ApplyBatchNonMaxSuppression(
    std::vector<std::vector<Detection>>* result_batch, float* output,
    int batch_size, int output_size, float confidence_thresh,
    float nms_thresh) {
  result_batch->resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    ApplyNonMaxSuppresion(&result_batch[i], &output[i * output_size],
                          confidence_thresh, nms_thresh);
  }
}

void DrawBox(std::vector<cv::Mat>& image_batch,
             std::vector<std::vector<Detection>>& result_batch) {
  for (size_t i = 0; i < image_batch.size(); i++) {
    auto& result = result_batch[i];
    cv::Mat image = image_batch[i];

    for (size_t j = 0; j < result.size(); j++) {
      cv::Rect rectangle = CreateRectangle(image, result[j].bounding_box);
      cv::rectangle(image, rectangle, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(image, Use static_cast<int>(result[j].class_id),
                  cv::Point(rectangle.x, rectangle.y - 1),
                  cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
  }
}
