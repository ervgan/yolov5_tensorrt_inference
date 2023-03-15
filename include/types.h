#ifndef YOLOV5_INFERENCE_TYPES_H_
#define YOLOV5_INFERENCE_TYPES_H_

#include "../include/config.h"

using yolov5_inference::kNumAnchor;

struct YoloKernel {
  int width;
  int height;
  float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
  float bounding_box_px[4];  // center_x center_y w h in pixels
  float confidence;          // bbox_conf * cls_conf
  float class_id;
  float mask[32];
};

#endif  // YOLOV5_INFERENCE_TYPES_H_
