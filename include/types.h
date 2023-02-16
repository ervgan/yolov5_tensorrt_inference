#pragma once

#include "CPP_INF_TENSORRT/include/config.h"

struct YoloKernel {
  int width;
  int height;
  float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
  float bounding_box[4];  // center_x center_y w h
  float confidence;       // bbox_conf * cls_conf
  float class_id;
  float mask[32];
};
