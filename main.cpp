#include "../include/config.h"
#include "../include/yolo_detector.h"

using yolov5_inference::kGpuId;
using yolov5_inference::States;
using yolov5_inference::YoloDetector;

int main(int argc, char** argv) {
  YoloDetector yoloDetector;
  cudaSetDevice(kGpuId);
  int state = yoloDetector.Init(argc, argv);

  if (state == static_cast<int>(States::kRunDetector)) {
    YoloDetector::ProcessImages(yoloDetector);
    YoloDetector::DrawDetections(yoloDetector);
  }

  return 0;
}
