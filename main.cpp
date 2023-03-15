#include "../include/config.h"
#include "../include/yolo_detector.h"

using yolov5_inference::kGpuId;
using yolov5_inference::States;
using yolov5_inference::devel::DrawDetections;
using yolov5_inference::devel::ProcessImages;
using yolov5_inference::YoloDetector::Init;

int main(int argc, char** argv) {
  yolov5_inference::YoloDetector yoloDetector;
  cudaSetDevice(kGpuId);
  int state = yoloDetector.Init(argc, argv);

  if (state == static_cast<int>(States::kRunDetector)) {
    ProcessImages(yoloDetector);
    DrawDetections(yoloDetector);
  }

  return 0;
}
