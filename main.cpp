#include "../include/config.h"
#include "../include/yolo_detector.h"

using yolov5_inference::kGpuId;
using yolov5_inference::States;
using yolov5_inference::YoloDetector;

int main(int argc, char** argv) {
  YoloDetector yoloDetector;
  cudaSetDevice(kGpuId);
  int state = yoloDetector.Init(argc, argv);

  // state 0 corresponds to serializing file to .engine
  // state 1 corresponds to using .engine for detection
  if (state == 1) {
    yoloDetector.DrawDetection();
  }

  return 0;
}
