#include "config.h"
#include "yolo_detector.h"

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);
  YoloDetector yoloDetector;
  int state = yoloDetector.Init(argc, argv);
  // state 0 corresponds to serializing file to .engine
  // state 1 corresponds to using .engine for detection
  if (state == 1) {
    yoloDetector.ProcessImages();
    yoloDetector.DrawDetections();
  }
  return 0;
}
