#include "config.h"
#include "yolo_detector.h"

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);
  YoloDetector yoloDetector;
  yoloDetector.Init(argc, argv);
  yoloDetector.ProcessImages();
  yoloDetector.DrawDetections();

  return 0;
}
