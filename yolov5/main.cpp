#include <yolo_detector.h>

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);
  YoloDetector yoloDetector();
  yoloDetector.init(argc, argv);
  yoloDetector.ProcessImages();
  yoloDetector.Detect();

  return 0;
}
