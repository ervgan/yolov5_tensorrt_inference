#ifndef YOLOV5_INFERENCE_CONFIG_H_
#define YOLOV5_INFERENCE_CONFIG_H_

/* --------------------------------------------------------
 * These configs are related to tensorrt model, if these are changed,
 * please re-compile and re-serialize the tensorrt model.
 * --------------------------------------------------------*/

#define USE_FP16  // tensorRT uses FP16

namespace yolov5_inference {
// These are used to define input/output tensor names,
// you can set them to whatever you want.
static const char* kInputTensorName = "data";
static const char* kOutputTensorName = "prob";

// Detection model' number of classes
static constexpr int kNumClass = 1;

static constexpr int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
static constexpr int kInputH = 640;
static constexpr int kInputW = 640;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
static constexpr int kMaxNumOutputBbox = 1000;

static constexpr int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in
// yololayer plugin.
static constexpr float kIgnoreThresh = 0.1f;

/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// NMS overlapping thresh and final detection confidence thresh
static const float kNmsThresh = 0.45f;
static const float kConfThresh = 0.5f;

static const int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
static const int kMaxInputImageSize = 4096 * 3112;

enum class States {
  kBuildDetector = 0,  // Serializing file to .engine
  kRunDetector = 1,    // Using .engine for detection
};

}  // namespace yolov5_inference

#endif  // YOLOV5_INFERENCE_CONFIG_H_
