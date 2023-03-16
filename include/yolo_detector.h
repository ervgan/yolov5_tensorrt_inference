#ifndef YOLOV5_INFERENCE_YOLO_DETECTOR_H_
#define YOLOV5_INFERENCE_YOLO_DETECTOR_H_
#include <memory>
#include <string>
#include <vector>

#include "../include/model.h"

// Compiled representation of a neural network
// contains topology, layer config, device memory alloc and inference methods
using nvinfer1::ICudaEngine;
// Context for performance inference on a ICudaEngine
// contains methods for setting and retrieving CUDA stream
using nvinfer1::IExecutionContext;
// Factory object to create a ICudaEngine object from serialized .engine
// files
using nvinfer1::createInferRuntime;
using nvinfer1::IRuntime;
// Creates and optimizes TensorRT networks (input, output tensors)
// used to create a TensorRTengine from a network definition
using nvinfer1::createInferBuilder;
using nvinfer1::IBuilder;
// Settings for IBuilder (max batch size, etc)
using nvinfer1::IBuilderConfig;
// host-side memory buffer for exchanging data between CPU and GPU
// allocated and deallocates memory used by TensorRT
using nvinfer1::IHostMemory;
// represents data types of tensors
using nvinfer1::DataType;

namespace yolov5_inference {
class YoloDetector {
 public:
  YoloDetector();

  ~YoloDetector();

  int Init(int argc, char** argv);

  void PrepareMemoryBuffers(ICudaEngine* engine, float** gpu_input_buffer,
                            float** gpu_output_buffer,
                            std::unique_ptr<float[]>& cpu_output_buffer);

  void SerializeEngine(unsigned int max_batch_size, const float& depth_multiple,
                       const float& width_multiple, const std::string& wts_file,
                       const std::string& engine_file);

  void DeserializeEngine(const std::string& engine_file, IRuntime** runtime,
                         ICudaEngine** engine, IExecutionContext** context);

  void RunInference(IExecutionContext* context, const cudaStream_t& stream,
                    void** gpu_buffers, float* output, int batch_size);

  void DrawDetections();
  void ProcessImages();

 private:
  std::string wts_file_ = "";
  std::string engine_file_ = "";
  float depth_multiple_ = 0.0f;
  float width_multiple_ = 0.0f;
  std::string image_directory_ = "";
  IRuntime* runtime_ = nullptr;
  ICudaEngine* engine_ = nullptr;
  IExecutionContext* context_ = nullptr;
  float* gpu_buffers_[2];
  // float* cpu_output_buffer_ = nullptr;
  std::unique_ptr<float[]> cpu_output_buffer_ =
      std::make_unique<float[]>(kBatchSize * kOutputSize);
  cudaStream_t stream_;
  // this will not be needed for live detection
  std::vector<std::string> file_names_;
};

}  // namespace yolov5_inference

#endif  // YOLOV5_INFERENCE_YOLO_DETECTOR_H_
