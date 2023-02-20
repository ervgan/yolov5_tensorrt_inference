
#include <string>
#include <vector>

#include "../include/model.h"

using nvinfer1::cudaStream_t;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IRuntime;

class YoloDetector {
 public:
  YoloDetector();

  ~YoloDetector();

  int Init(int argc, char **argv);

  void PrepareMemoryBuffers(ICudaEngine *engine, float **gpu_input_buffer,
                            float **gpu_output_buffer,
                            float **cpu_output_buffer);

  void SerializeEngine(unsigned int max_batch_size, const float &depth_multiple,
                       const float &width_multiple, const std::string &wts_file,
                       const std::string &engine_file);

  void DeserializeEngine(const std::string &engine_file, IRuntime **runtime,
                         ICudaEngine **engine, IExecutionContext **context);

  void RunInference(const IExecutionContext &context,
                    const cudaStream_t &stream, void **gpu_buffers,
                    float *output, int batch_size);

  void DrawDetections();

  void ProcessImages();

 private:
  std::string wts_file_ = "";
  std::string engine_file_ = "";
  float depth_multiple_ = 0.0f;
  float width_multiple_ = 0.0f;
  std::string image_directory_;
  IRuntime *runtime_ = nullptr;
  ICudaEngine *engine_ = nullptr;
  IExecutionContext *context_ = nullptr;
  float *gpu_buffers_[2];
  float *cpu_output_buffer_ = nullptr;
  cudaStream_t stream_;
  // this will not be needed for live detection
  std::vector<std::string> file_names_;
};
