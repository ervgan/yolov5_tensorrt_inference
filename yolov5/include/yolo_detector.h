
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "model.h"
#include "types.h"

class YoloDetector {
 public:
  YoloDetector();

  ~YoloDetector();

  void Init(int argc, char** argv);

  void PrepareMemoryBuffers(ICudaEngine* engine, float** gpu_input_buffer,
                            float** gpu_output_buffer,
                            float** cpu_output_buffer);

  void SerializeEngine(unsigned int max_batch_size, float& depth_multiple,
                       float& width_multiple, std::string& wts_file,
                       std::string& engine_file);

  void DeserializeEngine(std::string& engine_file, IRuntime** runtime,
                         ICudaEngine** engine, IExecutionContext** context);

  void RunInference(IExecutionContext& context, cudaStream_t& stream,
                    void** gpu_buffers, float* output, int batch_size);

  void DrawDetections();

  void ProcessImages();

 private:
  std::string wts_file_ = "";
  std::string engine_file_ = "";
  float depth_multiple_ = 0.0f;
  float width_multiple = 0.0f;
  std::string image_directory_;
  IRuntime* runtime_ = nullptr;
  ICudaEngine* engine_ = nullptr;
  IExecutionContext* context_ = nullptr;
  float* gpu_buffers_[2];
  float* cpu_output_buffer_ = nullptr;
  cudaStream_t stream_;
  // this will not be needed for live detection
  std::vector<std::string> file_names_;
  std::vector<std::vector<Detection>> result_batch_;
}
