#include <dirent.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "post_process.h"
#include "pre_process.h"

using namespace nvinfer1;

namespace {

Logger tensorrt_logger;
const int kOutputSize =
    kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool ParseArgs(int argc, char** argv, std::string& wts_file,
               std::string& engine_file, float& depth_multiple,
               float& width_multiple, std::string& image_directory) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts_file = std::string(argv[2]);
    engine_file = std::string(argv[3]);
    auto net = std::string(argv[4]);
    // the following model width and depth multiples
    // are defined in the yolov5.yaml files in the yolo repo
    if (net[0] == 'n') {
      depth_multiple = 0.33;
      width_multiple = 0.25;
    } else if (net[0] == 's') {
      depth_multiple = 0.33;
      width_multiple = 0.50;
    } else if (net[0] == 'm') {
      depth_multiple = 0.67;
      width_multiple = 0.75;
    } else if (net[0] == 'l') {
      depth_multiple = 1.0;
      width_multiple = 1.0;
    } else if (net[0] == 'x') {
      depth_multiple = 1.33;
      width_multiple = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      depth_multiple = atof(argv[5]);
      width_multiple = atof(argv[6]);
    } else {
      return false;
    }
  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine_file = std::string(argv[2]);
    image_directory = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

void PrepareMemoryBuffers(ICudaEngine* engine, float** gpu_input_buffer,
                          float** gpu_output_buffer,
                          float** cpu_output_buffer) {
  CHECK(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int kInputIndex = engine->getBindingIndex(kInputTensorName);
  const int kOutputIndex = engine->getBindingIndex(kOutputTensorName);
  CHECK_EQ(kInputIndex, 0);
  CHECK_EQ(kOutputIndex, 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer,
                        kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer,
                        kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void RunInference(IExecutionContext& context, cudaStream_t& stream,
                  void** gpu_buffers, float* output, int batch_size) {
  // Sets execution context for TensorRT
  context.enqueue(batch_size, gpu_buffers, stream, nullptr);
  // async memory copy between host and GPU device
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1],
                             batch_size * kOutputSize * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void SerializeEngine(unsigned int max_batch_size, float& depth_multiple,
                     float& width_multiple, std::string& wts_file,
                     std::string& engine_file) {
  // Create builder
  IBuilder* builder = createInferBuilder(tensorrt_logger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine* engine = nullptr;
  // configure tensorRT engine file
  engine =
      BuildDetectionEngine(max_batch_size, builder, config, DataType::kFLOAT,
                           depth_multiple, width_multiple, wts_file);
  CHECK_NOTNULL(engine);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  CHECK_NOTNULL(serialized_engine);

  // Save engine to file
  std::ofstream file(engine_file, std::ios::binary);
  if (!file) {
    std::cerr << "Could not open plan output file" << std::endl;
    CHECK(false);
  }
  file.write(reinterpret_cast<const char*>(serialized_engine->data()),
             serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void DeserializeEngine(std::string& engine_file, IRuntime** runtime,
                       ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_file << " error!" << std::endl;
    CHECK(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  CHECK_NOTNULL(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(tensorrt_logger);
  CHECK_NOTNULL(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  CHECK_NOTNULL(*engine);
  *context = (*engine)->createExecutionContext();
  CHECK_NOTNULL(*context);
  delete[] serialized_engine;
}

int ReadDirFiles(const char* directory_name,
                 std::vector<std::string>& file_names) {
  DIR* directory = opendir(directory_name);
  if (directory == nullptr) {
    return -1;
  }

  struct dirent* file = nullptr;
  while ((file = readdir(directory)) != nullptr) {
    if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0) {
      std::string cur_file_name(file->d_name);
      file_names.push_back(cur_file_name);
    }
  }

  closedir(directory);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_file = "";
  std::string engine_file = "";
  float depth_multiple = 0.0f, width_multiple = 0.0f;
  std::string image_directory;

  if (!ParseArgs(argc, argv, wts_file, engine_file, depth_multiple,
                 width_multiple, image_directory)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -s [.wts_file] [.engine_file] [n/s/m/l/x "
                 "or c depth_multiple width_multiple]  // serialize model to "
                 "plan file"
              << std::endl;
    std::cerr
        << "./yolov5_det -d [.engine_file] ../images  // deserialize plan "
           "file and run inference"
        << std::endl;
    return -1;
  }

  // Create a model using the API directly and serialize it to a file
  if (!wts_file.empty()) {
    SerializeEngine(kBatchSize, depth_multiple, width_multiple, wts_file,
                    engine_file);
    return 0;
  }

  // Deserialize the engine_file from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  DeserializeEngine(engine_file, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init cuda preprocessing
  CudaPreprocessInit(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_output_buffer = nullptr;
  PrepareMemoryBuffers(engine, &gpu_buffers[0], &gpu_buffers[1],
                       &cpu_output_buffer);

  // Read images from directory
  // This should read one frame at a time for deployment
  std::vector<std::string> file_names;
  if (ReadDirFiles(image_directory.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    std::vector<cv::Mat> image_batch;
    std::vector<std::string> image_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat image = cv::imread(image_directory + "/" + file_names[j]);
      image_batch.push_back(image);
      image_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    CudaPreprocessBatch(image_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    RunInference(*context, stream, (void**)gpu_buffers, cpu_output_buffer,
                 kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;

    // Run Non Maximum Suppresion
    std::vector<std::vector<Detection>> result_batch;
    ApplyBatchNonMaxSuppression(result_batch, cpu_output_buffer,
                                image_batch.size(), kOutputSize, kConfThresh,
                                kNmsThresh);

    // Draw bounding boxes
    DrawBox(image_batch, result_batch);

    // Save images
    // Delete this for deployment
    for (size_t j = 0; j < image_batch.size(); j++) {
      cv::imwrite("_" + image_name_batch[j], image_batch[j]);
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  CudaPreprocessDestroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
