#include <dirent.h>

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

static Logger gLogger;
const static int kOutputSize =
    kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool parse_args(int argc, char** argv, std::string& wts_file_name,
                std::string& engine_file, float& depth_multiple,
                float& width_multiple, std::string& image_directory) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts_file_name = std::string(argv[2]);
    engine_file = std::string(argv[3]);
    auto net = std::string(argv[4]);
    // the following model's width and depth multiples
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

void prepare_buffers(ICudaEngine* engine_file, float** gpu_input_buffer,
                     float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine_file->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int kInputIndex = engine_file->getBindingIndex(kInputTensorName);
  const int kOutputIndex = engine_file->getBindingIndex(kOutputTensorName);
  assert(kInputIndex == 0);
  assert(kOutputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer,
                        kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer,
                        kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers,
           float* output, int batch_size) {
  // Sets execution context for TensorRT
  context.enqueue(batch_size, gpu_buffers, stream, nullptr);
  // async memory copy between host and GPU device
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1],
                             batch_size * kOutputSize * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& depth_multiple,
                      float& width_multiple, std::string& wts_name,
                      std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine_file
  ICudaEngine* engine_file = nullptr;
  engine_file =
      BuildDetectionEngine(max_batchsize, builder, config, DataType::kFLOAT,
                           depth_multiple, width_multiple, wts_name);
  assert(engine_file != nullptr);

  // Serialize the engine_file
  IHostMemory* serialized_engine = engine_file->serialize();
  assert(serialized_engine != nullptr);

  // Save engine_file to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()),
          serialized_engine->size());

  // Close everything down
  engine_file->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime,
                        ICudaEngine** engine_file,
                        IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine_file = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine_file);
  *context = (*engine_file)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

int read_files_in_dir(const char* directory_name,
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

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  float depth_multiple = 0.0f, width_multiple = 0.0f;
  std::string image_directory;

  if (!parse_args(argc, argv, wts_name, engine_name, depth_multiple,
                  width_multiple, image_directory)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -s [.wts_file_name] [.engine_file] [n/s/m/l/x "
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
  if (!wts_name.empty()) {
    serialize_engine(kBatchSize, depth_multiple, width_multiple, wts_name,
                     engine_name);
    return 0;
  }

  // Deserialize the engine_file from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine_file = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine_file, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  CudaPreprocessInit(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_output_buffer = nullptr;
  prepare_buffers(engine_file, &gpu_buffers[0], &gpu_buffers[1],
                  &cpu_output_buffer);

  // Read images from directory
  std::vector<std::string> file_names;
  if (read_files_in_dir(image_directory.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat img = cv::imread(image_directory + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    CudaPreprocessBatch(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    ApplyBatchNonMaxSuppression(res_batch, cpu_output_buffer, img_batch.size(),
                                kOutputSize, kConfThresh, kNmsThresh);

    // Draw bounding boxes
    DrawBox(img_batch, res_batch);

    // Save images
    for (size_t j = 0; j < img_batch.size(); j++) {
      cv::imwrite("_" + img_name_batch[j], img_batch[j]);
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  CudaPreprocessDestroy();
  // Destroy the engine_file
  context->destroy();
  engine_file->destroy();
  runtime->destroy();

  // Print histogram of the output distribution
  // std::cout << "\nOutput:\n\n";
  // for (unsigned int i = 0; i < kOutputSize; i++) {
  //   std::cout << prob[i] << ", ";
  //   if (i % 10 == 0) std::cout << std::endl;
  // }
  // std::cout << std::endl;

  return 0;
}
