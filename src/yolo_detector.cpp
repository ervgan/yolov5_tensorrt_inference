#include "../include/yolo_detector.h"

#include <dirent.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../include/cuda_utils.h"
#include "../include/logging/logging.h"
#include "../include/model.h"
#include "../include/post_process.h"
#include "../include/pre_process.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Compiled representation of a neural network
// contains topology, layer config, device memory alloc and inference methods
using nvinfer1::ICudaEngine;
// Context for performance inference on a ICudaEngine
// contains methods for setting and retrieving CUDA stream
using nvinfer1::IExecutionContext;
// // Factory object ot create a ICudaEngine object from serialized .engine
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

Logger tensorrt_logger;
const int kOutputSize =
    kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

namespace {

bool ParseArgs(int argc, char **argv, std::string *wts_file,
               std::string *engine_file, float *depth_multiple,
               float *width_multiple, std::string *video_directory) {
  if (argc < 4) {
    return false;
  }

  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    *wts_file = std::string(argv[2]);
    *engine_file = std::string(argv[3]);
    auto net = std::string(argv[4]);

    // the following model width and depth multiples
    // are defined in the yolov5.yaml files in the yolo repo
    if (net[0] == 'n') {
      *depth_multiple = 0.33;
      *width_multiple = 0.25;

    } else if (net[0] == 's') {
      *depth_multiple = 0.33;
      *width_multiple = 0.50;

    } else if (net[0] == 'm') {
      *depth_multiple = 0.67;
      *width_multiple = 0.75;

    } else if (net[0] == 'l') {
      *depth_multiple = 1.0;
      *width_multiple = 1.0;

    } else if (net[0] == 'x') {
      *depth_multiple = 1.33;
      *width_multiple = 1.25;

    } else if (net[0] == 'c' && argc == 7) {
      *depth_multiple = atof(argv[5]);
      *width_multiple = atof(argv[6]);

    } else {
      return false;
    }

  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    *engine_file = std::string(argv[2]);
    *video_directory = std::string(argv[3]);

  } else {
    return false;
  }

  return true;
}

}  //  namespace

YoloDetector::YoloDetector() {}

YoloDetector::~YoloDetector() {
  cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(gpu_buffers_[0]));
  CUDA_CHECK(cudaFree(gpu_buffers_[1]));
  delete[] cpu_output_buffer_;
  CudaPreprocessDestroy();
  // Destroy the engine
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

void YoloDetector::PrepareMemoryBuffers(ICudaEngine *engine,
                                        float **gpu_input_buffer,
                                        float **gpu_output_buffer,
                                        float **cpu_output_buffer) {
  CHECK_EQ(engine->getNbBindings(), 2);
  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int kInputIndex = engine->getBindingIndex(kInputTensorName);
  const int kOutputIndex = engine->getBindingIndex(kOutputTensorName);
  CHECK_EQ(kInputIndex, 0);
  CHECK_EQ(kOutputIndex, 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(gpu_input_buffer),
                        kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(gpu_output_buffer),
                        kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void YoloDetector::RunInference(IExecutionContext *context,
                                const cudaStream_t &stream, void **gpu_buffers,
                                float *output, int batch_size) {
  // Sets execution context for TensorRT
  context->enqueue(batch_size, gpu_buffers, stream, nullptr);
  // async memory copy between host and GPU device
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1],
                             batch_size * kOutputSize * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  // makes sure all memory copies have been completed before returning
  cudaStreamSynchronize(stream);
}

// Serializes .wts file into .engine file
void YoloDetector::SerializeEngine(unsigned int max_batch_size,
                                   const float &depth_multiple,
                                   const float &width_multiple,
                                   const std::string &wts_file,
                                   const std::string &engine_file) {
  // Create builder
  IBuilder *builder = createInferBuilder(tensorrt_logger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine *engine = nullptr;
  // configure tensorRT engine file
  engine =
      BuildDetectionEngine(max_batch_size, builder, config, DataType::kFLOAT,
                           depth_multiple, width_multiple, wts_file);
  CHECK_NOTNULL(engine);

  // Serialize the engine
  IHostMemory *serialized_engine = engine->serialize();
  CHECK_NOTNULL(serialized_engine);

  // Save engine to file
  std::ofstream file(engine_file, std::ios::binary);

  if (!file) {
    std::cerr << "Could not open plan output file" << std::endl;
    CHECK(false);
  }

  file.write(reinterpret_cast<const char *>(serialized_engine->data()),
             serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void YoloDetector::DeserializeEngine(const std::string &engine_file,
                                     IRuntime **runtime, ICudaEngine **engine,
                                     IExecutionContext **context) {
  std::ifstream file(engine_file, std::ios::binary);

  if (!file.good()) {
    std::cerr << "read " << engine_file << " error!" << std::endl;
    CHECK(false);
  }

  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char *serialized_engine = new char[size];
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

int YoloDetector::Init(int argc, char **argv) {
  // sets parameters
  if (!ParseArgs(argc, argv, &wts_file_, &engine_file_, &depth_multiple_,
                 &width_multiple_, &video_directory_)) {
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
  // Converts .wts file to .engine file
  if (!wts_file_.empty()) {
    SerializeEngine(kBatchSize, depth_multiple_, width_multiple_, wts_file_,
                    engine_file_);
    return 0;
  }

  // Deserialize the engine_file from file to enable detection
  DeserializeEngine(engine_file_, &runtime_, &engine_, &context_);
  CUDA_CHECK(cudaStreamCreate(&stream_));
  // Init cuda preprocessing
  CudaPreprocessInit(kMaxInputImageSize);
  // Prepare cpu and gpu buffers
  PrepareMemoryBuffers(engine_, &gpu_buffers_[0], &gpu_buffers_[1],
                       &cpu_output_buffer_);
  return 1;
}

// This method should be modified to receive incoming frames from live video
void YoloDetector::DrawDetections() {
  cv::VideoCapture cap(video_directory_, cv::CAP_ANY);
  if (!cap.isOpened()) {
    std::cout << "!!! Failed to open file: " << video_directory_ << std::endl;
    return;
  }

  cv::Mat frame;
  cv::Mat resized_frame;
  int count = 0;
  for (;;) {
    if (!cap.read(frame)) break;

    cv::resize(frame, resized_frame, cv::Size(1200, 720));
    // Preprocess
    CudaPreprocess(resized_frame.data, resized_frame.cols, resized_frame.rows,
                   &gpu_buffers_[0][0], kInputW, kInputH, stream_);
    // Run inference
    auto start = std::chrono::system_clock::now();
    RunInference(context_, stream_, reinterpret_cast<void **>(gpu_buffers_),
                 cpu_output_buffer_, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;

    // Run Non Maximum Suppresion
    std::vector<Detection> result_batch;
    ApplyNonMaxSuppresion(&result_batch, &cpu_output_buffer_, kConfThresh,
                          kNmsThresh);

    // Draw bounding boxes
    DrawBox(resized_frame, &result_batch);

    // cv::imshow("window", resized_frame);

    // char key = cv::waitKey(10);
    // if (key == 27)  // ESC
    // break;
    cv::imwrite("_image" + std::to_string(count) + ".jpg", resized_frame);
    count++;
  }
}
