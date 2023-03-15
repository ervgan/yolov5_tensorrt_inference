#include <glog/logging.h>

#include <iostream>
#include <vector>

#include "../include/cuda_utils.h"
#include "../include/yolo_layer.h"

// Implements a TensorRT custom plugin
// to process the last tensor output according to the yolo specifications
// and produces final detection results with confidence scores and bounding
// boxes
// Source code for NvInfer.h:
// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/_nv_infer_8h_source.html
// Source code for NvInferRuntimeCommon.h:
// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/_nv_infer_runtime_common_8h_source.html

using yolov5_inference::kIgnoreThresh;

namespace read_write {
template <typename T>
void write(char** buffer, const T& value) {
  **reinterpret_cast<T**>(buffer) = value;
  *buffer += sizeof(T);
}

template <typename T>
void read(const char** buffer, T* value) {
  *value = **reinterpret_cast<const T**>(buffer);
  *buffer += sizeof(T);
}
}  // namespace read_write

namespace nvinfer1 {
YoloLayerPlugin::YoloLayerPlugin(int class_count, int neural_net_width,
                                 int neural_net_height, int max_output,
                                 bool is_segmentation,
                                 const std::vector<YoloKernel>& yolo_kernel) {
  class_count_ = class_count;
  yolov5_net_height_ = neural_net_width;
  yolov5_net_width_ = neural_net_height;
  max_output_object_ = max_output;
  is_segmentation_ = is_segmentation;
  yolo_kernel_ = yolo_kernel;
  kernel_count_ = yolo_kernel.size();

  CUDA_CHECK(cudaMallocHost(&anchor_, kernel_count_ * sizeof(void*)));
  size_t anchor_len = sizeof(float) * kNumAnchor * 2;
  for (int i = 0; i < kernel_count_; i++) {
    CUDA_CHECK(cudaMalloc(&anchor_[i], anchor_len));
    const auto& yolo = yolo_kernel_[i];
    CUDA_CHECK(cudaMemcpy(anchor_[i], yolo.anchors, anchor_len,
                          cudaMemcpyHostToDevice));
  }
}

YoloLayerPlugin::~YoloLayerPlugin() {
  for (int i = 0; i < kernel_count_; i++) {
    CUDA_CHECK(cudaFree(anchor_[i]));
  }
  CUDA_CHECK(cudaFreeHost(anchor_));
}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
  using read_write::read;
  const char *buffer = reinterpret_cast<const char*>(data),
             *buffer_start_pointer = buffer;
  read(&buffer, &class_count_);
  read(&buffer, &thread_count_);
  read(&buffer, &kernel_count_);
  read(&buffer, &yolov5_net_width_);
  read(&buffer, &yolov5_net_height_);
  read(&buffer, &max_output_object_);
  read(&buffer, &is_segmentation_);
  yolo_kernel_.resize(kernel_count_);
  auto kernel_size = kernel_count_ * sizeof(YoloKernel);
  memcpy(yolo_kernel_.data(), buffer, kernel_size);
  buffer += kernel_size;
  CUDA_CHECK(cudaMallocHost(&anchor_, kernel_count_ * sizeof(void*)));
  size_t anchor_len = sizeof(float) * kNumAnchor * 2;
  for (int i = 0; i < kernel_count_; i++) {
    CUDA_CHECK(cudaMalloc(&anchor_[i], anchor_len));
    const auto& yolo = yolo_kernel_[i];
    CUDA_CHECK(cudaMemcpy(anchor_[i], yolo.anchors, anchor_len,
                          cudaMemcpyHostToDevice));
  }
  CHECK_EQ(buffer, buffer_start_pointer + length);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  using read_write::write;
  char *buffer_1 = static_cast<char*>(buffer),
       *buffer_1_start_pointer = buffer_1;
  write(&buffer_1, class_count_);
  write(&buffer_1, thread_count_);
  write(&buffer_1, kernel_count_);
  write(&buffer_1, yolov5_net_width_);
  write(&buffer_1, yolov5_net_height_);
  write(&buffer_1, max_output_object_);
  write(&buffer_1, is_segmentation_);
  auto kernel_size = kernel_count_ * sizeof(YoloKernel);
  memcpy(buffer_1, yolo_kernel_.data(), kernel_size);
  buffer_1 += kernel_size;

  CHECK_EQ(buffer_1, buffer_1_start_pointer + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t size =
      sizeof(class_count_) + sizeof(thread_count_) + sizeof(kernel_count_);
  size += sizeof(YoloKernel) * yolo_kernel_.size();
  size += sizeof(yolov5_net_width_) + sizeof(yolov5_net_height_);
  size += sizeof(max_output_object_) + sizeof(is_segmentation_);
  return size;
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT { return 0; }

Dims YoloLayerPlugin::getOutputDimensions(
    int index, const Dims* inputs, int nb_input_dimensions) TRT_NOEXCEPT {
  // output the result to channel
  int total_size = max_output_object_ * sizeof(Detection) / sizeof(float);
  return Dims3(total_size + 1, 1, 1);
}

void YoloLayerPlugin::setPluginNamespace(const char* plugin_namespace)
    TRT_NOEXCEPT {
  plugin_namespace_ = plugin_namespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return plugin_namespace_;
}

// Return the DataType of the plugin output at the requested index
DataType YoloLayerPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch
bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* input_is_broadcasted,
    int nb_inputs) const TRT_NOEXCEPT {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without
// replication
bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int input_index) const
    TRT_NOEXCEPT {
  return false;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* input,
                                      int nb_input,
                                      const PluginTensorDesc* output,
                                      int nb_output) TRT_NOEXCEPT {}

// Attach the plugin object to an execution context and grant the plugin the
// access to some context resource
void YoloLayerPlugin::attachToContext(
    cudnnContext* cuda_dnn_context, cublasContext* cuda_blas_context,
    IGpuAllocator* gpu_allocator) TRT_NOEXCEPT {}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

void YoloLayerPlugin::destroy() TRT_NOEXCEPT { delete this; }

// Clone the plugin
IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {
  YoloLayerPlugin* plugin =
      new YoloLayerPlugin(class_count_, yolov5_net_width_, yolov5_net_height_,
                          max_output_object_, is_segmentation_, yolo_kernel_);
  plugin->setPluginNamespace(plugin_namespace_);
  return plugin;
}

// cuda specific function that can be called from a __global__
// function and executed on the GPU
__device__ float LogisticFunction(float data) {
  return 1.0f / (1.0f + expf(-data));
}

// cuda specific function executed on the GPU and can be called from
// the host CPU, runs across multiple threads
__global__ void CallDetection(const float* input, float* output,
                              int nb_elements, const int neural_net_width,
                              const int neural_net_height,
                              int max_output_object, int yolo_width,
                              int yolo_height,
                              const float anchors[kNumAnchor * 2], int classes,
                              int output_element, bool is_segmentation) {
  // cuda specific parameters to get a unique thread id
  // threadIdx.x represents the thread id within a block
  // blockDim.x represents the number of threads in a block
  // blockIdx.x represents the block id within a grid
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (thread_id >= nb_elements) return;

  // size of yolo kernel -> set of predefined anchor boxes used to detect
  // objects
  int total_grid = yolo_width * yolo_height;
  // this is used if we process multiple images in parallel
  // the default case if kBatchSize = 1 set in config.h
  // so thread_id should remain the same in the case
  int batch_normalization_thread_id = thread_id / total_grid;
  thread_id = thread_id - total_grid * batch_normalization_thread_id;
  // 5 corresponds to the dimensions of our Detection being center_x, center_y,
  // w, h in pixels and confidence level as a float
  int info_len_i = 5 + classes;
  // in case of object segmentation, 32 represents the size of the segmentation
  // mask prototypes that will be used -> feature vector will be multiplied with
  // mask prototypes to produce a segmentation that aligns with the object
  if (is_segmentation) info_len_i += 32;
  // get address of start of input tensor with batch_normalization_thread_id = 0
  const float* current_input =
      input +
      batch_normalization_thread_id * (info_len_i * total_grid * kNumAnchor);

  // loop through all anchor boxes, by default yolov5 uses 3 anchor boxes per
  // yolo kernel
  for (int k = 0; k < kNumAnchor; ++k) {
    // applies logistic function to get confidence score of the anchor box
    // stored at the index 4 in the input tensor
    float box_prob =
        LogisticFunction(current_input[thread_id + k * info_len_i * total_grid +
                                       4 * total_grid]);
    // if confidence score is less than 0.1, we ignore it
    if (box_prob < kIgnoreThresh) continue;
    // get probability of each object class for this anchor box
    // for LPS, we only have one object class
    // this info is stored starting at index 5 of the input tensor
    int class_id = 0;
    float max_cls_prob = 0.0;
    for (int i = 5; i < 5 + classes; ++i) {
      float prob = LogisticFunction(
          current_input[thread_id + k * info_len_i * total_grid +
                        i * total_grid]);
      if (prob > max_cls_prob) {
        max_cls_prob = prob;
        class_id = i - 5;
      }
    }
    float* result_count =
        output + batch_normalization_thread_id * output_element;
    // atomic add is a cuda specific method allowing for value addition
    // in a thread safe manner without race conditions
    // it increments result_count by one for each detection
    int count = static_cast<int>(atomicAdd(result_count, 1));
    if (count >= max_output_object) return;
    // get pointer to the output buffer where current detection should be
    // written
    char* data = reinterpret_cast<char*>(result_count) + sizeof(float) +
                 count * sizeof(Detection);
    Detection* detection = reinterpret_cast<Detection*>(data);

    int row = thread_id / yolo_width;
    int col = thread_id % yolo_width;
    // the following code applies logistic regression to the input tensors
    // at the correct index to get the bounding box coordinates of the detection

    // grid cell are normalized to dimensions of 1 and
    // network predicts object's center from top-left corner of the grid cell
    // 0.5f then represents a shifting parameter to compute x and y center
    // coordinates relative to the center of grid cell and not top-left corner

    // we also need to scale coordinates in terms of pixels of original image
    // with neural_net_width / yolo_width and neural_net_height / yolo_height

    // finally, logistic regression outputs a probability in range [0, 1]
    // in Yolov5 implementation, the logistic regression is multiplied by 2.0f
    // to return a value in the range of [0, 2.0] which represents two
    // adjacent grid cells. The choice of 2.0 is specific to Yolov5
    // it ensures that the resulting values are still in reasonable range

    // find center_x of bounding_box detection at correct index of input tensor
    detection->bounding_box_px[0] =
        (col - 0.5f +
         2.0f * LogisticFunction(
                    current_input[thread_id + k * info_len_i * total_grid +
                                  0 * total_grid])) *
        neural_net_width / yolo_width;

    // find center_y of bounding_box detection at correct index of input tensor
    detection->bounding_box_px[1] =
        (row - 0.5f +
         2.0f * LogisticFunction(
                    current_input[thread_id + k * info_len_i * total_grid +
                                  1 * total_grid])) *
        neural_net_height / yolo_height;

    // find width of bounding_box detection at correct index of input tensor
    detection->bounding_box_px[2] =
        2.0f *
        LogisticFunction(current_input[thread_id + k * info_len_i * total_grid +
                                       2 * total_grid]);
    // bounding box width and height are relative to the anchor box
    // so this is needed to scale to width and height ot original image
    detection->bounding_box_px[2] = detection->bounding_box_px[2] *
                                    detection->bounding_box_px[2] *
                                    anchors[2 * k];

    // find height of bounding_box detection at correct index of input tensor
    detection->bounding_box_px[3] =
        2.0f *
        LogisticFunction(current_input[thread_id + k * info_len_i * total_grid +
                                       3 * total_grid]);
    // bounding box width and height are relative to the anchor box
    // so this is needed to scale to width and height ot original image
    detection->bounding_box_px[3] = detection->bounding_box_px[3] *
                                    detection->bounding_box_px[3] *
                                    anchors[2 * k + 1];
    detection->confidence = box_prob * max_cls_prob;
    detection->class_id = class_id;

    for (int i = 0; is_segmentation && i < 32; i++) {
      detection->mask[i] =
          current_input[thread_id + k * info_len_i * total_grid +
                        (i + 5 + classes) * total_grid];
    }
  }
}

// Performs forward propagation of Yolo layer and launches
// a CUDA kernel for object detection
void YoloLayerPlugin::ForwardGpu(const float* const* inputs, float* output,
                                 cudaStream_t stream, int batch_size) {
  int output_element =
      1 + max_output_object_ * sizeof(Detection) / sizeof(float);
  for (int i = 0; i < batch_size; ++i) {
    CUDA_CHECK(
        cudaMemsetAsync(output + i * output_element, 0, sizeof(float), stream));
  }
  int num_element = 0;
  for (unsigned int i = 0; i < yolo_kernel_.size(); ++i) {
    const auto& yolo = yolo_kernel_[i];
    num_element = yolo.width * yolo.height * batch_size;
    if (num_element < thread_count_) thread_count_ = num_element;

    CallDetection<<<(num_element + thread_count_ - 1) / thread_count_,
                    thread_count_, 0, stream>>>(
        inputs[i], output, num_element, yolov5_net_width_, yolov5_net_height_,
        max_output_object_, yolo.width, yolo.height,
        reinterpret_cast<float*>(anchor_[i]), class_count_, output_element,
        is_segmentation_);
  }
}

int YoloLayerPlugin::enqueue(int batch_size, const void* const* inputs,
                             void* TRT_CONST_ENQUEUE* outputs, void* workspace,
                             cudaStream_t stream) TRT_NOEXCEPT {
  ForwardGpu((const float* const*)inputs, reinterpret_cast<float*>(outputs[0]),
             stream, batch_size);
  return 0;
}

PluginFieldCollection YoloPluginCreator::plugin_fields_{};
std::vector<PluginField> YoloPluginCreator::plugin_attributes_;

YoloPluginCreator::YoloPluginCreator() {
  plugin_attributes_.clear();
  plugin_fields_.nbFields = plugin_attributes_.size();
  plugin_fields_.fields = plugin_attributes_.data();
}

const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT {
  return &plugin_fields_;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* plugin_fields) TRT_NOEXCEPT {
  CHECK_EQ(plugin_fields->nbFields, 2);
  CHECK_EQ(strcmp(plugin_fields->fields[0].name, "netinfo"), 0);
  CHECK_EQ(strcmp(plugin_fields->fields[1].name, "kernels"), 0);
  const int* neural_net_info =
      reinterpret_cast<const int*>(plugin_fields->fields[0].data);
  int class_count = neural_net_info[0];
  int input_width = neural_net_info[1];
  int input_height = neural_net_info[2];
  int max_output_object_count = neural_net_info[3];
  bool is_segmentation = static_cast<bool>(neural_net_info[4]);
  std::vector<YoloKernel> kernels(plugin_fields->fields[1].length);
  memcpy(&kernels[0], plugin_fields->fields[1].data,
         kernels.size() * sizeof(YoloKernel));
  YoloLayerPlugin* plugin_object =
      new YoloLayerPlugin(class_count, input_width, input_height,
                          max_output_object_count, is_segmentation, kernels);
  plugin_object->setPluginNamespace(namespace_.c_str());
  return plugin_object;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(
    const char* name, const void* serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call YoloLayerPlugin::destroy()
  YoloLayerPlugin* plugin_object =
      new YoloLayerPlugin(serial_data, serial_length);
  plugin_object->setPluginNamespace(namespace_.c_str());
  return plugin_object;
}
}  // namespace nvinfer1
