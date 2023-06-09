#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "../include/macros.h"
#include "../include/types.h"

namespace nvinfer1 {
class YoloLayerPlugin : public IPluginV2IOExt {
 public:
  YoloLayerPlugin(int class_count, int neural_net_width, int neural_net_height,
                  int max_output, bool is_segmentation,
                  const std::vector<YoloKernel> &yolo_kernel);
  YoloLayerPlugin(const void *data, size_t length);
  ~YoloLayerPlugin();

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs,
                           int nb_input_dimensions) TRT_NOEXCEPT override;

  int initialize() TRT_NOEXCEPT override;

  void terminate() TRT_NOEXCEPT override{};

  size_t getWorkspaceSize(int max_batch_size) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(int batch_size, const void *const *inputs,
              void *TRT_CONST_ENQUEUE *outputs, void *workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void *buffer) const TRT_NOEXCEPT override;

  bool supportsFormatCombination(int index,
                                 const PluginTensorDesc *input_output_metadata,
                                 int nb_inputs,
                                 int nb_outputs) const TRT_NOEXCEPT override {
    return input_output_metadata[index].format == TensorFormat::kLINEAR &&
           input_output_metadata[index].type == DataType::kFLOAT;
  }

  const char *getPluginType() const TRT_NOEXCEPT override;

  const char *getPluginVersion() const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override;

  IPluginV2IOExt *clone() const TRT_NOEXCEPT override;

  void setPluginNamespace(const char *plugin_namespace) TRT_NOEXCEPT override;

  const char *getPluginNamespace() const TRT_NOEXCEPT override;

  DataType getOutputDataType(int index, const nvinfer1::DataType *input_types,
                             int nb_inputs) const TRT_NOEXCEPT override;

  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool *is_input_broadcasted,
                                    int nb_inputs) const TRT_NOEXCEPT override;

  bool canBroadcastInputAcrossBatch(int input_index) const
      TRT_NOEXCEPT override;

  // cuda_dnn refers to cuda deep learning library
  // cuda_blas refers to cuda's basic linear algebra library
  void attachToContext(cudnnContext *cuda_dnn_context,
                       cublasContext *cuda_blas_context,
                       IGpuAllocator *gpu_allocator) TRT_NOEXCEPT override;

  void configurePlugin(const PluginTensorDesc *input, int nb_input,
                       const PluginTensorDesc *output,
                       int nb_output) TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

 private:
  void ForwardGpu(const float *const *inputs, float *output,
                  cudaStream_t stream, int batch_size);
  int thread_count_ = 256;
  const char *plugin_namespace_;
  int kernel_count_;
  int class_count_;
  int yolov5_net_width_;
  int yolov5_net_height_;
  int max_output_object_;
  bool is_segmentation_;
  std::vector<YoloKernel> yolo_kernel_;
  void **anchor_;
};

class YoloPluginCreator : public IPluginCreator {
 public:
  YoloPluginCreator();

  ~YoloPluginCreator() override = default;

  const char *getPluginName() const TRT_NOEXCEPT override;

  const char *getPluginVersion() const TRT_NOEXCEPT override;

  const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override;

  IPluginV2IOExt *createPlugin(const char *name,
                               const PluginFieldCollection *plugin_fields)
      TRT_NOEXCEPT override;

  IPluginV2IOExt *deserializePlugin(const char *name, const void *serial_data,
                                    size_t serial_length) TRT_NOEXCEPT override;

  void setPluginNamespace(const char *lib_namespace) TRT_NOEXCEPT override {
    namespace_ = lib_namespace;
  }

  const char *getPluginNamespace() const TRT_NOEXCEPT override {
    return namespace_.c_str();
  }

 private:
  std::string namespace_;
  static PluginFieldCollection plugin_fields_;
  static std::vector<PluginField> plugin_attributes_;
};
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};  // namespace nvinfer1
