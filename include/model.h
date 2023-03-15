#ifndef YOLOV5_INFERENCE_MODEL_H_
#define YOLOV5_INFERENCE_MODEL_H_

#include <NvInfer.h>

#include <string>

using nvinfer1::ActivationType;
using nvinfer1::BuilderFlag;
using nvinfer1::DataType;
using nvinfer1::Dims3;
using nvinfer1::DimsHW;
using nvinfer1::ElementWiseOperation;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::IConvolutionLayer;
using nvinfer1::ICudaEngine;
using nvinfer1::IPluginV2;
using nvinfer1::IPluginV2Layer;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PoolingType;
using nvinfer1::ResizeMode;
// base interface for all layers in TensorRT
using nvinfer1::ILayer;
// inherits from ILayer
using nvinfer1::IScaleLayer;
// Neural network definition: specifies layers and connections between them
using nvinfer1::INetworkDefinition;
using nvinfer1::ITensor;
// specifies how scale values are applied to input tensor of a layer
using nvinfer1::DataType;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::ScaleMode;
using nvinfer1::Weights;

// C++ implementation of Yolov5 modules in models/common.py
// https://github.com/ultralytics/yolov5/blob/d02ee60512c50d9573bb7a136d8baade8a0bd332/models/common.py#L159

namespace yolov5_inference {
// Divisor to make the width of the output channel divisible by 32
// because Yolov5's use of feature pyramid network requires the width to be
// multiples of 32
constexpr int kDivisor = 8;
// Builds the TensorRT engine representing the Yolov5 neural network
ICudaEngine* BuildDetectionEngine(unsigned int maxBatchSize, IBuilder* builder,
                                  IBuilderConfig* config, DataType dt,
                                  const float& depth_multiplier,
                                  const float& width_multiplier,
                                  const std::string& wts_name);
}  // namespace yolov5_inference

#endif  // YOLOV5_INFERENCE_MODEL_H_
