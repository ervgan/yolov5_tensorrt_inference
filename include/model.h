#pragma once

#include <NvInfer.h>

#include <string>

using nvinfer1::DataType;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;

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
using nvinfer1::ScaleMode;
using nvinfer1::Weights;

ICudaEngine* BuildDetectionEngine(unsigned int maxBatchSize, IBuilder* builder,
                                  IBuilderConfig* config, DataType dt,
                                  const float& depth_multiplier,
                                  const float& width_multiplier,
                                  const std::string& wts_name);
