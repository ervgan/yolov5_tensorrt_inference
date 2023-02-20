#pragma once

#include <NvInfer.h>

#include <string>

using nvinfer1::DataType;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;

ICudaEngine* BuildDetectionEngine(unsigned int maxBatchSize, IBuilder* builder,
                                  IBuilderConfig* config, DataType dt,
                                  const float& depth_multiplier,
                                  const float& width_multiplier,
                                  const std::string& wts_name);
