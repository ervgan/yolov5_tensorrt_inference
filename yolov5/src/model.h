#pragma once

#include <NvInfer.h>

#include <string>

nvinfer1::ICudaEngine* BuildDetectionEngine(unsigned int maxBatchSize,
                                            nvinfer1::IBuilder* builder,
                                            nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, float& gd,
                                            float& gw,
                                            const std::string& wts_name);
