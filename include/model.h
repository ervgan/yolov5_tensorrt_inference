#pragma once

#include <NvInfer.h>

#include <string>

nvinfer1::ICudaEngine *BuildDetectionEngine(unsigned int maxBatchSize,
                                            nvinfer1::IBuilder *builder,
                                            nvinfer1::IBuilderConfig *config,
                                            nvinfer1::DataType dt,
                                            const float &depth_multiplier,
                                            const float &width_multiplier,
                                            const std::string &wts_name);
