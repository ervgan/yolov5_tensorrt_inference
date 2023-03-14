#include "../include/model.h"

#include <glog/logging.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include "../include/config.h"
#include "../include/yolo_layer.h"

// C++ implementation of Yolov5 modules in models/common.py
// https://github.com/ultralytics/yolov5/blob/d02ee60512c50d9573bb7a136d8baade8a0bd332/models/common.py#L159
// Builds all the layers of the detection engine

namespace {

// Loads wts file and returns a map of names with correpsonding weights
// TensorRT wts weight files have a simple space delimited format :
// [type] [size] <data x size in hex>
std::map<std::string, Weights> LoadWeights(const std::string &file) {
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, Weights> weight_map;
  std::ifstream input(file);
  CHECK(input.is_open() &&
        "Unable to load weight file. please check if the .wts file path");
  // Read number of weight blobs
  int32_t count;
  input >> count;
  CHECK(count > 0 && "Invalid weight map file.");

  // get weights for each
  for (int32_t i = 0; i < count; i++) {
    Weights weight{DataType::kFLOAT, nullptr, 0};
    uint32_t size;
    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    CHECK(input.fail() && "Error reading name and size from wts file");
    weight.type = DataType::kFLOAT;
    // Load blob
    auto weights = std::make_unique<uint32_t[]>(size);

    for (uint32_t x = 0, y = size; x < y; ++x) {
      input >> std::hex >> weights[x];
    }

    weight.values = weights;
    weight.count = size;
    weight_map[name] = weight;
  }

  return weight_map;
}

// get nb of output channels
int get_width(int width, float width_multiplier) {
  const int divisor = 8;
  return static_cast<int>(ceil((width * width_multiplier) / divisor)) * divisor;
}

int get_width(int width, const float width_multiplier, int divisor) {
  return static_cast<int>(ceil((width * width_multiplier) / divisor)) * divisor;
}

// get number of layers in the network
int get_depth(int depth, const float depth_multiplier) {
  if (depth == 1) {
    return 1;
  }

  int scaled_depth = round(depth * depth_multiplier);

  if (depth * depth_multiplier - static_cast<int>(depth * depth_multiplier) ==
          0.5 &&
      (static_cast<int>(depth * depth_multiplier) % 2) == 0) {
    --scaled_depth;
  }

  return std::max<int>(scaled_depth, 1);
}

// add batch normalization to address internal covariate shift
// returns normalized layer
IScaleLayer *AddBatchNorm2D(INetworkDefinition *network,
                            std::map<std::string, Weights> *weight_map,
                            ITensor *input, const std::string &layer_name,
                            float eps) {
  const float *gamma = reinterpret_cast<const float *>(
      (*weight_map)[layer_name + ".weight"].values);
  const float *beta = reinterpret_cast<const float *>(
      (*weight_map)[layer_name + ".bias"].values);
  const float *mean = reinterpret_cast<const float *>(
      (*weight_map)[layer_name + ".running_mean"].values);
  const float *var = reinterpret_cast<const float *>(
      (*weight_map)[layer_name + ".running_var"].values);

  const int kLen = (*weight_map)[layer_name + ".running_var"].count;

  float *scale_values = reinterpret_cast<float *>(malloc(sizeof(float) * kLen));

  for (int i = 0; i < kLen; i++) {
    scale_values[i] = gamma[i] / sqrt(var[i] + eps);
  }

  Weights scale{DataType::kFLOAT, scale_values, kLen};

  float *shift_values = reinterpret_cast<float *>(malloc(sizeof(float) * kLen));

  for (int i = 0; i < kLen; i++) {
    shift_values[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
  }

  Weights shift{DataType::kFLOAT, shift_values, kLen};

  float *power_values = reinterpret_cast<float *>(malloc(sizeof(float) * kLen));

  for (int i = 0; i < kLen; i++) {
    power_values[i] = 1.0;
  }

  Weights power{DataType::kFLOAT, power_values, kLen};

  (*weight_map)[layer_name + ".scale"] = scale;
  (*weight_map)[layer_name + ".shift"] = shift;
  (*weight_map)[layer_name + ".power"] = power;

  IScaleLayer *scale_layer =
      network->addScale(*input, ScaleMode::kCHANNEL, shift, scale, power);
  CHECK_NOTNULL(scale_layer);
  return scale_layer;
}

// returns a single TensorRT layer
ILayer *CreateConvoLayer(INetworkDefinition *network,
                         std::map<std::string, Weights> *weight_map,
                         ITensor *input, int output, int kernel_size,
                         int stride, int nb_groups,
                         const std::string &layer_name) {
  Weights empty_wts{DataType::kFLOAT, nullptr, 0};
  const int kPadding = kernel_size / 3;
  IConvolutionLayer *convo_layer = network->addConvolutionNd(
      *input, output, DimsHW{kernel_size, kernel_size},
      (*weight_map)[layer_name + ".conv.weight"], empty_wts);

  CHECK_NOTNULL(convo_layer);

  convo_layer->setStrideNd(DimsHW{stride, stride});
  convo_layer->setPaddingNd(DimsHW{kPadding, kPadding});
  convo_layer->setNbGroups(nb_groups);
  convo_layer->setName((layer_name + ".conv").c_str());
  IScaleLayer *batch_norm_layer = AddBatchNorm2D(
      network, weight_map, convo_layer->getOutput(0), layer_name + ".bn", 1e-3);

  // using silu activation method = input * sigmoid
  auto sigmoid_activation = network->addActivation(
      *batch_norm_layer->getOutput(0), ActivationType::kSIGMOID);
  CHECK_NOTNULL(sigmoid_activation);
  auto silu_activation = network->addElementWise(
      *batch_norm_layer->getOutput(0), *sigmoid_activation->getOutput(0),
      ElementWiseOperation::kPROD);
  CHECK_NOTNULL(silu_activation);
  return silu_activation;
}

// Creates Bottleneck Layer
ILayer *CreateBottleneckLayer(INetworkDefinition *network,
                              std::map<std::string, Weights> *weight_map,
                              ITensor *input, int intput_channel,
                              int output_channel, bool shortcut, int nb_groups,
                              float expansion, const std::string &layer_name) {
  const int kOutputMaps1 =
      static_cast<int>(static_cast<float>(output_channel) * expansion);
  const int kKernelSize1 = 1;
  const int kStride1 = 1;
  const int kNbGroups1 = 1;
  const int kKernelSize2 = 3;
  const int kStride2 = 1;

  auto convo_layer_1 =
      CreateConvoLayer(network, weight_map, input, kOutputMaps1, kKernelSize1,
                       kStride1, kNbGroups1, layer_name + ".cv1");

  auto convo_layer_2 = CreateConvoLayer(
      network, weight_map, convo_layer_1->getOutput(0), output_channel,
      kKernelSize2, kStride2, nb_groups, layer_name + ".cv2");

  if (shortcut && intput_channel == output_channel) {
    auto element_wise_layer = network->addElementWise(
        *input, *convo_layer_2->getOutput(0), ElementWiseOperation::kSUM);
    return element_wise_layer;
  }

  return convo_layer_2;
}

// Simplified cross stage partial bottleneck using partial bottlenecks
// to extract multi-scale features
ILayer *CreateC3Bottleneck(INetworkDefinition *network,
                           std::map<std::string, Weights> *weight_map,
                           ITensor *input, int input_channel,
                           int output_channel, int n, bool shortcut,
                           int nb_groups, float expansion,
                           const std::string &layer_name) {
  int hidden_channel =
      static_cast<int>(static_cast<float>(output_channel) * expansion);

  auto convo_layer_1 = CreateConvoLayer(
      network, weight_map, input, hidden_channel, 1, 1, 1, layer_name + ".cv1");

  auto convo_layer_2 = CreateConvoLayer(
      network, weight_map, input, hidden_channel, 1, 1, 1, layer_name + ".cv2");

  ITensor *y1 = convo_layer_1->getOutput(0);

  for (int i = 0; i < n; i++) {
    auto b = CreateBottleneckLayer(network, weight_map, y1, hidden_channel,
                                   hidden_channel, shortcut, nb_groups, 1.0,
                                   layer_name + ".m." + std::to_string(i));
    y1 = b->getOutput(0);
  }

  ITensor *inputTensors[] = {y1, convo_layer_2->getOutput(0)};
  auto cat = network->addConcatenation(inputTensors, 2);

  auto convo_layer_3 =
      CreateConvoLayer(network, weight_map, cat->getOutput(0), output_channel,
                       1, 1, 1, layer_name + ".cv3");
  return convo_layer_3;
}

// Faster implementation of the spatial pyramid pooling layer
// with less FLOPs
ILayer *CreateSPPFLayer(INetworkDefinition *network,
                        std::map<std::string, Weights> *weight_map,
                        ITensor *input, int input_channel, int output_channel,
                        int dimensions, std::string layer_name) {
  int hidden_channels = input_channel / 2;

  auto convo_layer_1 =
      CreateConvoLayer(network, weight_map, input, hidden_channels, 1, 1, 1,
                       layer_name + ".cv1");
  // all three pooling layers used in SPP
  auto pooling_layer_1 =
      network->addPoolingNd(*convo_layer_1->getOutput(0), PoolingType::kMAX,
                            DimsHW{dimensions, dimensions});
  pooling_layer_1->setPaddingNd(DimsHW{dimensions / 2, dimensions / 2});
  pooling_layer_1->setStrideNd(DimsHW{1, 1});
  auto pooling_layer_2 =
      network->addPoolingNd(*pooling_layer_1->getOutput(0), PoolingType::kMAX,
                            DimsHW{dimensions, dimensions});
  pooling_layer_2->setPaddingNd(DimsHW{dimensions / 2, dimensions / 2});
  pooling_layer_2->setStrideNd(DimsHW{1, 1});
  auto pooling_layer_3 =
      network->addPoolingNd(*pooling_layer_2->getOutput(0), PoolingType::kMAX,
                            DimsHW{dimensions, dimensions});
  pooling_layer_3->setPaddingNd(DimsHW{dimensions / 2, dimensions / 2});
  pooling_layer_3->setStrideNd(DimsHW{1, 1});
  ITensor *inputTensors[] = {
      convo_layer_1->getOutput(0), pooling_layer_1->getOutput(0),
      pooling_layer_2->getOutput(0), pooling_layer_3->getOutput(0)};

  auto concatenation_layer = network->addConcatenation(inputTensors, 4);

  auto convo_layer_2 =
      CreateConvoLayer(network, weight_map, concatenation_layer->getOutput(0),
                       output_channel, 1, 1, 1, layer_name + ".cv2");
  return convo_layer_2;
}

std::vector<std::vector<float>> getAnchors(
    std::map<std::string, Weights> *weight_map, const std::string &layer_name) {
  std::vector<std::vector<float>> anchors;
  Weights weights = (*weight_map)[layer_name + ".anchor_grid"];
  const int anchor_len = kNumAnchor * 2;

  for (int i = 0; i < weights.count / anchor_len; i++) {
    auto *p = (const float *)weights.values + i * anchor_len;
    std::vector<float> anchor(p, p + anchor_len);
    anchors.push_back(anchor);
  }

  return anchors;
}

IPluginV2Layer *AddYoLoLayer(INetworkDefinition *network,
                             std::map<std::string, Weights> *weight_map,
                             std::string layer_name,
                             std::vector<IConvolutionLayer *> dets,
                             bool is_segmentation = false) {
  auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
  auto anchors = getAnchors(weight_map, layer_name);
  PluginField plugin_fields[2];
  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox,
                    static_cast<int>(is_segmentation)};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;

  // load strides from Detect layer
  CHECK((*weight_map).find(layer_name + ".strides") != (*weight_map).end() &&
        "Not found `strides`, please check wts_converter.py!!!");
  Weights strides = (*weight_map)[layer_name + ".strides"];
  auto *p = (const float *)(strides.values);
  std::vector<int> scales(p, p + strides.count);

  std::vector<YoloKernel> kernels;

  for (size_t i = 0; i < anchors.size(); i++) {
    YoloKernel kernel;
    kernel.width = kInputW / scales[i];
    kernel.height = kInputH / scales[i];
    memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
    kernels.push_back(kernel);
  }

  plugin_fields[1].data = &kernels[0];
  plugin_fields[1].length = kernels.size();
  plugin_fields[1].name = "kernels";
  plugin_fields[1].type = PluginFieldType::kFLOAT32;
  PluginFieldCollection plugin_data;
  plugin_data.nbFields = 2;
  plugin_data.fields = plugin_fields;
  IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
  std::vector<ITensor *> input_tensors;

  for (auto det : dets) {
    input_tensors.push_back(det->getOutput(0));
  }

  auto yolo_layer = network->addPluginV2(&input_tensors[0],
                                         input_tensors.size(), *plugin_obj);
  return yolo_layer;
}

}  // namespace

ICudaEngine *BuildDetectionEngine(unsigned int max_batch_size,
                                  IBuilder *builder, IBuilderConfig *config,
                                  DataType dt, const float &depth_multiplier,
                                  const float &width_multiplier,
                                  const std::string &wts_file_name) {
  INetworkDefinition *network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, kInputH, kInputW}
  ITensor *data =
      network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
  CHECK_NOTNULL(data);
  std::map<std::string, Weights> weight_map = LoadWeights(wts_file_name);

  // Backbone layers
  auto convo_layer_0 =
      CreateConvoLayer(network, &weight_map, data,
                       get_width(64, width_multiplier), 6, 2, 1, "model.0");
  CHECK_NOTNULL(convo_layer_0);
  auto convo_layer_1 =
      CreateConvoLayer(network, &weight_map, convo_layer_0->getOutput(0),
                       get_width(128, width_multiplier), 3, 2, 1, "model.1");

  auto bottleneck_layer_2 = CreateC3Bottleneck(
      network, &weight_map, convo_layer_1->getOutput(0),
      get_width(128, width_multiplier), get_width(128, width_multiplier),
      get_depth(3, depth_multiplier), true, 1, 0.5, "model.2");

  auto convo_layer_3 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_2->getOutput(0),
                       get_width(256, width_multiplier), 3, 2, 1, "model.3");

  auto bottleneck_layer_4 = CreateC3Bottleneck(
      network, &weight_map, convo_layer_3->getOutput(0),
      get_width(256, width_multiplier), get_width(256, width_multiplier),
      get_depth(6, depth_multiplier), true, 1, 0.5, "model.4");

  auto convo_layer_5 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_4->getOutput(0),
                       get_width(512, width_multiplier), 3, 2, 1, "model.5");

  auto bottleneck_layer_6 = CreateC3Bottleneck(
      network, &weight_map, convo_layer_5->getOutput(0),
      get_width(512, width_multiplier), get_width(512, width_multiplier),
      get_depth(9, depth_multiplier), true, 1, 0.5, "model.6");

  auto convo_layer_7 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_6->getOutput(0),
                       get_width(1024, width_multiplier), 3, 2, 1, "model.7");

  auto bottleneck_layer_8 = CreateC3Bottleneck(
      network, &weight_map, convo_layer_7->getOutput(0),
      get_width(1024, width_multiplier), get_width(1024, width_multiplier),
      get_depth(3, depth_multiplier), true, 1, 0.5, "model.8");

  auto spp_layer_9 =
      CreateSPPFLayer(network, &weight_map, bottleneck_layer_8->getOutput(0),
                      get_width(1024, width_multiplier),
                      get_width(1024, width_multiplier), 5, "model.9");

  // Head layer
  auto convo_layer_10 =
      CreateConvoLayer(network, &weight_map, spp_layer_9->getOutput(0),
                       get_width(512, width_multiplier), 1, 1, 1, "model.10");

  // Layer increasing spatial resolution of image
  auto upsample_layer_11 = network->addResize(*convo_layer_10->getOutput(0));
  CHECK_NOTNULL(upsample_layer_11);
  upsample_layer_11->setResizeMode(ResizeMode::kNEAREST);
  upsample_layer_11->setOutputDimensions(
      bottleneck_layer_6->getOutput(0)->getDimensions());

  ITensor *input_tensors_layer_12[] = {upsample_layer_11->getOutput(0),
                                       bottleneck_layer_6->getOutput(0)};

  auto concatenation_layer_12 =
      network->addConcatenation(input_tensors_layer_12, 2);

  auto bottleneck_layer_13 = CreateC3Bottleneck(
      network, &weight_map, concatenation_layer_12->getOutput(0),
      get_width(1024, width_multiplier), get_width(512, width_multiplier),
      get_depth(3, depth_multiplier), false, 1, 0.5, "model.13");

  auto convo_layer_14 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_13->getOutput(0),
                       get_width(256, width_multiplier), 1, 1, 1, "model.14");

  auto upsample_layer_15 = network->addResize(*convo_layer_14->getOutput(0));
  CHECK_NOTNULL(upsample_layer_15);
  upsample_layer_15->setResizeMode(ResizeMode::kNEAREST);
  upsample_layer_15->setOutputDimensions(
      bottleneck_layer_4->getOutput(0)->getDimensions());

  ITensor *input_tensors_layer_16[] = {upsample_layer_15->getOutput(0),
                                       bottleneck_layer_4->getOutput(0)};
  auto concatenation_layer_16 =
      network->addConcatenation(input_tensors_layer_16, 2);

  auto bottleneck_layer_17 = CreateC3Bottleneck(
      network, &weight_map, concatenation_layer_16->getOutput(0),
      get_width(512, width_multiplier), get_width(256, width_multiplier),
      get_depth(3, depth_multiplier), false, 1, 0.5, "model.17");

  // Detect
  IConvolutionLayer *det0 = network->addConvolutionNd(
      *bottleneck_layer_17->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.0.weight"], weight_map["model.24.m.0.bias"]);

  auto convo_layer_18 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_17->getOutput(0),
                       get_width(256, width_multiplier), 3, 2, 1, "model.18");

  ITensor *input_tensors_layer_19[] = {convo_layer_18->getOutput(0),
                                       convo_layer_14->getOutput(0)};
  auto cat19 = network->addConcatenation(input_tensors_layer_19, 2);

  auto bottleneck_layer_20 = CreateC3Bottleneck(
      network, &weight_map, cat19->getOutput(0),
      get_width(512, width_multiplier), get_width(512, width_multiplier),
      get_depth(3, depth_multiplier), false, 1, 0.5, "model.20");

  IConvolutionLayer *det1 = network->addConvolutionNd(
      *bottleneck_layer_20->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.1.weight"], weight_map["model.24.m.1.bias"]);

  auto convo_layer_21 =
      CreateConvoLayer(network, &weight_map, bottleneck_layer_20->getOutput(0),
                       get_width(512, width_multiplier), 3, 2, 1, "model.21");

  ITensor *input_tensors_layer_22[] = {convo_layer_21->getOutput(0),
                                       convo_layer_10->getOutput(0)};

  auto concatenation_layer_22 =
      network->addConcatenation(input_tensors_layer_22, 2);

  auto bottleneck_layer_23 = CreateC3Bottleneck(
      network, &weight_map, concatenation_layer_22->getOutput(0),
      get_width(1024, width_multiplier), get_width(1024, width_multiplier),
      get_depth(3, depth_multiplier), false, 1, 0.5, "model.23");

  IConvolutionLayer *det2 = network->addConvolutionNd(
      *bottleneck_layer_23->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.2.weight"], weight_map["model.24.m.2.bias"]);

  auto yolo_layer =
      AddYoLoLayer(network, &weight_map, "model.24",
                   std::vector<IConvolutionLayer *>{det0, det1, det2});
  yolo_layer->getOutput(0)->setName(kOutputTensorName);
  network->markOutput(*yolo_layer->getOutput(0));

  // Engine config
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#endif

  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  return engine;
}
