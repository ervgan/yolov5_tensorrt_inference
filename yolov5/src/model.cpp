#include "model.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include "calibrator.h"
#include "config.h"
#include "yololayer.h"

using namespace nvinfer1;

// file comments

namespace {

// Loads wts file and returns a map of names with correpsonding weights
// TensorRT wts weight files have a simple space delimited format :
// [type] [size] <data x size in hex>
std::map<std::string, Weights> LoadWeights(const std::string& file) {
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, Weights> weight_map;
  std::ifstream input(file);
  assert(input.is_open() &&
         "Unable to load weight file. please check if the .wts file path");
  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");
  // get weights for each
  while (count--) {
    Weights weight<{DataType::kFLOAT, nullptr, 0}>;
    uint32_t size;
    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    weight.type = DataType::kFLOAT;
    // Load blob
    uint32_t* weights =
        reinterpret_cast<uint32_t*>(malloc(sizeof(weights) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input >> std::hex >> weights[x];
    }
    weight.values = weights;
    weight.count = size;
    weight_map[name] = weight;
  }
  return weight_map;
}

// nb of output feature maps
int GetWidth(int x, float gw) {
  const int divisor = 8;
  return int(ceil((x * gw) / divisor)) * divisor;
}

int GetWidth(int x, float gw, int divisor) {
  return int(ceil((x * gw) / divisor)) * divisor;
}

int GetDepth(int x, float gd) {
  if (x == 1) return 1;
  int r = round(x * gd);
  if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
    --r;
  }
  return std::max<int>(r, 1);
}

// add batch normalization to address internal covariate shift
// returns normalized layer
IScaleLayer* AddBatchNorm2D(INetworkDefinition* network,
                            std::map<std::string, Weights>& weight_map,
                            const ITensor& input, const std::string& layer_name,
                            float eps) {
  float* gamma = (float*)weight_map[layer_name + ".weight"].values;
  float* beta = (float*)weight_map[layer_name + ".bias"].values;
  float* mean = (float*)weight_map[layer_name + ".running_mean"].values;
  float* var = (float*)weight_map[layer_name + ".running_var"].values;
  int len = weight_map[layer_name + ".running_var"].count;

  float* scale_values = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    scale_values[i] = gamma[i] / sqrt(var[i] + eps);
  }
  Weights scale{DataType::kFLOAT, scale_values, len};

  float* shift_values = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    shift_values[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
  }
  Weights shift{DataType::kFLOAT, shift_values, len};

  float* power_values = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    power_values[i] = 1.0;
  }
  Weights power{DataType::kFLOAT, power_values, len};

  weight_map[layer_name + ".scale"] = scale;
  weight_map[layer_name + ".shift"] = shift;
  weight_map[layer_name + ".power"] = power;

  IScaleLayer* scale_layer =
      network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
  assert(scale_layer);
  return scale_layer;
}

// returns a single TensorRT layer
ILayer* CreateConvoLayer(INetworkDefinition* network,
                         std::map<std::string, Weights>& weight_map,
                         const ITensor& input, int output, int kernel_size,
                         int stride, int nb_groups, std::string layer_name) {
  Weights empty_wts{DataType::kFLOAT, nullptr, 0};
  const int padding = kernel_size / 3;
  IConvolutionLayer* convo_layer = network->addConvolutionNd(
      input, output, DimsHW{kernel_size, kernel_size},
      weight_map[layer_name + ".conv.weight"], empty_wts);

  assert(convo_layer);

  convo_layer->setStrideNd(DimsHW{stride, stride});
  convo_layer->setPaddingNd(DimsHW{padding, padding});
  convo_layer->setNbGroups(nb_groups);
  convo_layer->setName((layer_name + ".conv").c_str());
  IScaleLayer* batch_norm_layer =
      AddBatchNorm2D(network, weight_map, *convo_layer->getOutput(0),
                     layer_name + ".bn", 1e-3);

  // using silu activation method = input * sigmoid
  auto sigmoid_activation = network->addActivation(
      *batch_norm_layer->getOutput(0), ActivationType::kSIGMOID);
  assert(sigmoid_activation);
  auto silu_activation =
      network->addElementWise(*batch_norm_layer->getOutput(0),
                              *sig->getOutput(0), ElementWiseOperation::kPROD);
  assert(silu_activation);
  return silu_activation;
}

ILayer* bottleneck(INetworkDefinition* network,
                   std::map<std::string, Weights>& weight_map, ITensor& input,
                   int c1, int c2, bool shortcut, int g, float e,
                   std::string layer_name) {
  auto cv1 = CreateConvoLayer(network, weight_map, input, (int)((float)c2 * e),
                              1, 1, 1, layer_name + ".cv1");
  auto cv2 = CreateConvoLayer(network, weight_map, *cv1->getOutput(0), c2, 3, 1,
                              g, layer_name + ".cv2");
  if (shortcut && c1 == c2) {
    auto ew = network->addElementWise(input, *cv2->getOutput(0),
                                      ElementWiseOperation::kSUM);
    return ew;
  }
  return cv2;
}

ILayer* C3(INetworkDefinition* network,
           std::map<std::string, Weights>& weight_map, ITensor& input, int c1,
           int c2, int n, bool shortcut, int g, float e,
           std::string layer_name) {
  int c_ = (int)((float)c2 * e);
  auto cv1 = CreateConvoLayer(network, weight_map, input, c_, 1, 1, 1,
                              layer_name + ".cv1");
  auto cv2 = CreateConvoLayer(network, weight_map, input, c_, 1, 1, 1,
                              layer_name + ".cv2");
  ITensor* y1 = cv1->getOutput(0);
  for (int i = 0; i < n; i++) {
    auto b = bottleneck(network, weight_map, *y1, c_, c_, shortcut, g, 1.0,
                        layer_name + ".m." + std::to_string(i));
    y1 = b->getOutput(0);
  }

  ITensor* inputTensors[] = {y1, cv2->getOutput(0)};
  auto cat = network->addConcatenation(inputTensors, 2);

  auto cv3 = CreateConvoLayer(network, weight_map, *cat->getOutput(0), c2, 1, 1,
                              1, layer_name + ".cv3");
  return cv3;
}

ILayer* SPP(INetworkDefinition* network,
            std::map<std::string, Weights>& weight_map, ITensor& input, int c1,
            int c2, int k1, int k2, int k3, std::string layer_name) {
  int c_ = c1 / 2;
  auto cv1 = CreateConvoLayer(network, weight_map, input, c_, 1, 1, 1,
                              layer_name + ".cv1");

  auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k1, k1});
  pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
  pool1->setStrideNd(DimsHW{1, 1});
  auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k2, k2});
  pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
  pool2->setStrideNd(DimsHW{1, 1});
  auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k3, k3});
  pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
  pool3->setStrideNd(DimsHW{1, 1});

  ITensor* inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0),
                             pool2->getOutput(0), pool3->getOutput(0)};
  auto cat = network->addConcatenation(inputTensors, 4);

  auto cv2 = CreateConvoLayer(network, weight_map, *cat->getOutput(0), c2, 1, 1,
                              1, layer_name + ".cv2");
  return cv2;
}

ILayer* SPPF(INetworkDefinition* network,
             std::map<std::string, Weights>& weight_map, ITensor& input, int c1,
             int c2, int k, std::string layer_name) {
  int c_ = c1 / 2;
  auto cv1 = CreateConvoLayer(network, weight_map, input, c_, 1, 1, 1,
                              layer_name + ".cv1");

  auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k, k});
  pool1->setPaddingNd(DimsHW{k / 2, k / 2});
  pool1->setStrideNd(DimsHW{1, 1});
  auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k, k});
  pool2->setPaddingNd(DimsHW{k / 2, k / 2});
  pool2->setStrideNd(DimsHW{1, 1});
  auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX,
                                     DimsHW{k, k});
  pool3->setPaddingNd(DimsHW{k / 2, k / 2});
  pool3->setStrideNd(DimsHW{1, 1});
  ITensor* inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0),
                             pool2->getOutput(0), pool3->getOutput(0)};
  auto cat = network->addConcatenation(inputTensors, 4);
  auto cv2 = CreateConvoLayer(network, weight_map, *cat->getOutput(0), c2, 1, 1,
                              1, layer_name + ".cv2");
  return cv2;
}

ILayer* Proto(INetworkDefinition* network,
              std::map<std::string, Weights>& weight_map, ITensor& input,
              int c_, int c2, std::string layer_name) {
  auto cv1 = CreateConvoLayer(network, weight_map, input, c_, 3, 1, 1,
                              layer_name + ".cv1");

  auto upsample = network->addResize(*cv1->getOutput(0));
  assert(upsample);
  upsample->setResizeMode(ResizeMode::kNEAREST);
  const float scales[] = {1, 2, 2};
  upsample->setScales(scales, 3);

  auto cv2 = CreateConvoLayer(network, weight_map, *upsample->getOutput(0), c_,
                              3, 1, 1, layer_name + ".cv2");
  auto cv3 = CreateConvoLayer(network, weight_map, *cv2->getOutput(0), c2, 1, 1,
                              1, layer_name + ".cv3");
  assert(cv3);
  return cv3;
}

std::vector<std::vector<float>> getAnchors(
    std::map<std::string, Weights>& weight_map, std::string layer_name) {
  std::vector<std::vector<float>> anchors;
  Weights wts = weight_map[layer_name + ".anchor_grid"];
  int anchor_len = kNumAnchor * 2;
  for (int i = 0; i < wts.count / anchor_len; i++) {
    auto* p = (const float*)wts.values + i * anchor_len;
    std::vector<float> anchor(p, p + anchor_len);
    anchors.push_back(anchor);
  }
  return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition* network,
                             std::map<std::string, Weights>& weight_map,
                             std::string layer_name,
                             std::vector<IConvolutionLayer*> dets,
                             bool is_segmentation = false) {
  auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
  auto anchors = getAnchors(weight_map, layer_name);
  PluginField plugin_fields[2];
  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox,
                    (int)is_segmentation};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;

  // load strides from Detect layer
  assert(weight_map.find(layer_name + ".strides") != weight_map.end() &&
         "Not found `strides`, please check gen_wts.py!!!");
  Weights strides = weight_map[layer_name + ".strides"];
  auto* p = (const float*)(strides.values);
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
  IPluginV2* plugin_obj = creator->createPlugin("yololayer", &plugin_data);
  std::vector<ITensor*> input_tensors;
  for (auto det : dets) {
    input_tensors.push_back(det->getOutput(0));
  }
  auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(),
                                   *plugin_obj);
  return yolo;
}

}  // namespace

ICudaEngine* build_det_engine(unsigned int maxBatchSize, IBuilder* builder,
                              IBuilderConfig* config, DataType dt, float& gd,
                              float& gw, std::string& wts_name) {
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, kInputH, kInputW}
  ITensor* data =
      network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
  assert(data);
  std::map<std::string, Weights> weight_map = loadWeights(wts_name);

  // Backbone
  auto conv0 = CreateConvoLayer(network, weight_map, *data, GetWidth(64, gw), 6,
                                2, 1, "model.0");
  assert(conv0);
  auto conv1 = CreateConvoLayer(network, weight_map, *conv0->getOutput(0),
                                GetWidth(128, gw), 3, 2, 1, "model.1");

  auto bottleneck_CSP2 =
      C3(network, weight_map, *conv1->getOutput(0), GetWidth(128, gw),
         GetWidth(128, gw), GetDepth(3, gd), true, 1, 0.5, "model.2");
  auto conv3 =
      CreateConvoLayer(network, weight_map, *bottleneck_CSP2->getOutput(0),
                       GetWidth(256, gw), 3, 2, 1, "model.3");

  auto bottleneck_csp4 =
      C3(network, weight_map, *conv3->getOutput(0), GetWidth(256, gw),
         GetWidth(256, gw), GetDepth(6, gd), true, 1, 0.5, "model.4");
  auto conv5 =
      CreateConvoLayer(network, weight_map, *bottleneck_csp4->getOutput(0),
                       GetWidth(512, gw), 3, 2, 1, "model.5");

  auto bottleneck_csp6 =
      C3(network, weight_map, *conv5->getOutput(0), GetWidth(512, gw),
         GetWidth(512, gw), GetDepth(9, gd), true, 1, 0.5, "model.6");
  auto conv7 =
      CreateConvoLayer(network, weight_map, *bottleneck_csp6->getOutput(0),
                       GetWidth(1024, gw), 3, 2, 1, "model.7");

  auto bottleneck_csp8 =
      C3(network, weight_map, *conv7->getOutput(0), GetWidth(1024, gw),
         GetWidth(1024, gw), GetDepth(3, gd), true, 1, 0.5, "model.8");
  auto spp9 = SPPF(network, weight_map, *bottleneck_csp8->getOutput(0),
                   GetWidth(1024, gw), GetWidth(1024, gw), 5, "model.9");

  // Head
  auto conv10 = CreateConvoLayer(network, weight_map, *spp9->getOutput(0),
                                 GetWidth(512, gw), 1, 1, 1, "model.10");

  auto upsample11 = network->addResize(*conv10->getOutput(0));
  assert(upsample11);
  upsample11->setResizeMode(ResizeMode::kNEAREST);
  upsample11->setOutputDimensions(
      bottleneck_csp6->getOutput(0)->getDimensions());

  ITensor* inputTensors12[] = {upsample11->getOutput(0),
                               bottleneck_csp6->getOutput(0)};
  auto cat12 = network->addConcatenation(inputTensors12, 2);
  auto bottleneck_csp13 =
      C3(network, weight_map, *cat12->getOutput(0), GetWidth(1024, gw),
         GetWidth(512, gw), GetDepth(3, gd), false, 1, 0.5, "model.13");
  auto conv14 =
      CreateConvoLayer(network, weight_map, *bottleneck_csp13->getOutput(0),
                       GetWidth(256, gw), 1, 1, 1, "model.14");

  auto upsample15 = network->addResize(*conv14->getOutput(0));
  assert(upsample15);
  upsample15->setResizeMode(ResizeMode::kNEAREST);
  upsample15->setOutputDimensions(
      bottleneck_csp4->getOutput(0)->getDimensions());

  ITensor* inputTensors16[] = {upsample15->getOutput(0),
                               bottleneck_csp4->getOutput(0)};
  auto cat16 = network->addConcatenation(inputTensors16, 2);

  auto bottleneck_csp17 =
      C3(network, weight_map, *cat16->getOutput(0), GetWidth(512, gw),
         GetWidth(256, gw), GetDepth(3, gd), false, 1, 0.5, "model.17");

  // Detect
  IConvolutionLayer* det0 = network->addConvolutionNd(
      *bottleneck_csp17->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.0.weight"], weight_map["model.24.m.0.bias"]);
  auto conv18 =
      CreateConvoLayer(network, weight_map, *bottleneck_csp17->getOutput(0),
                       GetWidth(256, gw), 3, 2, 1, "model.18");
  ITensor* inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
  auto cat19 = network->addConcatenation(inputTensors19, 2);
  auto bottleneck_csp20 =
      C3(network, weight_map, *cat19->getOutput(0), GetWidth(512, gw),
         GetWidth(512, gw), GetDepth(3, gd), false, 1, 0.5, "model.20");
  IConvolutionLayer* det1 = network->addConvolutionNd(
      *bottleneck_csp20->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.1.weight"], weight_map["model.24.m.1.bias"]);
  auto conv21 =
      CreateConvoLayer(network, weight_map, *bottleneck_csp20->getOutput(0),
                       GetWidth(512, gw), 3, 2, 1, "model.21");
  ITensor* inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
  auto cat22 = network->addConcatenation(inputTensors22, 2);
  auto bottleneck_csp23 =
      C3(network, weight_map, *cat22->getOutput(0), GetWidth(1024, gw),
         GetWidth(1024, gw), GetDepth(3, gd), false, 1, 0.5, "model.23");
  IConvolutionLayer* det2 = network->addConvolutionNd(
      *bottleneck_csp23->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.24.m.2.weight"], weight_map["model.24.m.2.bias"]);

  auto yolo = addYoLoLayer(network, weight_map, "model.24",
                           std::vector<IConvolutionLayer*>{det0, det1, det2});
  yolo->getOutput(0)->setName(kOutputTensorName);
  network->markOutput(*yolo->getOutput(0));

  // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: "
            << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  Int8EntropyCalibrator2* calibrator =
      new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/",
                                 "int8calib.table", kInputTensorName);
  config->setInt8Calibrator(calibrator);
#endif

  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto& mem : weight_map) {
    free((void*)(mem.second.values));
  }

  return engine;
}

ICudaEngine* build_det_p6_engine(unsigned int maxBatchSize, IBuilder* builder,
                                 IBuilderConfig* config, DataType dt, float& gd,
                                 float& gw, std::string& wts_name) {
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, kInputH, kInputW}
  ITensor* data =
      network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
  assert(data);

  std::map<std::string, Weights> weight_map = loadWeights(wts_name);

  // Backbone
  auto conv0 = CreateConvoLayer(network, weight_map, *data, GetWidth(64, gw), 6,
                                2, 1, "model.0");
  auto conv1 = CreateConvoLayer(network, weight_map, *conv0->getOutput(0),
                                GetWidth(128, gw), 3, 2, 1, "model.1");
  auto c3_2 = C3(network, weight_map, *conv1->getOutput(0), GetWidth(128, gw),
                 GetWidth(128, gw), GetDepth(3, gd), true, 1, 0.5, "model.2");
  auto conv3 = CreateConvoLayer(network, weight_map, *c3_2->getOutput(0),
                                GetWidth(256, gw), 3, 2, 1, "model.3");
  auto c3_4 = C3(network, weight_map, *conv3->getOutput(0), GetWidth(256, gw),
                 GetWidth(256, gw), GetDepth(6, gd), true, 1, 0.5, "model.4");
  auto conv5 = CreateConvoLayer(network, weight_map, *c3_4->getOutput(0),
                                GetWidth(512, gw), 3, 2, 1, "model.5");
  auto c3_6 = C3(network, weight_map, *conv5->getOutput(0), GetWidth(512, gw),
                 GetWidth(512, gw), GetDepth(9, gd), true, 1, 0.5, "model.6");
  auto conv7 = CreateConvoLayer(network, weight_map, *c3_6->getOutput(0),
                                GetWidth(768, gw), 3, 2, 1, "model.7");
  auto c3_8 = C3(network, weight_map, *conv7->getOutput(0), GetWidth(768, gw),
                 GetWidth(768, gw), GetDepth(3, gd), true, 1, 0.5, "model.8");
  auto conv9 = CreateConvoLayer(network, weight_map, *c3_8->getOutput(0),
                                GetWidth(1024, gw), 3, 2, 1, "model.9");
  auto c3_10 =
      C3(network, weight_map, *conv9->getOutput(0), GetWidth(1024, gw),
         GetWidth(1024, gw), GetDepth(3, gd), true, 1, 0.5, "model.10");
  auto sppf11 = SPPF(network, weight_map, *c3_10->getOutput(0),
                     GetWidth(1024, gw), GetWidth(1024, gw), 5, "model.11");

  // Head
  auto conv12 = CreateConvoLayer(network, weight_map, *sppf11->getOutput(0),
                                 GetWidth(768, gw), 1, 1, 1, "model.12");
  auto upsample13 = network->addResize(*conv12->getOutput(0));
  assert(upsample13);
  upsample13->setResizeMode(ResizeMode::kNEAREST);
  upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
  ITensor* inputTensors14[] = {upsample13->getOutput(0), c3_8->getOutput(0)};
  auto cat14 = network->addConcatenation(inputTensors14, 2);
  auto c3_15 =
      C3(network, weight_map, *cat14->getOutput(0), GetWidth(1536, gw),
         GetWidth(768, gw), GetDepth(3, gd), false, 1, 0.5, "model.15");

  auto conv16 = CreateConvoLayer(network, weight_map, *c3_15->getOutput(0),
                                 GetWidth(512, gw), 1, 1, 1, "model.16");
  auto upsample17 = network->addResize(*conv16->getOutput(0));
  assert(upsample17);
  upsample17->setResizeMode(ResizeMode::kNEAREST);
  upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
  ITensor* inputTensors18[] = {upsample17->getOutput(0), c3_6->getOutput(0)};
  auto cat18 = network->addConcatenation(inputTensors18, 2);
  auto c3_19 =
      C3(network, weight_map, *cat18->getOutput(0), GetWidth(1024, gw),
         GetWidth(512, gw), GetDepth(3, gd), false, 1, 0.5, "model.19");

  auto conv20 = CreateConvoLayer(network, weight_map, *c3_19->getOutput(0),
                                 GetWidth(256, gw), 1, 1, 1, "model.20");
  auto upsample21 = network->addResize(*conv20->getOutput(0));
  assert(upsample21);
  upsample21->setResizeMode(ResizeMode::kNEAREST);
  upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
  ITensor* inputTensors21[] = {upsample21->getOutput(0), c3_4->getOutput(0)};
  auto cat22 = network->addConcatenation(inputTensors21, 2);
  auto c3_23 =
      C3(network, weight_map, *cat22->getOutput(0), GetWidth(512, gw),
         GetWidth(256, gw), GetDepth(3, gd), false, 1, 0.5, "model.23");

  auto conv24 = CreateConvoLayer(network, weight_map, *c3_23->getOutput(0),
                                 GetWidth(256, gw), 3, 2, 1, "model.24");
  ITensor* inputTensors25[] = {conv24->getOutput(0), conv20->getOutput(0)};
  auto cat25 = network->addConcatenation(inputTensors25, 2);
  auto c3_26 =
      C3(network, weight_map, *cat25->getOutput(0), GetWidth(1024, gw),
         GetWidth(512, gw), GetDepth(3, gd), false, 1, 0.5, "model.26");

  auto conv27 = CreateConvoLayer(network, weight_map, *c3_26->getOutput(0),
                                 GetWidth(512, gw), 3, 2, 1, "model.27");
  ITensor* inputTensors28[] = {conv27->getOutput(0), conv16->getOutput(0)};
  auto cat28 = network->addConcatenation(inputTensors28, 2);
  auto c3_29 =
      C3(network, weight_map, *cat28->getOutput(0), GetWidth(1536, gw),
         GetWidth(768, gw), GetDepth(3, gd), false, 1, 0.5, "model.29");

  auto conv30 = CreateConvoLayer(network, weight_map, *c3_29->getOutput(0),
                                 GetWidth(768, gw), 3, 2, 1, "model.30");
  ITensor* inputTensors31[] = {conv30->getOutput(0), conv12->getOutput(0)};
  auto cat31 = network->addConcatenation(inputTensors31, 2);
  auto c3_32 =
      C3(network, weight_map, *cat31->getOutput(0), GetWidth(2048, gw),
         GetWidth(1024, gw), GetDepth(3, gd), false, 1, 0.5, "model.32");

  // Detect
  IConvolutionLayer* det0 = network->addConvolutionNd(
      *c3_23->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.33.m.0.weight"], weight_map["model.33.m.0.bias"]);
  IConvolutionLayer* det1 = network->addConvolutionNd(
      *c3_26->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.33.m.1.weight"], weight_map["model.33.m.1.bias"]);
  IConvolutionLayer* det2 = network->addConvolutionNd(
      *c3_29->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.33.m.2.weight"], weight_map["model.33.m.2.bias"]);
  IConvolutionLayer* det3 = network->addConvolutionNd(
      *c3_32->getOutput(0), 3 * (kNumClass + 5), DimsHW{1, 1},
      weight_map["model.33.m.3.weight"], weight_map["model.33.m.3.bias"]);

  auto yolo =
      addYoLoLayer(network, weight_map, "model.33",
                   std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
  yolo->getOutput(0)->setName(kOutputTensorName);
  network->markOutput(*yolo->getOutput(0));

  // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#endif

  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto& mem : weight_map) {
    free((void*)(mem.second.values));
  }

  return engine;
}
