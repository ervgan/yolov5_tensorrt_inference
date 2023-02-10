## Different versions of yolov5

Currently, support for yolov5 v1.0, v2.0, v3.0, v3.1, v4.0, v5.0, v6.0, v6.2, v7.0

# TensorRTx

TensorRTx aims to implement popular deep learning networks with TensorRT network definition API.

This repo is forked from: https://github.com/wang-xinyu/tensorrtx and is adapted for detection only using Yolov5.


The basic workflow of TensorRTx is:
1. Get the trained models from pytorch, mxnet or tensorflow, etc. Some pytorch models can be found in my repo [pytorchx](https://github.com/wang-xinyu/pytorchx), the remaining are from popular open-source repos.
2. Export the weights to a plain text file -- [.wts file](./tutorials/getting_started.md#the-wts-content-format).
3. Load weights in TensorRT, define the network, build a TensorRT engine.
4. Load the TensorRT engine and run inference.

## Config

- Choose the YOLOv5 sub-model n/s/m/l/x from command line arguments.
- Other configs please check [src/config.h](src/config.h)

## Build and Run

### Detection

1. generate .wts from pytorch with .pt

```
# For example using the yolov5s model
python wts_converter.py -w yolov5s.pt -o yolov5s.wts
# A file 'yolov5s.wts' will be generated.
```

2. build tensorrtx/yolov5 and run

```
cd [PATH-TO-TENSORRTX]/yolov5/
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov5]/yolov5s.wts .
cmake ..
make

./yolov5_det -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file
./yolov5_det -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

# For example yolov5s
./yolov5_det -s yolov5s.wts yolov5s.engine s
./yolov5_det -d yolov5s.engine ../images

# For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
./yolov5_det -s yolov5_custom.wts yolov5.engine c 0.17 0.25
./yolov5_det -d yolov5.engine ../images
```

## Acknowledgments & Contact

Any contributions, questions and discussions are welcomed, contact me by following info.

E-mail: wangxinyu_es@163.com

WeChat ID: wangxinyu0375 (可加我微信进tensorrtx交流群，**备注：tensorrtx**)





