## Different versions of yolov5

Currently supports yolov5 v1.0, v2.0, v3.0, v3.1, v4.0, v5.0, v6.0, v6.2, v7.0

# TensorRTx

Implementation of yolov5 deep learning networks with TensorRT network definition API.

The basic workflow to run inference from a pytorch is as follows:
1. Get the trained models from pytorch.
2. Export the weights to a plain text file -- [.wts file] using the wts_converter.py file (see below for an example).
3. Load weights in TensorRT, define the network, build a TensorRT engine.
4. Load the TensorRT engine and run inference.

## Config

- Choose the YOLOv5 sub-model n/s/m/l/x from command line arguments.
- Other configs please check [include/config.h](include/config.h)

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

## Acknowledgments

For more info, refer to https://github.com/wang-xinyu/tensorrtx





