## Dependencies and versions

Supports yolov5 v1.0, v2.0, v3.0, v3.1, v4.0, v5.0, v6.0, v6.2, v7.0

Dependencies:
- wts_converter: yolov5 repo
- yolov5_inference: tensorRT 7x or 8x < 8.6.0, CUDA, openCV

# Yolov5 TensorRT implementation

Implementation of yolov5 deep learning networks with TensorRT network definition API.

The basic workflow to run inference from a pytorch is as follows:
1. Get the trained model from pytorch.
2. Export the weights to a plain text file -- [.wts file] using the wts_converter.py file (see below for an example).
--- Skip the first two steps if you already converted the pytorch model in .wts format during Google Colab training using the wts_converter.py stored on Google Drive
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
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
cd yolov5/
cp wts_converter.py .
python wts_converter.py -w yolov5s.pt -o yolov5s.wts
# A file 'yolov5s.wts' will be generated.
```

2. build yolov5_inference and run

```
cd [PATH-TO-yolov5_inference]/
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-wts_file]/yolov5s.wts .
cmake ..
make

./main -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file
./main -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.

# For example using yolov5s weights
./main -s yolov5s.wts yolov5s.engine s
./main -d yolov5s.engine ../images

