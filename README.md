# TensorRTx

TensorRTx aims to implement popular deep learning networks with TensorRT network definition API.

This repo is forked from: https://github.com/wang-xinyu/tensorrtx and is adapted for detection only using Yolov5.


The basic workflow of TensorRTx is:
1. Get the trained models from pytorch, mxnet or tensorflow, etc. Some pytorch models can be found in my repo [pytorchx](https://github.com/wang-xinyu/pytorchx), the remaining are from popular open-source repos.
2. Export the weights to a plain text file -- [.wts file](./tutorials/getting_started.md#the-wts-content-format).
3. Load weights in TensorRT, define the network, build a TensorRT engine.
4. Load the TensorRT engine and run inference.

## Acknowledgments & Contact

Any contributions, questions and discussions are welcomed, contact me by following info.

E-mail: wangxinyu_es@163.com

WeChat ID: wangxinyu0375 (可加我微信进tensorrtx交流群，**备注：tensorrtx**)
