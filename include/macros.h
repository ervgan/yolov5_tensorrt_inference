#ifndef YOLOV5_INFERENCE_MACROS_H_
#define YOLOV5_INFERENCE_MACROS_H_

#include <NvInfer.h>

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif  // YOLOV5_INFERENCE_MACROS_H_
