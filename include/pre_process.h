#ifndef YOLOV5_INFERENCE_PRE_PROCESS_H_
#define YOLOV5_INFERENCE_PRE_PROCESS_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

namespace yolov5_inference {
void CudaPreprocessInit(int max_image_size);
void CudaPreprocessDestroy();
// Applies an affine transformation to an image and copies it in GPU buffer
// memory
void CudaPreprocessBatch(std::vector<cv::Mat>* image_batch,
                         int processing_image_width,
                         int processing_image_height, cudaStream_t stream,
                         float* output_gpu_image_buffer);

void CudaPreprocess(uint8_t* image, int image_width, int image_height,
                    int processing_image_width, int processing_image_height,
                    cudaStream_t stream, float* output_gpu_image_buffer);
}  // namespace yolov5_inference

#endif  // YOLOV5_INFERENCE_PRE_PROCESS_H_
