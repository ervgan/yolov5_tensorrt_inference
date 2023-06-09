#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

void CudaPreprocessInit(int max_image_size);
void CudaPreprocessDestroy();
void CudaPreprocessBatch(std::vector<cv::Mat> *image_batch, float *image_buffer,
                         int processing_image_width,
                         int processing_image_height, cudaStream_t stream);
