#include <glog/logging.h>

#include "../include/cuda_utils.h"
#include "../include/pre_process.h"

namespace yolov5_inference {
static uint8_t* image_buffer_host = nullptr;
static uint8_t* image_buffer_device = nullptr;

namespace {
struct AffineMatrix {
  float value[6];
};

// cuda implementation of openCV cv::warpAffine method
// performs an affine transformation to the input image
// and performs HWC2CHW conversion in addition
__global__ void WarpAffineKernel(uint8_t* image_buffer, int image_line_size,
                                 int image_width, int image_height,
                                 int processing_image_width,
                                 int processing_image_height,
                                 uint8_t constant_rgb_value,
                                 AffineMatrix invert_scalar, int image_edge,
                                 float* output_gpu_image_buffer) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= image_edge) return;

  // get affine transformation
  float rotation_x = invert_scalar.value[0];
  float scaling_x = invert_scalar.value[1];
  float translation_x = invert_scalar.value[2];
  float rotation_y = invert_scalar.value[3];
  float scaling_y = invert_scalar.value[4];
  float translation_y = invert_scalar.value[5];

  int destination_x = thread_id % processing_image_width;
  int destination_y = thread_id / processing_image_width;
  float image_x = rotation_x * destination_x + scaling_x * destination_y +
                  translation_x + 0.5f;
  float image_y = rotation_y * destination_x + scaling_y * destination_y +
                  translation_y + 0.5f;
  float color_0, color_1, color_2;

  // out of range
  if (image_x <= -1 || image_x >= image_width || image_y <= -1 ||
      image_y >= image_height) {
    // constant_rbg_value = 128
    color_0 = constant_rgb_value;
    color_1 = constant_rgb_value;
    color_2 = constant_rgb_value;
  } else {
    // bilinear interpolation to obtain RGB values
    // of pixel (x,y) by averaging four nearby pixels

    // get the x and y of the four neighbor pixels
    int y_low = floorf(image_y);
    int x_low = floorf(image_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    float ly = image_y - y_low;
    float lx = image_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;

    uint8_t constant_rgb_values[] = {constant_rgb_value, constant_rgb_value,
                                     constant_rgb_value};

    float weight_pixel_1 = hy * hx, weight_pixel_2 = hy * lx,
          weight_pixel_3 = ly * hx, weight_pixel_4 = ly * lx;
    uint8_t* color_pixel_1 = constant_rgb_values;
    uint8_t* color_pixel_2 = constant_rgb_values;
    uint8_t* color_pixel_3 = constant_rgb_values;
    uint8_t* color_pixel_4 = constant_rgb_values;

    if (y_low >= 0) {
      if (x_low >= 0)
        // get memory address of first neighbor pixel
        color_pixel_1 = image_buffer + y_low * image_line_size + x_low * 3;

      if (x_high < image_width)
        // get memory address of second neighbor pixel
        color_pixel_2 = image_buffer + y_low * image_line_size + x_high * 3;
    }

    if (y_high < image_height) {
      if (x_low >= 0)
        // get memory address of third neighbor pixel
        color_pixel_3 = image_buffer + y_high * image_line_size + x_low * 3;

      if (x_high < image_width)
        // get memory address of fourth neighbor pixel
        color_pixel_4 = image_buffer + y_high * image_line_size + x_high * 3;
    }

    color_0 =
        weight_pixel_1 * color_pixel_1[0] + weight_pixel_2 * color_pixel_2[0] +
        weight_pixel_3 * color_pixel_3[0] + weight_pixel_4 * color_pixel_4[0];
    color_1 =
        weight_pixel_1 * color_pixel_1[1] + weight_pixel_2 * color_pixel_2[1] +
        weight_pixel_3 * color_pixel_3[1] + weight_pixel_4 * color_pixel_4[1];
    color_2 =
        weight_pixel_1 * color_pixel_1[2] + weight_pixel_2 * color_pixel_2[2] +
        weight_pixel_3 * color_pixel_3[2] + weight_pixel_4 * color_pixel_4[2];
  }

  // convert bgr to rgb
  float temp_color = color_2;
  color_2 = color_0;
  color_0 = temp_color;

  // normalization
  color_0 = color_0 / 255.0f;
  color_1 = color_1 / 255.0f;
  color_2 = color_2 / 255.0f;

  // rgbrgbrgb to rrrgggbbb
  int image_area = processing_image_width * processing_image_height;
  float* color_0_pointer = output_gpu_image_buffer +
                           destination_y * processing_image_width +
                           destination_x;
  float* color_1_pointer = color_0_pointer + image_area;
  float* color_2_pointer = color_1_pointer + image_area;
  *color_0_pointer = color_0;
  *color_1_pointer = color_1;
  *color_2_pointer = color_2;
}
}  // namespace

// preprocess images by creating the affine tranformation
// and applying it to the original image by calling warpAffineKernel
void CudaPreprocess(uint8_t* image, int image_width, int image_height,
                    int processing_image_width, int processing_image_height,
                    cudaStream_t stream, float* output_gpu_image_buffer) {
  int image_size = image_width * image_height * 3;
  // copy data to CPU pinned memory
  memcpy(image_buffer_host, image, image_size);
  // copy data to GPU device memory
  CUDA_CHECK(cudaMemcpyAsync(image_buffer_device, image_buffer_host, image_size,
                             cudaMemcpyHostToDevice, stream));

  AffineMatrix original_scalar, invert_scalar;
  float scale =
      std::min(processing_image_height / static_cast<float>(image_height),
               processing_image_width / static_cast<float>(image_width));

  // rotation components
  original_scalar.value[0] = scale;
  original_scalar.value[1] = 0;
  // scaling components
  original_scalar.value[2] =
      -scale * image_width * 0.5 + processing_image_width * 0.5;
  original_scalar.value[3] = 0;
  // translation components
  original_scalar.value[4] = scale;
  original_scalar.value[5] =
      -scale * image_height * 0.5 + processing_image_height * 0.5;

  cv::Mat original_scalar_matrix(2, 3, CV_32F, original_scalar.value);
  cv::Mat invert_scalar_matrix(2, 3, CV_32F, invert_scalar.value);
  cv::invertAffineTransform(original_scalar_matrix, invert_scalar_matrix);

  memcpy(invert_scalar.value, invert_scalar_matrix.ptr<float>(0),
         sizeof(invert_scalar.value));

  // number of pixels to process = 640*640
  int jobs = processing_image_height * processing_image_width;
  // equivalent to a 2D thread block of size 16*16 dim3 (16,16,1)
  int threads = 256;
  // calculate number of blocks to be able to process all jobs
  int blocks = ceil(jobs / static_cast<float>(threads));

  WarpAffineKernel<<<blocks, threads, 0, stream>>>(
      image_buffer_device, image_width * 3, image_width, image_height,
      processing_image_width, processing_image_height, 128, invert_scalar, jobs,
      output_gpu_image_buffer);
  // synchronizes host CPU thread and the execution of a CUDA stream (sequence
  // of tasks)
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CudaPreprocessBatch(std::vector<cv::Mat>* image_batch,
                         int processing_image_width,
                         int processing_image_height, cudaStream_t stream,
                         float* output_gpu_image_buffer) {
  int processing_image_size =
      processing_image_width * processing_image_height * 3;
  for (size_t i = 0; i < image_batch->size(); ++i) {
    CudaPreprocess((*image_batch)[i].ptr(), (*image_batch)[i].cols,
                   (*image_batch)[i].rows, processing_image_width,
                   processing_image_height, stream,
                   &output_gpu_image_buffer[processing_image_size * i]);
    // synchronizes host CPU thread and the execution of a CUDA stream (sequence
    // of tasks)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

void CudaPreprocessInit(int max_image_size) {
  // prepare input data in  CPU pinned memory
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&image_buffer_host),
                            max_image_size * 3));
  // prepare input data in GPU device memory
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&image_buffer_device),
                        max_image_size * 3));
}

void CudaPreprocessDestroy() {
  CUDA_CHECK(cudaFree(image_buffer_device));
  CUDA_CHECK(cudaFreeHost(image_buffer_host));
}

}  // namespace yolov5_inference
