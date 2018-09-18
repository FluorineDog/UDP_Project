#include <time.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define PI 3.14159265358979323846
#define N 2048
#define M 256    //接收阵元到发射阵元的最大距离（阵元个数），所以接收孔径为2*M+1
#define ELE_NO 2048
#define OD 64
#define NSAMPLE 3750

int parallel_emit_sum = 1;

__device__ float dev_ele_coord_x[ELE_NO];
__device__ float dev_ele_coord_y[ELE_NO];    //写到纹理内存里面
__device__ float dev_filter_data[OD];        //filter parameter

float image_data[N * N] = {0};
int image_point_count[N * N] = {0};

// 滤波函数
// image_data as output location
// parallel_emit_sum is data trucks
// trans_sdata is raw input
// we have to use it against the best method
__global__ void calc_func(const int global_step, float *image_data,
                          int *point_count, const float *trans_sdata,
                          const int parallel_emit_sum) {
  int count = 1520;
  float fs = 25e6;
  float image_width = 200.0 / 1000;
  float image_length = 200.0 / 1000;
  float data_diameter = 220.0 / 1000;
  int point_length = data_diameter / count * fs + 0.5;
  float d_x = image_width / (N - 1);
  float d_z = image_length / (N - 1);

  int middot =
      -160;    //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点

  int image_x_id = blockIdx.y;    //线
  int image_z_id = blockIdx.x;    //点
  int image_z_dim = gridDim.x;
  //blockIdx.x+blockIdx.y * gridDimx.x
  int recv_id = threadIdx.x;    //接收阵元

  __shared__ float cache_image[2 * M];
  __shared__ int cache_point[2 * M];
  int cacheIndex = threadIdx.x;

  if (image_x_id < N && image_z_id < N && recv_id < 2 * M) {
    float u = 0;
    int point_count_1 = 0;
    float z1 = -image_length / 2 + d_z * image_z_id;
    float x = -image_length / 2 + d_x * image_x_id;
    float xg = 0.0014;

    for (int step_offset = 0; step_offset < parallel_emit_sum; step_offset++) {
      int step = global_step + step_offset;
      int j = step - M + recv_id;    //接收阵元
      j = (j + ELE_NO) % ELE_NO;

      float disi =
          sqrtf((dev_ele_coord_x[step] - x) * (dev_ele_coord_x[step] - x) +
                (z1 - dev_ele_coord_y[step]) * (z1 - dev_ele_coord_y[step]));
      float disj = sqrtf((dev_ele_coord_x[j] - x) * (dev_ele_coord_x[j] - x) +
                         (z1 - dev_ele_coord_y[j]) * (z1 - dev_ele_coord_y[j]));
      float ilength = 112.0 / 1000;
      float imagelength = sqrtf(x * x + z1 * z1);
      float angle =
          acosf((ilength * ilength + disi * disi - imagelength * imagelength) /
                2 / ilength / disi);
      if ((disi >= 0.1 * 2 / 3 &&
           (abs(step - j) < 256 || abs(step - j) > 2048 - 256)) ||
          (disi >= 0.1 * 1 / 3 &&
           (abs(step - j) < 200 || abs(step - j) > 2048 - 200)) ||
          (disi >= 0 && (abs(step - j) < 100 || abs(step - j) > 2048 - 100))) {
        int num = (disi + disj) / count * fs + 0.5;

        if (((num + middot + (OD - 1 - 1) / 2) > 100) &&
            ((num + middot + (OD - 1 - 1) / 2) <= point_length) &&
            (angle < PI / 9)) {
          u += trans_sdata[(num + middot + (OD - 1 - 1) / 2) * ELE_NO + j +
                           step_offset * ELE_NO * NSAMPLE] *
               expf(xg * (num - 1));

          point_count_1 += 1;
        }
      }
    }
    cache_image[cacheIndex] = u;
    cache_point[cacheIndex] = point_count_1;

    __syncthreads();
    // sum up
    int step = blockDim.x / 2;
    while (step != 0) {
      if (cacheIndex < step) {
        cache_image[cacheIndex] += cache_image[cacheIndex + step];
        cache_point[cacheIndex] += cache_point[cacheIndex + step];
      }
      __syncthreads();
      step /= 2;
    }

    if (cacheIndex == 0) {
      int pixel_index = image_z_id + image_x_id * image_z_dim;    //线程块的索引
      image_data[pixel_index] = cache_image[0];
      point_count[pixel_index] = cache_point[0];
    }
  }
}

int main() {
  dim3 dim(1, 1);
  calc_func<<<dim, 512>>>(0, 0, 0, 0, 0);
}