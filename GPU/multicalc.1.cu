#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
// #include <math.h>
#include <time.h>
#include <cstring>
#include <random>

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

// short data_samples_in_process[NSAMPLE * ELE_NO * 2] = {0}; //写到页锁定主机内存
float image_data[N * N] = {0};
int image_point_count[N * N] = {0};

// 滤波函数

__global__ void calc_func(int i, float *image_data, int *point_count,
                          const float *trans_sdata, int parallel_emit_sum) {
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
               //const int ELE_NO=1024;

  int image_y_id = blockIdx.y;    //线
  int image_x_id = blockIdx.x;        //点
                              //blockIdx.x+blockIdx.y * gridDimx.x
  int y = threadIdx.x;        //接收阵元
  int pixel_index = blockIdx.x + blockIdx.y * gridDim.x;    //线程块的索引
  // int tid = blockDim.x * pixel_index + threadIkdx.x;            //线程的索引

  __shared__ float cache_image[2 * M];
  __shared__ int cache_point[2 * M];
  /*__shared__ float cache_image2[512];*/
  int cacheIndex = threadIdx.x;

  if (image_y_id < N && image_x_id < N && y < 2 * M) {
    float u = 0;
    int point_count_1 = 0;
    float z1 = -image_length / 2 + d_z * image_x_id;
    float x = -image_length / 2 + d_x * image_y_id;
    float xg = 0.0014;

    // for(int jj=1;jj<=ELE_NO/M;jj++)
    // {
    //    int j=y*ELE_NO/M+jj;
    for (int jj = 0; jj < parallel_emit_sum; jj++) {
      i = i + jj;
      int j = i - M + y;    //接收阵元
      j = (j + ELE_NO) % ELE_NO;

      float disi = sqrtf((dev_ele_coord_x[i] - x) * (dev_ele_coord_x[i] - x) +
                         (z1 - dev_ele_coord_y[i]) * (z1 - dev_ele_coord_y[i]));
      float disj = sqrtf((dev_ele_coord_x[j] - x) * (dev_ele_coord_x[j] - x) +
                         (z1 - dev_ele_coord_y[j]) * (z1 - dev_ele_coord_y[j]));
      float ilength = 112.0 / 1000;
      float imagelength = sqrtf(x * x + z1 * z1);
      float angle =
          acosf((ilength * ilength + disi * disi - imagelength * imagelength) /
                2 / ilength / disi);
      if ((disi >= 0.1 * 2 / 3 &&
           (abs(i - j) < 256 || abs(i - j) > 2048 - 256)) ||
          (disi >= 0.1 * 1 / 3 &&
           (abs(i - j) < 200 || abs(i - j) > 2048 - 200)) ||
          (disi >= 0 && (abs(i - j) < 100 || abs(i - j) > 2048 - 100))) {
        int num = (disi + disj) / count * fs + 0.5;

        if (((num + middot + (OD - 1 - 1) / 2) > 100) &&
            ((num + middot + (OD - 1 - 1) / 2) <= point_length) &&
            (angle < PI / 9)) {
          u += trans_sdata[(num + middot + (OD - 1 - 1) / 2) * ELE_NO + j +
                           jj * ELE_NO * NSAMPLE] *
               expf(xg * (num - 1));

          point_count_1 += 1;
        }
      }
    }
    cache_image[cacheIndex] = u;
    cache_point[cacheIndex] = point_count_1;

    __syncthreads();

    /*int jj=1;
                while((cacheIndex + jj)< blockDim.x){
                cache_image2[cacheIndex]+=cache_image[cacheIndex]*cache_image[cacheIndex + jj];
                 __syncthreads();
                                 jj= jj+1;
                }*/

    int i = blockDim.x / 2;
    while (i != 0) {
      if (cacheIndex < i) {
        cache_image[cacheIndex] += cache_image[cacheIndex + i];
        cache_point[cacheIndex] += cache_point[cacheIndex + i];
      }
      __syncthreads();
      i /= 2;
    }

    if (cacheIndex == 0) {
      image_data[pixel_index] = cache_image[0];
      point_count[pixel_index] = cache_point[0];
    }
  }
}


int main(){
    dim3 dim(1, 1); 
    calc_func<<<dim, 512>>>(0,0,0,0,0);
}