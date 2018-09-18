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

inline __device__ float distance(float x1, float y1, float x2, float y2) {
    auto dx = x1 - x2;
    auto dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

bool __device__ __host__ is_close(int delta, int range) {
    int abs_delta = abs(delta);
    return (abs_delta < range || range > 2048 - range);
    // return (delta + range + 2047) % 2048 < 2 * range - 1;
}

__global__ void calc_func(const int global_step, float *image_data,
                          int *point_count, const float *trans_sdata,
                          const int parallel_emit_sum) {
    int sound_speed = 1520;
    float fs = 25e6;
    float image_width = 200.0 / 1000;
    float image_length = 200.0 / 1000;
    float data_diameter = 220.0 / 1000;
    // 3618
    int point_length = data_diameter / sound_speed * fs + 0.5;
    float d_x = image_width / (N - 1);
    float d_z = image_length / (N - 1);
    // magic code...
    int middot =
        -160;    //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点

    int image_x_id = blockIdx.y;    //线
    int image_z_id = blockIdx.x;    //点
    int image_z_dim = gridDim.x;
    int recv_center_id = threadIdx.x;    //center of 接收阵元

    __shared__ float cache_image[2 * M];
    __shared__ int cache_point[2 * M];
    int cacheIndex = threadIdx.x;

    if (image_x_id < N && image_z_id < N && recv_center_id < 2 * M) {
        float sum_image = 0;
        int sum_point = 0;
        float value_z = -image_length / 2 + d_z * image_z_id;
        float value_x = -image_length / 2 + d_x * image_x_id;
        // what the hell is this !!! need more comments!!!
        float xg = 0.0014;

        for (int step_offset = 0; step_offset < parallel_emit_sum;
             step_offset++) {
            int step = global_step + step_offset;
            int send_id = step;                            // as send_id
            int recv_id = send_id - M + recv_center_id;    //接收阵元
            recv_id = (recv_id + ELE_NO) % ELE_NO;

            float disi = distance(dev_ele_coord_x[send_id],
                                  dev_ele_coord_y[send_id], value_x, value_z);
            float disj = distance(dev_ele_coord_x[recv_id],
                                  dev_ele_coord_y[recv_id], value_x, value_z);
            // what the hell is this !!! need more comments!!!
            // i guess it is a radius?
            float ilength = 112.0 / 1000;
            float imagelength = sqrtf(value_x * value_x + value_z * value_z);
            // 2 * R * disi * cosTheta = R^2 + disi^2 - |(x, z)|^2
            float angle = acosf(
                (ilength * ilength + disi * disi - imagelength * imagelength) /
                2 / ilength / disi);

            // put disi constraint onto for;
            // and since
            auto diff = send_id - recv_id;
            bool is_valid = (disi >= 0.1 * 2 / 3 && is_close(diff, 256)) ||
                            (disi >= 0.1 * 1 / 3 && is_close(diff, 200)) ||
                            (disi >= 0.1 * 0 / 3 && is_close(diff, 100));
            if (is_valid) {
                int num = (disi + disj) / sound_speed * fs + 0.5;

                if (((num + middot + (OD - 1 - 1) / 2) > 100) &&
                    ((num + middot + (OD - 1 - 1) / 2) <= point_length) &&
                    (angle < PI / 9)) {
                    sum_image +=
                        trans_sdata[(num + middot + (OD - 1 - 1) / 2) * ELE_NO +
                                    recv_id + step_offset * ELE_NO * NSAMPLE] *
                        expf(xg * (num - 1));

                    sum_point += 1;
                }
            }
        }
        cache_image[cacheIndex] = sum_image;
        cache_point[cacheIndex] = sum_point;

        __syncthreads();
        // sum up cache_image and cacheIndex, and i have way to make this part disappear
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
            int pixel_index =
                image_z_id + image_x_id * image_z_dim;    //线程块的索引
            image_data[pixel_index] = cache_image[0];
            point_count[pixel_index] = cache_point[0];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 0) {
        dim3 dim(1, 1);
        calc_func<<<dim, 512>>>(0, 0, 0, 0, 0);
    }
    return 0;
}