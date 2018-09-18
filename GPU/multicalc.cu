#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
// #include <math.h>
#include <time.h>
#include <cstring>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define PI 3.14159265358979323846
#define N 2048
#define M 256 //接收阵元到发射阵元的最大距离（阵元个数），所以接收孔径为2*M+1
#define ELE_NO 2048
#define OD 64
#define NSAMPLE 3750

int parallel_emit_sum = 1;

/* struct CONST_VALUE
{
    float sample_frequency_div_sound_speed;
    float image_width;
    float image_length;
    float data_diameter;
    int point_length;
    float d_x;
    float d_z;
    int middot; //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点
                       //const int ELE_NO=1024;
};

__constant__ CONST_VALUE const_value; */

//const int sample_point_num=5000;

__device__ float dev_ele_coord_x[ELE_NO];
__device__ float dev_ele_coord_y[ELE_NO]; //写到纹理内存里面
__device__ float dev_filter_data[OD];     //filter parameter

// short data_samples_in_process[NSAMPLE * ELE_NO * 2] = {0}; //写到页锁定主机内存
float image_data[N * N] = {0};
int image_point_count[N * N] = {0};

__global__ void kernel3(float *filtered_data, short *data_in_process)
{
    int column_id = blockDim.x * blockIdx.x + threadIdx.x;
    short data[NSAMPLE];
    float filter_temp_data[NSAMPLE];

    if (column_id < gridDim.x * blockDim.x) // 没有意义，但是不能删除
    {
        memset(filter_temp_data, 0, NSAMPLE * sizeof(float));
        for (int sample_cnt = 0; sample_cnt < NSAMPLE; sample_cnt++)
        {
            data[sample_cnt] = data_in_process[sample_cnt * ELE_NO + column_id];
            for (int j = 0; sample_cnt >= j && j < OD; j++)

            {
                filter_temp_data[sample_cnt] += (dev_filter_data[j] * data[sample_cnt - j]);
            }
        }

        for (int i = 0; i < NSAMPLE; i++)
        {
            filtered_data[i * ELE_NO + column_id] = filter_temp_data[i];
        }
    }
}

// 滤波函数
__global__ void filter_func(float *filtered_data, short *data_in_process)
{
    int column_id = blockDim.x * blockIdx.x + threadIdx.x;
    for (int sample_cnt = 0; sample_cnt < NSAMPLE; sample_cnt++)
    {
        for (int j = 0; sample_cnt >= j && j < OD; j++)
        {
            filtered_data[(column_id / 2048) * ELE_NO * NSAMPLE + sample_cnt * ELE_NO + column_id % 2048] += (dev_filter_data[j] * data_in_process[(sample_cnt - j) * ELE_NO + (column_id / 2048) * ELE_NO * NSAMPLE + column_id % 2048]);
        }
    }
}

// DOG: consider add const to trans_data, 
// DOG: replace "int i" with "const int global_i" to avoid similiar bug in the following
__global__ void kernel(int i, float *image_data, int *point_count, float *trans_sdata, int parallel_emit_sum)
{
    int c = 1520;
    // DOG: WARING: 2.5e6 or just 25 * 1e6? Using the second(means no modification).
    float fs = 25e6;
    float image_width = 200.0 / 1000;
    float image_length = 200.0 / 1000;
    float data_diameter = 220.0 / 1000;
    int point_length = data_diameter / c * fs + 0.5;

    const int no_lines = N;
    const int no_point = N;

    float d_x = image_width / (no_lines - 1);
    float d_z = image_length / (no_point - 1);

    int middot = -160; //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点
                       //const int ELE_NO=1024;

    int k_line = blockIdx.y;                       //线
    int nn = blockIdx.x;                           //点
                                                   //blockIdx.x+blockIdx.y * gridDimx.x
    int y = threadIdx.x;                           //接收阵元
    // DOG: j is never used; removing
    int j = i - M + y;                             //接收阵元
    int bid = blockIdx.x + blockIdx.y * gridDim.x; //线程块的索引
    // int tid = blockDim.x * bid + threadIkdx.x;      //线程的索引

    __shared__ float cache_image[2 * M];
    __shared__ int cache_point[2 * M];
    /*__shared__ float cache_image2[512];*/
    int cacheIndex = threadIdx.x;

    if (k_line < N && nn < N && y < 2 * M)
    {
        float u = 0;
        int point_count_1 = 0;
        // DOG: trigger bug when image_length != image_width
        // DOG: use x = -image_width instead
        float z1 = -image_length / 2 + d_z * nn;
        float x = -image_length / 2 + d_x * k_line;
        float xg = 0.0014;

        // for(int jj=1;jj<=ELE_NO/M;jj++)
        // {
        //  int j=y*ELE_NO/M+jj;
        for (int jj = 0; jj < parallel_emit_sum; jj++)
        {
            // DOG: trigger bug when parallel_emit_sum >= 2
            // DOG: use int i = ::i + jj instead
            // DOG: or int i = global_i + jj; if parameter replaced
            i = i + jj;
            int j = i - M + y; //接收阵元
            j = (j + ELE_NO) % ELE_NO;
            // DOG: ugly name pair <dev_ele_coord_y, z1>
            // DOG: put off the renaming
            float disi = sqrtf((dev_ele_coord_x[i] - x) * (dev_ele_coord_x[i] - x) + (z1 - dev_ele_coord_y[i]) * (z1 - dev_ele_coord_y[i]));
            float disj = sqrtf((dev_ele_coord_x[j] - x) * (dev_ele_coord_x[j] - x) + (z1 - dev_ele_coord_y[j]) * (z1 - dev_ele_coord_y[j]));
            float ilength = 112.0 / 1000;
            float imagelength = sqrtf(x * x + z1 * z1);
            float angle = acosf((ilength * ilength + disi * disi - imagelength * imagelength) / 2 / ilength / disi);
            if ((disi >= 0.1 * 2 / 3 && (abs(i - j) < 256 || abs(i - j) > 2048 - 256)) || (disi >= 0.1 * 1 / 3 && (abs(i - j) < 200 || abs(i - j) > 2048 - 200)) || (disi >= 0 && (abs(i - j) < 100 || abs(i - j) > 2048 - 100)))
            {
                int num = (disi + disj) / c * fs + 0.5;

                if (((num + middot + (OD - 1 - 1) / 2) > 100) && ((num + middot + (OD - 1 - 1) / 2) <= point_length) && (angle < PI / 9))
                {
                    u += trans_sdata[(num + middot + (OD - 1 - 1) / 2) * ELE_NO + j] * expf(xg * (num - 1));

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
        while (i != 0)
        {
            if (cacheIndex < i)
            {
                cache_image[cacheIndex] += cache_image[cacheIndex + i];
                cache_point[cacheIndex] += cache_point[cacheIndex + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (cacheIndex == 0)
        {
            image_data[bid] = cache_image[0];
            point_count[bid] = cache_point[0];
        }
    }
}

__global__ void calc_func(int i, float *image_data, int *point_count, float *trans_sdata, int parallel_emit_sum)
{
    int c = 1520;
    float fs = 25e6;
    float image_width = 200.0 / 1000;
    float image_length = 200.0 / 1000;
    float data_diameter = 220.0 / 1000;
    int point_length = data_diameter / c * fs + 0.5;
    float d_x = image_width / (N - 1);
    float d_z = image_length / (N - 1);

    int middot = -160; //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点
                       //const int ELE_NO=1024;

    int k_line = blockIdx.y;                       //线
    int nn = blockIdx.x;                           //点
                                                   //blockIdx.x+blockIdx.y * gridDimx.x
    int y = threadIdx.x;                           //接收阵元
    int j = i - M + y;                             //接收阵元
    int bid = blockIdx.x + blockIdx.y * gridDim.x; //线程块的索引
    // int tid = blockDim.x * bid + threadIkdx.x;      //线程的索引

    __shared__ float cache_image[2 * M];
    __shared__ int cache_point[2 * M];
    /*__shared__ float cache_image2[512];*/
    int cacheIndex = threadIdx.x;

    if (k_line < N && nn < N && y < 2 * M)
    {
        float u = 0;
        int point_count_1 = 0;
        float z1 = -image_length / 2 + d_z * nn;
        float x = -image_length / 2 + d_x * k_line;
        float xg = 0.0014;

        // for(int jj=1;jj<=ELE_NO/M;jj++)
        // {
        //  int j=y*ELE_NO/M+jj;
        for (int jj = 0; jj < parallel_emit_sum; jj++)
        {
            i = i + jj;
            int j = i - M + y; //接收阵元
            j = (j + ELE_NO) % ELE_NO;

            float disi = sqrtf((dev_ele_coord_x[i] - x) * (dev_ele_coord_x[i] - x) + (z1 - dev_ele_coord_y[i]) * (z1 - dev_ele_coord_y[i]));
            float disj = sqrtf((dev_ele_coord_x[j] - x) * (dev_ele_coord_x[j] - x) + (z1 - dev_ele_coord_y[j]) * (z1 - dev_ele_coord_y[j]));
            float ilength = 112.0 / 1000;
            float imagelength = sqrtf(x * x + z1 * z1);
            float angle = acosf((ilength * ilength + disi * disi - imagelength * imagelength) / 2 / ilength / disi);
            if ((disi >= 0.1 * 2 / 3 && (abs(i - j) < 256 || abs(i - j) > 2048 - 256)) || (disi >= 0.1 * 1 / 3 && (abs(i - j) < 200 || abs(i - j) > 2048 - 200)) || (disi >= 0 && (abs(i - j) < 100 || abs(i - j) > 2048 - 100)))
            {
                int num = (disi + disj) / c * fs + 0.5;

                if (((num + middot + (OD - 1 - 1) / 2) > 100) && ((num + middot + (OD - 1 - 1) / 2) <= point_length) && (angle < PI / 9))
                {
                    u += trans_sdata[(num + middot + (OD - 1 - 1) / 2) * ELE_NO + j + jj * ELE_NO * NSAMPLE] * expf(xg * (num - 1));

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
        while (i != 0)
        {
            if (cacheIndex < i)
            {
                cache_image[cacheIndex] += cache_image[cacheIndex + i];
                cache_point[cacheIndex] += cache_point[cacheIndex + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (cacheIndex == 0)
        {
            image_data[bid] = cache_image[0];
            point_count[bid] = cache_point[0];
        }
    }
}

__global__ void add(float *sumdata, int *sumpoint, float *imagedata, int *point_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N * N)
    {
        sumdata[tid] += imagedata[tid];
        sumpoint[tid] += point_count[tid];
        tid += blockDim.x * gridDim.x;
    }
}

cudaError_t precalcWithCuda(short *dev_data_samples_in_process, int ele_emit_id, float *dev_sumdata, int *dev_sumpoint, float *dev_filterdata, float *dev_imagedata, int *dev_pointcount, int parallel_emit_sum)
{
    cudaError_t cudaStatus;

    //kernel 1,kernel2 decode
    //kernel3 filter
    cudaMemset(dev_filterdata, 0, NSAMPLE * ELE_NO * sizeof(short) * parallel_emit_sum * 2);
    filter_func<<<4 * parallel_emit_sum, 512>>>(dev_filterdata, dev_data_samples_in_process);
    // cudaStatus = cudaGetLastError();
    // if (cudaStatus != cudaSuccess)
    // {
    //     cout << "filter_func launch failed: " << cudaGetErrorString(cudaStatus);
    //     //goto Error;
    //     return cudaStatus;
    // }

    // cudaStatus = cudaDeviceSynchronize();

    dim3 gridimage(N, N);
    //dim3 threads(M);
    calc_func<<<gridimage, 2 * M>>>(ele_emit_id, dev_imagedata, dev_pointcount, dev_filterdata, parallel_emit_sum); //启动一个二维的N*N个block，每个block里面M个thread

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "calcKernel launch failed: " << cudaGetErrorString(cudaStatus);
        //goto Error;
        return cudaStatus;
    }
    // cudaDeviceSynchronize();

    //把所有的结果加到一起
    add<<<32, 32>>>(dev_sumdata, dev_sumpoint, dev_imagedata, dev_pointcount);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus);
        //goto Error;
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    // cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess)
    // {
    //     fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //     return cudaStatus;
    // }

    return cudaStatus;
}

void get_ele_position(float *ele_coord_x, float *ele_coord_y)
{
    float rfocus = (float)112 / 1000;
    float ele_angle = (2 * PI * 43.4695 / (256 - 1)) / 360; //阵元间隔角度
    float first_one = 2 * PI * (45 - 43.4695) / 360;        //第一个阵元角度

    for (int i = 0; i < 256; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + i * ele_angle);
        ele_coord_y[i] = -rfocus * sin(first_one + i * ele_angle);
    }
    for (int i = 256; i < 512; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 256) * ele_angle + PI / 4);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 256) * ele_angle + PI / 4);
    }
    for (int i = 512; i < 768; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 512) * ele_angle + PI / 2);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 512) * ele_angle + PI / 2);
    }
    for (int i = 768; i < 1024; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 768) * ele_angle + 3 * PI / 4);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 768) * ele_angle + 3 * PI / 4);
    }
    for (int i = 1024; i < 1280; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 1024) * ele_angle + PI);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 1024) * ele_angle + PI);
    }
    for (int i = 1280; i < 1536; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 1280) * ele_angle + 5 * PI / 4);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 1280) * ele_angle + 5 * PI / 4);
    }
    for (int i = 1536; i < 1792; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 1536) * ele_angle + 3 * PI / 2);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 1536) * ele_angle + 3 * PI / 2);
    }
    for (int i = 1792; i < 2048; i++)
    {
        ele_coord_x[i] = rfocus * cos(first_one + (i - 1792) * ele_angle + 7 * PI / 4);
        ele_coord_y[i] = -rfocus * sin(first_one + (i - 1792) * ele_angle + 7 * PI / 4);
    }
}

void write_txtfile(std::string output_path)
{
    ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        cout << " the file open fail" << endl;
        exit(1);
    }

    for (int k = 0; k < N; k++)
    {
        for (int j = 0; j < N; j++)
        {
            if (image_point_count[k * N + j] == 0)
                outfile << image_data[k * N + j] << " ";
            else
                outfile << image_data[k * N + j] / image_point_count[k * N + j] << " ";
        }
        outfile << "\r\n";
    }

    outfile.close();
}

int main(int argc, char const *argv[])
{
    time_t start, over;
    start = time(NULL);

    /* float sound_speed = 1520;
    float sample_frequency = 25e6;
    struct CONST_VALUE temp;
    temp.sample_frequency_div_sound_speed = sample_frequency/sound_speed;
    temp.image_width = 200.0 / 1000;
    temp.image_length = 200.0 / 1000;
    temp.data_diameter = 220.0 / 1000;
    temp.point_length = temp.data_diameter * temp.sample_frequency_div_sound_speed + 0.5;
    temp.d_x = temp.image_width / (N - 1);
    temp.d_z = temp.image_length / (N - 1);
    temp.middot = -160; //发射前1us开始接收，也就是约为12.5个点之后发射,数据显示约16个点
                       //const int ELE_NO=1024;

    if (cudaMemcpyToSymbol(&const_value, &temp, sizeof(temp)) != cudaSuccess)
    {
        cout << "ERROR :: struct CONST_VALUE copy failed." << endl;
    } */

    std::string filter_path = "";
    std::string bin_path = "";
    std::string output_path = "";
    switch (argc)
    {
    case 4:
        parallel_emit_sum = atoi(argv[1]);
        filter_path = argv[2];
        bin_path = argv[3];
        output_path = "origin.txt";
        break;
    case 5:
        parallel_emit_sum = atoi(argv[1]);
        filter_path = argv[2];
        bin_path = argv[3];
        output_path = argv[4];
        break;
    default:
        std::cout << "Please input 3 or 4 paras" << std::endl;
        std::cout << "[parallel emit sum] [filter path] [bin path]" << std::endl;
        std::cout << "[parallel emit sum] [filter path] [bin path] [output path]" << std::endl;
        exit(-1);
        break;
    }

    cudaError_t cudaStatus;

    time_t start_read, over_read;
    start_read = time(NULL);
    // Read filter data and put in GPU
    ifstream file_read;
    file_read.open(filter_path.c_str(), ios_base::in | ios::binary);
    if (!file_read.is_open())
    {
        cout << " the file filter open fail" << endl;
        return -1;
    }
    float filter_data[OD];
    for (int ii = 0; ii < OD; ii++)
    {
        file_read.read((char *)&filter_data[ii], sizeof(float));
    }
    file_read.close();
    cudaStatus = cudaMemcpyToSymbol(dev_filter_data, filter_data, sizeof(float) * OD);

    if (cudaStatus != cudaSuccess)
    {
        cout << "center Fail to cudaMemcpyToSymbol on GPU" << endl;
        return;
    }

    file_read.open(bin_path.c_str(), ios_base::in | ios::binary | ios::ate);
    if (!file_read.is_open())
    {
        cout << " the bin file open fail" << endl;
        return -1;
    }
    long long int filesize = file_read.tellg();
    file_read.seekg(0, file_read.beg);
    // 为 bin_buffer 申请空间，并把 filepath 的数据载入内存
    char *bin_buffer = (char *)std::malloc(filesize);
    if (bin_buffer == NULL)
    {
        std::cout << "ERROR :: Malloc data for buffer failed." << std::endl;
        return 0;
    }
    file_read.read(bin_buffer, filesize);
    if (file_read.peek() == EOF)
    {
        file_read.close();
    }
    else
    {
        std::cout << "ERROR :: Read bin file error." << std::endl;
        exit(-1);
    }
    over_read = time(NULL);
    cout << "Reading time is : " << difftime(over_read, start_read) << "s!" << endl;

    //image grid
    float ele_coord_x[ELE_NO] = {0};
    float ele_coord_y[ELE_NO] = {0};
    get_ele_position(&ele_coord_x[0], &ele_coord_y[0]);

    if (cudaMemcpyToSymbol(dev_ele_coord_x, ele_coord_x, sizeof(float) * ELE_NO) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMemcpyToSymbol dev_ele_coord_x." << endl;
        return -1;
    }

    if (cudaMemcpyToSymbol(dev_ele_coord_y, ele_coord_y, sizeof(float) * ELE_NO) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMemcpyToSymbol dev_ele_coord_y." << endl;
        return -1;
    }

    float *dev_sumdata;
    int *dev_sumpoint;
    if (cudaMalloc((void **)(&dev_sumdata), N * N * sizeof(float)) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMalloc dev_sumdata." << endl;
        return -1;
    }
    if (cudaMalloc((void **)(&dev_sumpoint), N * N * sizeof(int)) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMalloc dev_sumpoint." << endl;
        return -1;
    }
    // init dev_sumdata and dev_sumpoint
    if (cudaMemcpy(dev_sumdata, image_data, N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMemcpy dev_sumdata." << endl;
        return -1;
    }
    if (cudaMemcpy(dev_sumpoint, image_point_count, N * N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << "ERROR :: Failed for cudaMemcpy dev_sumpoint." << endl;
        return -1;
    }

    long long length_of_data_in_process = NSAMPLE * ELE_NO * sizeof(short) * parallel_emit_sum;
    short *dev_data_samples_in_process;
    float *dev_filterdata;

    cudaStatus = cudaMalloc((void **)(&dev_data_samples_in_process), length_of_data_in_process);
    if (cudaStatus != cudaSuccess)
    {
        cout << "data_samples_in_process Fail to cudaMalloc on GPU" << endl;
        return -1;
    }

    if (cudaMalloc((void **)(&dev_filterdata), length_of_data_in_process * 2) != cudaSuccess) // 转 float 乘以 2
    {
        cout << "ERROR :: Failed for cudaMalloc dev_filterdata." << endl;
        return -1;
    }

    float *dev_imagedata;
    // float *dev_trans_sdata;

    int *dev_pointcount;
    //  int dev_i;

    cudaStatus = cudaMalloc((void **)(&dev_imagedata), N * N * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cout << "imagedata Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void **)(&dev_pointcount), N * N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        cout << "pointcount Fail to cudaMalloc on GPU" << endl;
        //goto Error;
        return cudaStatus;
    }

    long long bin_buffer_index = 0;
    for (int ele_emit_id = 0; ele_emit_id < ELE_NO; ele_emit_id += parallel_emit_sum)
    //for (i=1;i<=1;i++)
    {
        printf("Number of element : %d\n", ele_emit_id);

        // memcpy(&data_samples_in_process[0], &bin_buffer[bin_buffer_index], length_of_data_in_process);
        // bin_buffer_index = bin_buffer_index + length_of_data_in_process;

        // cudaStatus = cudaMemcpy(dev_data_samples_in_process, data_samples_in_process, length_of_data_in_process, cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_data_samples_in_process, &bin_buffer[bin_buffer_index], length_of_data_in_process, cudaMemcpyHostToDevice);
        bin_buffer_index = bin_buffer_index + length_of_data_in_process;
        if (cudaStatus != cudaSuccess)
        {
            cout << "data_samples_in_process Fail to cudaMemcpy on GPU" << endl;
            //goto Error;
            return cudaStatus;
        }
        cudaStatus = precalcWithCuda(dev_data_samples_in_process, ele_emit_id, dev_sumdata, dev_sumpoint, dev_filterdata, dev_imagedata, dev_pointcount, parallel_emit_sum);
        //}
        // over=time(NULL);
        // cout<<"Running time is : "<<difftime(over,start)<<"s!"<<endl;
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "calcWithCuda failed!");
            return 1;
        }
        // cudaError_t cudaStatus = calcWithCuda( i,dev_sumdata,dev_sumpoint,dev_filterdata);
    }
    cudaStatus = cudaMemcpy(image_data, dev_sumdata, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        cout << "allimagedata Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    }

    cudaStatus = cudaMemcpy(image_point_count, dev_sumpoint, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        cout << "allpointcount Fail to cudaMemcpy to CPU" << endl;
        return 1;
        //goto Error;
    }

    write_txtfile(output_path);
    over = time(NULL);
    cout << "Running time is : " << difftime(over, start) / 60 << "min!" << endl;
    cudaFree(dev_sumdata);
    cudaFree(dev_sumpoint);
    cudaFree(dev_data_samples_in_process);
    cudaFree(dev_filterdata);
    cudaFree(dev_imagedata);
    cudaFree(dev_pointcount);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
}

//cudaStatus= cudaMemcpy( data_in_process,dev_filterdata , 5000*1024 * sizeof(float),cudaMemcpyDeviceToHost ) ;
// if (cudaStatus != cudaSuccess) {
//cout<<"data_output Fail to cudaMemcpy to CPU"<<endl;
// goto Error;
//goto Error;
// }
