#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "get_time.h"
#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void vector_add_cpu(float *h_a, float *h_b, float *h_c_cpu, int n)
{
    for (int i = 0; i < n; i++)
    {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
}
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void init_vector(float *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    // h_ -> stands for host => CPU
    // d_ -> stands for device => GPU

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu_1d = (float *)malloc(size);
    h_c_gpu_3d = (float *)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    int nx = 100, ny = 100, nz = 1000; // N=100*100*1000

    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z);
    // Warm up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }
    printf("Benchmarking CPU implementation...");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    printf("Benchmarking GPU 1d implementation...");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        cudaMemset(d_c_1d, 0, size);
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 100.0;
    // Verify 1D results immediately
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4)
        {
            correct_1d = false;
            std::cout << i << "cpu: " << h_c_cpu[i] << "!=" << h_c_gpu_1d[i] << std::endl;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    // Benchmarking GPU 3D implementation
    printf("Benchmarking GPU 3D implementation...");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        cudaMemset(d_c_3d, 0, size);
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 100.0;
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4)
        {
            correct_3d = false;
            std::cout << i << " cpu:" << h_c_cpu[i] << "!=" << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    printf("CPU average time: %f ms\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f ms\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f ms\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D) %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D) %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D) %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);
}
