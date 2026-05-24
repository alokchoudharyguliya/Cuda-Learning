#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\"%s\"\n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel1(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= 2.0f;
}

__global__ void kernel2(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] += 1.0f;
}
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData)
{
    printf("Stream callback: Operation completed\n");
}

int main()
{
    const int N = 1000000;

    size_t size = N * sizeof(float);

    float *h_data, *d_data;

    cudaStream_t s1, s2;
    cudaEvent_t e;
    std::cout << e << std::endl;

    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    for (int i = 0; i < N; i++)
    {
        h_data[i] = static_cast<float>(i);
    }

    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, greatestPriority));

    CHECK_CUDA_ERROR(cudaEventCreate(&e));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, s1));
    kernel1<<<(N + 255) / 256, 256, 0, s1>>>(d_data, N);

    CHECK_CUDA_ERROR(cudaEventRecord(e, s1));

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(s2, e, 0));

    kernel2<<<(N + 255) / 256, 256, 0, s2>>>(d_data, N);

    CHECK_CUDA_ERROR(cudaStreamAddCallback(s2, myStreamCallback, NULL, 0));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, s2));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(s1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s2));

    for (int i = 0; i < N; i++)
    {
        float expected = (static_cast<float>(i) * 2.0f + 1.0f);
        if (fabs(h_data[i] - expected) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(s1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(s2));
    CHECK_CUDA_ERROR(cudaEventDestroy(e));
}