#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1
#define NUM_BLOCKS 1

__global__ void incrementCounterNonAtomic(int *counter)
{
    int old = *counter;
    int new_value = old + 1;
    *counter = new_value;
}

__global__ void incrementCounterAtomic(int *counter)
{
    int a = atomicAdd(counter, 1);
    // printf("%d\n",a);
    // atomicAdd returns the pre-increment value, and so a -> will have the value before increment, and there will be increment done at the counter memory location
}

int main()
{
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    cudaMalloc((void **)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void **)&d_counterAtomic, sizeof(int));

    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    cudaFree(d_counterAtomic);
    cudaFree(d_counterNonAtomic);
    free(&h_counterAtomic);
    free(&h_counterNonAtomic);
}