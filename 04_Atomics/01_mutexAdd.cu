#include <cuda_runtime.h>
#include <stdio.h>

struct Mutex
{
    int *lock;
};

// __host__ void initMutex(Mutex *m) -> we can omit the __host__ keyword
__host__ void initMutex(Mutex *m)
{
    cudaMalloc((void **)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

__device__ void lock(Mutex *m)
{
    while (atomicCAS(m->lock, 0, 1) != 0)
    {
    }
}

__device__ void unlock(Mutex *m)
{
    atomicExch(m->lock, 0);
}

__global__ void mutexKernel(int *counter, Mutex *m)
{
    lock(m);
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main()
{
    Mutex m;
    initMutex(&m);

    int *d_counter;
    cudaMalloc((void **)&d_counter, sizeof(int));

    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    Mutex*d_m;
    cudaMalloc(&d_m,sizeof(Mutex));
    cudaMemcpy(d_m,&m,sizeof(Mutex),cudaMemcpyHostToDevice);
    mutexKernel<<<1, 1000>>>(d_counter, d_m);
    int res;
    cudaMemcpy(&res, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Counter values: %d\n", res);

    cudaFree(m.lock);
    cudaFree(d_counter);
    cudaFree(d_m);
}