#include<stdio.h>
#include<iostream>
#include<string>
__device__ char*getName(){
    return "/";
}

__global__ void dker(){
    printf("#");
    printf("%s",getName());
}


__global__ void dkernel(){
    printf(".");
    // printf("%s",getName());
}


int main(){
    cudaStream_t s1, s2;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    dker<<<64,128,0,s1>>>();
    // cudaDeviceSynchronize();
    dkernel<<<64,128,0,s2>>>();
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("concurrentKernels=%d\n", p.concurrentKernels);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    // cudaDeviceSynchronize();
    // CPU(host) will asynchronously call the GPU(device)
    printf("Here");


    // create streams s1,s2
    // dker_long<<<g,b,0,s1>>>();
    // dkernel_long<<<g,b,0,s2>>>();
    // synchronize streams separately to observe overlap
    // cudaStreamSynchronize(s1);
    // cudaStreamSynchronize(s2);
}