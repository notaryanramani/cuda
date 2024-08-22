#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void addVectors(int *a, int *b, int *c, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        c[idx] = a[idx] + b[idx];
    }
    
}

int main(){
    int n = 1000000;
    size_t size = n * sizeof(int);

    int *ha = (int *)malloc(size);
    int *hb = (int *)malloc(size);
    int *hc = (int *)malloc(size);

    for (int i=0; i<n; i++){
        ha[i] = i;
        hb[i] = i * 2;
    }

    int *da, *db, *dc;
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaError_t err =  cudaMalloc(&dc, size);
    if (err != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel call
    addVectors<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "GPU Elapsed time: " << elapsedTime << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    auto start_c = std::chrono::high_resolution_clock::now();
    int a;
    for (int i = 0; i < n; i++) {
        a = ha[i] + hb[i];
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_c - start_c;
    std::cout << "CPU execution time: " << duration.count() * 1000 << " ms" << std::endl;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    free(hc);

    return 0;
}