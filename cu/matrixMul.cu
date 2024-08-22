#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>

__global__ void matrixMul(float *a, float *b, float *c, int R, int C, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < R && col < K){
        float sum = 0.0f;
        for (int k = 0; k < C; k++){
            sum += a[row * C + k] * b[col * C + k]; 
        }
        c[row * K + col] = sum;
    }
}

int main(){
    int R = 10000,  C = 1000, K = 10;
    size_t a = R * C * sizeof(float);
    size_t b = C * K * sizeof(float);
    size_t c = R * K * sizeof(float);

    float *ha = (float *)malloc(a);
    float *hb = (float *)malloc(b);
    float *hc = (float *)malloc(c);

    for(int i=0; i < R * C; i++){
        ha[i] = 1.0f;
        
    }
    for(int i = 0; i < C * K; i++){
        hb[i] = 1.0f;
    }

    float *da, *db, *dc;
    cudaMalloc(&da, a);
    cudaMalloc(&db, b);
    cudaMalloc(&dc, c);

    cudaMemcpy(da, ha, a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, b, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (R + threadsPerBlock.y - 1) / threadsPerBlock.y);


    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel call
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, R, C, K);
    cudaMemcpy(hc, dc, c, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "GPU Elapsed time: " << elapsedTime << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bool success = true;

    auto start_c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < C; ++k) {
                sum += ha[i * C + k] * hb[j * C + k];
            }
            if (sum != hc[i * K + j]){
                std::cerr << "Error at Index: " << i * K + j << std::endl;
                success = false;
                break;
            }
        }
        if (!success){
            break;
        }
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