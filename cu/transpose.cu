#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void transpose(float *in, float *out, int R, int C){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < R && col < C){
        out[col * R + row] = in[row * C + col];
    }
}


void print_matrix(float *mat, int R, int C){
    for(int i = 0; i < R; i++){
        for(int j = 0; j < C; j++){
            std::cout << mat[i * C + j] << " \t";
        }
        std::cout << std::endl;
    }
}


int main(){
    float *din, *dout;
    int R=5, C=4;

    size_t size = sizeof(float) * R * C;

    din = (float*)malloc(size);
    dout = (float*)malloc(size);

    for(int i = 0; i < R * C ; i++){
        din[i] = rand() % 100;
    } 
    std::cout << "Original Matrix: " << std::endl;
    print_matrix(din, R, C);

    float *gin, *gout;

    cudaMalloc(&gin, size);
    cudaMalloc(&gout, size);

    cudaMemcpy(gin, din, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((R + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (C + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose<<<blocksPerGrid, threadsPerBlock>>>(gin, gout, R, C);

    cudaMemcpy(dout, gout, size, cudaMemcpyDeviceToHost);
    std::cout << "Transposed Matrix: " << std::endl;
    print_matrix(dout, C, R);

    cudaFree(gin);
    cudaFree(gout);
    free(din);
    free(dout);

    return 0;
}