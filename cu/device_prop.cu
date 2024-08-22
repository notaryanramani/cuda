#include <iostream>
#include <cuda_runtime.h>

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.maxThreadsPerBlock << std::endl;
    std::cout << prop.maxGridSize[0] << std::endl;
}