#include <iostream>

__global__ void helloCUDA() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch the kernel function with 1 block and 1 thread
    helloCUDA<<<1, 1>>>();

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello, World from CPU!" << std::endl;

    return 0;
}
