#include <stdio.h>

//GPU Kernel function
__global__ void sayHello() {
    int threadId = threadIdx.x;  
    printf("Hello from your NVIDIA GPU-thread %d!\n", threadId);
}

int main() {
    
    sayHello<<<1, 5>>>();

    
    cudaDeviceSynchronize();

    printf("All done from the CPU!\n");
    return 0;
}