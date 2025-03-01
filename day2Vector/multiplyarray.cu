#include <stdio.h>

// CUDA kernel to multiply each element by 2
__global__ void multiplyArray(int* array) {
    int i = threadIdx.x;  // Each thread gets its index
    array[i] = array[i] * 2;  // Multiply by 2
}

int main() {
    const int size = 5;  // Array size
    int h_array[5] = {1, 2, 3, 4, 5};  // Host array
    int* d_array;  // Device array pointer

    // Allocate GPU memory
    cudaMalloc(&d_array, size * sizeof(int));

    // Copy from CPU to GPU
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    multiplyArray<<<1, size>>>(d_array);

    // Copy back to CPU
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Wait and clean up
    cudaDeviceSynchronize();
    cudaFree(d_array);

    // Print result
    printf("Multiplied array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    return 0;
}