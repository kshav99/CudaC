#include <iostream>

// CUDA Kernel (Runs on GPU)
__global__ void add(int *a, int *b, int *c, int size) {
    int index = threadIdx.x;  // Each thread handles one element
    if (index < size)
        c[index] = a[index] + b[index];
}

int main() {
    const int size = 5;
    int h_a[size] = {1, 2, 3, 4, 5};  // Host (CPU) arrays
    int h_b[size] = {10, 20, 30, 40, 50};
    int h_c[size];  // To store result

    int *d_a, *d_b, *d_c;  // Device (GPU) pointers

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel with `size` threads
    add<<<1, size>>>(d_a, d_b, d_c, size);

    // Copy result back to CPU
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < size; i++)
        std::cout << h_c[i] << " ";
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
