#include <stdio.h>

#define ROWS 15
#define COLS 16

__global__ void matrixAdd(int* a, int* b, int* c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    int idx = row * cols + col;                      // 1D index into flattened array
    if (row < rows && col < cols) {                  // Bounds check
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Host matrices (2D, flattened to 1D)
    int h_a[ROWS * COLS];
    int h_b[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) {
        h_a[i] = rand() % 100;
    }

    // Fill array h_b with random integers (e.g., between 0 and 99)
    for (int i = 0; i < ROWS * COLS; i++) {
        h_b[i] = rand() % 100;
    }
    int h_c[ROWS * COLS];                        

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc(&d_a, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_b, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_c, ROWS * COLS * sizeof(int));

    // Copy to GPU
    cudaMemcpy(d_a, h_a, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(2, 2);  // 2x2 threads per block
    dim3 blocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);  // 2x1 blocks

    // Launch kernel
    matrixAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, ROWS, COLS);

    // Copy result back
    cudaMemcpy(h_c, d_c, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print matrices

    printf("Matrix C (A + B):\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", h_c[i * COLS + j]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}