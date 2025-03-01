#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix convolution
__global__ void convolution_kernel(float *input, float *output, float *kernel, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputHeight && col < inputWidth) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int inputRow = row - kernelHeight / 2 + ky;
                int inputCol = col - kernelWidth / 2 + kx;

                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                    sum += input[inputRow * inputWidth + inputCol] * kernel[ky * kernelWidth + kx];
                }
            }
        }
        output[row * inputWidth + col] = sum;
    }
}

// Host function to perform matrix convolution using CUDA
void convolution_cuda(float *input, float *output, float *kernel, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    float *d_input, *d_output, *d_kernel;
    int inputSize = inputWidth * inputHeight * sizeof(float);
    int kernelSize = kernelWidth * kernelHeight * sizeof(float);

 
    cudaMalloc((void **)&d_input, inputSize);
    cudaMalloc((void **)&d_output, inputSize);
    cudaMalloc((void **)&d_kernel, kernelSize);

    
    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    // Configure the grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size as needed
    dim3 gridDim((inputWidth + blockDim.x - 1) / blockDim.x, (inputHeight + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    convolution_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, inputWidth, inputHeight, kernelWidth, kernelHeight);

    // Synchronize the device before copying the result back to the host.
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(output, d_output, inputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}


void initializeMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i * width + j] = (float)rand() / RAND_MAX;
        }
    }
}

// Function to initialize a kernel with an average filter
void initializeKernel(float *kernel, int kernelWidth, int kernelHeight) {
    float kernelValue = 1.0f / (kernelWidth * kernelHeight);
    for (int i = 0; i < kernelHeight; i++) {
        for (int j = 0; j < kernelWidth; j++) {
            kernel[i * kernelWidth + j] = kernelValue;
        }
    }
}

// Function to print a portion of a 2D matrix
void printMatrixSection(float *matrix, int width, int height, int rows, int cols) {
    for (int i = 0; i < rows && i < height; i++) {
        for (int j = 0; j < cols && j < width; j++) {
            printf("%8.4f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int inputWidth = 1024; 
    int inputHeight = 1024;
    int kernelWidth = 5; 
    int kernelHeight = 5;

    float *input = (float *)malloc(inputWidth * inputHeight * sizeof(float));
    float *output = (float *)malloc(inputWidth * inputHeight * sizeof(float));
    float *kernel = (float *)malloc(kernelWidth * kernelHeight * sizeof(float));

    
    initializeMatrix(input, inputWidth, inputHeight);
    initializeKernel(kernel, kernelWidth, kernelHeight);

    // Perform convolution
    convolution_cuda(input, output, kernel, inputWidth, inputHeight, kernelWidth, kernelHeight);

    /
    printf("\nResults of the Matrix Convolution (Top Left 10x10):\n");
    printMatrixSection(output, inputWidth, inputHeight, 10, 10);
    printf("\n");

    // Free host memory
    free(input);
    free(output);
    free(kernel);

    return 0;
}