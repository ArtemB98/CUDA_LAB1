
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 32 // submatrix size
#define N 256 // matrix size is N*N

typedef struct {
    int n;
    int* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * N + col];
}

__device__ void SetElement(Matrix A, int row, int col, int value) {
    A.elements[row * N + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int n) {
    Matrix ASub;
    ASub.n = BLOCK_SIZE;
    ASub.elements = &A.elements[n * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return ASub;
}

__global__ void matMulKernel(int* a, int* b, int* c, int n) {

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix cc;
    cc.elements = c;
    Matrix aa;
    aa.elements = a;
    Matrix bb;
    bb.elements = b;

    Matrix Csub = GetSubMatrix(cc, blockRow, blockCol,n);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (N / BLOCK_SIZE); ++m) {

        Matrix Asub = GetSubMatrix(aa, blockRow, m,n);
        Matrix Bsub = GetSubMatrix(bb, m, blockCol,n);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}

int** matMulCPU(int** a, int** b, int** c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            c[i][j] = 0;
            for (int k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
    return c;
}

bool checkResult(int** a, int* b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (a[i][j] != b[N * i + j])
                return false;
    return true;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));

    int numBytes = N * N * sizeof(int);

    int** a_c;
    int** b_c;
    int** c_c;
    a_c = new int* [N];
    for (int i = 0; i < N; i++)
        a_c[i] = new int[N];
    
    b_c = new int* [N];
    for (int i = 0; i < N; i++)
        b_c[i] = new int[N];
    
    c_c = new int* [N];
    for (int i = 0; i < N; i++)
        c_c[i] = new int[N];

    int* a = new int[N * N];
    int* b = new int[N * N];
    int* c = new int[N * N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int k = N * i + j;
            a[k] = (rand()) % 100;
            b[k] = (rand()) % 100;
            a_c[i][j] = a[k];
            b_c[i][j] = b[k];
            c_c[i][j] = 0;
        }

    clock_t time;
    time = clock();
    c_c=matMulCPU(a_c, b_c, c_c, N);
    time = clock() - time;

    int* adev = NULL;
    int* bdev = NULL;
    int* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMulKernel << <blocks, threads >> > (adev, bdev, cdev, N);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    if (checkResult(c_c, c, N)) {
        printf("Time spent executing by the GPU: %.2f millseconds\n", gpuTime);
        std::cout << "Time spent executing by the CPU: " << time * 1000.0 / CLOCKS_PER_SEC << " millseconds" << std::endl;

    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);

    delete a;
    delete b;
    delete c;
    for (int i = 0; i < N; i++)
    {
        delete b_c[i];
        delete a_c[i];
        delete c_c[i];
    }
    delete[]a_c;
    delete[]b_c;
    delete[]c_c;

    return 0;

}