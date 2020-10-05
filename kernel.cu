
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 16 // submatrix size
#define N 1024 // matrix size is N*N

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
    // Координаты блока в матрице (номер блока в строке и в столбце)
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Каждый блок нитей вычисляет одну подматрицу Csub, при этом
    // каждая нить создаёт свой собственный дескриптор матрицы Csub
    Matrix cc;
    cc.elements = c;
    Matrix aa;
    aa.elements = a;
    Matrix bb;
    bb.elements = b;

    Matrix Csub = GetSubMatrix(cc, blockRow, blockCol,n);
    // Каждая нить вычисляет один элемент подматрицы Csub
    float Cvalue = 0;
    // thread row and col WITHIN CSUB
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Цикл по всем подматрицам строки блоков A и столбца блоков B;
    // этот цикл необходим для вычисления всей подматрицы Csub.
    // Блочное умножение пары подматриц и аккумуляция результата
    for (int m = 0; m < (N / BLOCK_SIZE); ++m) {
        // Формирование дескрипторов подматриц Asub и Bsub
        Matrix Asub = GetSubMatrix(aa, blockRow, m,n);
        Matrix Bsub = GetSubMatrix(bb, m, blockCol,n);
        // Копирование элементов ASub и Bsub в разделяемую память
        // Каждая нить загружает один элемент ASub и один – Bsub
        // Обратите внимание: каждая нить объявляет матрицы As и Bs,
        // хотя блок нитей содержит только одну матрицу As и одну Bs
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Нити синхронизируются – чтобы убедиться, что всё прочитано
        __syncthreads();
        // Скалярное произведение одной строки Asub и одного столбца Bsub
        // формируют (частично) один элемент результирующей подматрицы
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Надо убедиться, что все Cvalues просуммированы до начала
        // считывания последующих блоков в разделяемые матрицы As, Bs
        __syncthreads();
    }
    // Запись Csub в глобальную память: каждая нить записывает «свой» элемент
    SetElement(Csub, row, col, Cvalue);
}

//__global__ void matMulKernel(float* a, float* b, float* c, int n) {
    // каждая нить вычисляет один элемент матрицы C
  //  float cvalue = 0;
 //   int row = blockIdx.y * blockDim.y + threadIdx.y;
 //   int col = blockIdx.x * blockDim.x + threadIdx.x;
  //  for (int e = 0; e < n; ++e)
  //      cvalue += a[row * n + e] * b[e * n + col];
  //  c[row * n + col] = cvalue;
//}

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

    // выделение памяти на хосте
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
           // a[k] = (rand())%100;
           // b[k] = (rand()) % 100;
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
    //std::cout << static_cast<float>(time) / CLOCKS_PER_SEC << std::endl;
    //std::cout << c_c[0][2] << std::endl;
    //std::cout << c_c[2][7];
    // выделение памяти на девайсе

    int* adev = NULL;
    int* bdev = NULL;
    int* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    // Установка конфигурации запуска ядра

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    // Создание обработчика событий CUDA

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // асинхронно выдаем работу на GPU (все в поток 0)

    cudaEventRecord(start, 0);
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMulKernel << <blocks, threads >> > (adev, bdev, cdev, N);
    //matMulKernel << <blocks, threads >> > (adev, bdev, cdev,N);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    // Печатаем время работы на GPU и CPU
    //std::cout << c[2] <<" "<<c[3] <<std::endl;
    if (checkResult(c_c, c, N)) {
        printf("Time spent executing by the GPU: %.2f millseconds\n", gpuTime);
        std::cout << "Time spent executing by the CPU: " << static_cast<float>(time) / CLOCKS_PER_SEC << std::endl;

    }
        
    // Освобождение ресурсов

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