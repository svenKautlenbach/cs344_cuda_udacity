#include <iostream>
#include <stdio.h>

#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>

#include "gputimer.h"

__global__ void printSome()
{
    printf("Hi I am block %d\n", blockIdx.x);
}

__global__ void syncThreads()
{
    int idx = threadIdx.x;
    __shared__ int array[128];

    if (idx < 127)
    {
        array[idx] = idx;
        __syncthreads();
        int temp = array[idx + 1];
        __syncthreads();
        array[idx] = temp;
        __syncthreads();
        printf("T:%d=%d ", idx, array[idx]);
    }
}

__global__ void use_local_memory(float in)
{
    float f; // variable "f" is in local memory and private to each thread.
    f = in; // parameter "in" is in local memory and private to each thread.
    (void)f;
    printf("%s\n", __FUNCTION__);
}

__global__ void use_shared_memory(float* in)
{
    int i;
    int index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in a block
    // and have same lifetime as the thread block.
    __shared__ float sh_arr[128];

    // copy data from global input array to sh_arr in the shared memory.
    // Each thread copies a single element.
    sh_arr[index] = in[index];

    __syncthreads();

    for (i = 0; i < index; i++)
    {
        sum += sh_arr[i];
    }

    average = sum / (index + 1.0f);

    // Just some shit to do. But note that no sync is needed since we are
    // operating on global memory now and previously shared memory was used to calculate average
    // so no sync is needed.
    if (in[index] > average)
    {
        in[index] = average;
    }

    printf("%s T:%d avg %f\n", __FUNCTION__, index, average);
}

__global__ void use_global_memory(float* in)
{
    // "in" is a pointer to global memory on the GPU device
    // Which means it has been cudaMalloc()-ed and cudaMemcpy()-ed from host to device.
     in[threadIdx.x] = 2.0f * (float)threadIdx.x;
    printf("%s T:%d\n", __FUNCTION__, threadIdx.x);
}

__global__ void increment_atomic(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % 10;
    atomicAdd(g + i, 1);
}

__global__ void incrementMilElementsMilThreads(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int elemValue = g[i];
    g[i] = elemValue + 1;
}

__global__ void incrementMilElementsMilThreadsAtomic(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(g + i, 1);
}

__global__ void increment100ElementsMilThreads(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % 100;
    int elemValue = g[i];
    g[i] = elemValue + 1;
}

__global__ void increment100ElementsMilThreadsAtomic(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % 100;
    atomicAdd(g + i, 1);
}

__global__ void increment100Elements10MilThreadsAtomic(int* g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % 100;
    atomicAdd(g + i, 1);
}

int tere(int exampleNr, bool print)
{
    if (exampleNr == 1)
    {
        printSome<<<16, 1>>>();
    }
    else if (exampleNr == 2)
    {
        syncThreads<<<1, 16>>>();
    }
    else if (exampleNr == 3)
    {
        use_local_memory<<<1, 1>>>(5.0f);

        float h_array[128];
        float* d_array;
        cudaMalloc(&d_array, sizeof(float) * 128);
        cudaMemcpy(d_array, h_array, sizeof(float) * 128, cudaMemcpyHostToDevice);
        use_global_memory<<<1, 128>>>(d_array);
        cudaMemcpy(h_array, d_array, sizeof(float) * 128, cudaMemcpyDeviceToHost);

        use_shared_memory<<<1, 128>>>(d_array);

        cudaFree(d_array);
    }
    else if (exampleNr == 4)
    {
        constexpr size_t ARRAY_ITEMS = 10;
        constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
        constexpr int THREAD_COUNT = 1000000;
        constexpr int BLOCK_WIDTH = 10000;


        int h_arr[10];
        int* d_array;
        cudaMalloc(&d_array, ARRAY_SIZE);
        cudaMemset(d_array, 0, ARRAY_SIZE);

        GpuTimer timer;
        timer.Start();
        increment_atomic<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
        timer.Stop();

        printf("time elapsed %g ms\n", timer.Elapsed());

        cudaFree(d_array);
    }
    else if (exampleNr == 5)
    {
        {
            constexpr size_t ARRAY_ITEMS = 1000000;
            constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
            constexpr int THREAD_COUNT = ARRAY_ITEMS;
            constexpr int BLOCK_WIDTH = 10000;

            int* d_array;
            cudaMalloc(&d_array, ARRAY_SIZE);
            cudaMemset(d_array, 0, ARRAY_SIZE);

            GpuTimer timer;
            timer.Start();
            incrementMilElementsMilThreads<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
            timer.Stop();

            printf("incrementMilElementsMilThreads took %g ms\n", timer.Elapsed());

            cudaFree(d_array);
        }

        {
            constexpr size_t ARRAY_ITEMS = 1000000;
            constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
            constexpr int THREAD_COUNT = ARRAY_ITEMS;
            constexpr int BLOCK_WIDTH = 10000;

            int* d_array;
            cudaMalloc(&d_array, ARRAY_SIZE);
            cudaMemset(d_array, 0, ARRAY_SIZE);

            GpuTimer timer;
            timer.Start();
            incrementMilElementsMilThreadsAtomic<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
            timer.Stop();

            printf("incrementMilElementsMilThreadsAtomic took %g ms\n", timer.Elapsed());

            cudaFree(d_array);
        }

        {
            constexpr size_t ARRAY_ITEMS = 100;
            constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
            constexpr int THREAD_COUNT = 1000000;
            constexpr int BLOCK_WIDTH = 10000;

            int* d_array;
            cudaMalloc(&d_array, ARRAY_SIZE);
            cudaMemset(d_array, 0, ARRAY_SIZE);

            GpuTimer timer;
            timer.Start();
            increment100ElementsMilThreads<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
            timer.Stop();

            printf("increment100ElementsMilThreads took %g ms\n", timer.Elapsed());

            cudaFree(d_array);
        }

        {
            constexpr size_t ARRAY_ITEMS = 100;
            constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
            constexpr int THREAD_COUNT = 1000000;
            constexpr int BLOCK_WIDTH = 10000;

            int* d_array;
            cudaMalloc(&d_array, ARRAY_SIZE);
            cudaMemset(d_array, 0, ARRAY_SIZE);

            GpuTimer timer;
            timer.Start();
            increment100ElementsMilThreadsAtomic<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
            timer.Stop();

            printf("increment100ElementsMilThreadsAtomic took %g ms\n", timer.Elapsed());

            cudaFree(d_array);
        }

        {
            constexpr size_t ARRAY_ITEMS = 100;
            constexpr size_t ARRAY_SIZE = ARRAY_ITEMS * sizeof(int);
            constexpr int THREAD_COUNT = 10000000;
            constexpr int BLOCK_WIDTH = 10000;

            int* d_array;
            cudaMalloc(&d_array, ARRAY_SIZE);
            cudaMemset(d_array, 0, ARRAY_SIZE);

            GpuTimer timer;
            timer.Start();
            increment100Elements10MilThreadsAtomic<<<THREAD_COUNT/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
            timer.Stop();

            printf("increment100Elements10MilThreadsAtomic took %g ms\n", timer.Elapsed());

            cudaFree(d_array);
        }
    }

    cudaDeviceSynchronize();

    return 0;
}