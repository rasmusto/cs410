/* A toy example that adds two numbers on the device. */
#include <stdio.h>

__global__ void add(int *c, int a, int b) {
    *c = a + b;
}

int main(void) {
    int result;
    int *result_dev;
    cudaMalloc(&result_dev, sizeof(int));
    // <<<1,1>>> means: run the kernel on a grid of one block, where each block
    // has just one thread.
    add<<<1,1>>>(result_dev, 5, 6);
    cudaMemcpy(&result, result_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(result_dev);

    printf("%d\n", result);
    return 0;
}
