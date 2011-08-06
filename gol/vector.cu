#include <stdio.h>

/* The old-fashioned CPU-only way to add two vectors */
void add_vectors_host(int *result, int *a, int *b, int n) {
    for (int i=0; i<n; i++)
        result[i] = a[i] + b[i];
}

/* The kernel that will execute on the GPU */
__global__ void add_vectors_kernel(int *result, int *a, int *b, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // If we have more threads than the magnitude of our vector, we need to
    // make sure that the excess threads don't try to save results into
    // unallocated memory.
    if (idx < n)
        result[idx] = a[idx] + b[idx];
}

/* This encapsulates the process of setting up, executing, and tearing down the
 * environment to execute our vector addition kernel:
 *   1. Allocating memory on the device to hold our vectors
 *   2. Copy the vectors to device memory
 *   3. Executing the kernel
 *   4. Retrieve the result by copying the result vector from the device back
 *      to the host
 *   5. Free memory on the device
 * Doing this lets us do vector addition from main as easily as if we had just
 * called the host vector addition function.
 */
void add_vectors_dev(int *result, int *a, int *b, int n) {
    // Step 1: Allocating memory
    int *a_dev, *b_dev, *result_dev;

    // Since cudaMalloc does not return a pointer like C's traditional malloc
    // (it returns a success status instead), we provide as it's first argument
    // the address of our device pointer variable so that it can change the
    // value of our pointer to the correct device address.
    cudaMalloc((void **) &result_dev, sizeof(int) * n);
    cudaMalloc((void **) &a_dev, sizeof(int) * n);
    cudaMalloc((void **) &b_dev, sizeof(int) * n);

    // Step 2: Copying input vectors to the device
    cudaMemcpy(a_dev, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 3: Invoke the kernel
    // We allocate enough blocks (each 256 threads long) in the grid to
    // accomodate all `n` elements in the vectors. The 256 long block size
    // is somewhat arbitrary, but with the constraint that we know the
    // hardware will support blocks of that size.
    dim3 dimGrid((n + 256 - 1) / 256, 1, 1);
    dim3 dimBlock(256, 1, 1);
    add_vectors_kernel<<<dimGrid, dimBlock>>>(result_dev, a_dev, b_dev, n);

    // Step 4: Retrieve the results
    cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(result_dev);
}

void print_vector(int *array, int n) {
    int i;
    for (i=0; i<n; i++)
        printf("%d ", array[i]);
    printf("\n");
}

int main(void) {
    int n = 5; // Size of the arrays
    int a[] = {0, 1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    int host_result[5];
    int device_result[5];

    printf("The CPU's answer: ");
    add_vectors_host(host_result, a, b, n);
    print_vector(host_result, n);
    
    printf("The GPU's answer: ");
    add_vectors_dev(device_result, a, b, n);
    print_vector(device_result, n);
    return 0;
}
