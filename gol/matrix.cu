/* This is a somewhat naive matrix-multiplication implementation. The most
 * glaring shortcoming is the lack of shared-memory usage.
 *
 * Compile with `nvcc matrix_multiplication.cu`.
 *
 * Author: Christopher Mitchell <chrism@lclark.edu>
 * Date: 2011-07-15
 */

#include <stdio.h>
#include <stdlib.h>


typedef struct {
    int width;
    int height;
    // This should point to a row-major array of elements. We use a 1D array
    // instead of a 2d array to represent our matrix because copying 2D data to
    // the device is more complex.
    float *elements; 
} Matrix;


/* Matrix multiply A by B on the CPU, storing the product into result. */
void mat_mult_host(Matrix *result, Matrix *A, Matrix *B) {
    int row, col, sum, idx;
    for (col=0; col < (A->width); col++) {
        for (row=0; row < (A->height); row++) {
            sum = 0;
            for (idx=0; idx < (A->height); idx++) {
                sum += A->elements[row * A->width + idx] * B->elements[idx * B->width + col];
            }
            result->elements[row * result->width + col] = sum;
        }
    }
}


/* The matrix multiplication kernel used by the mat_mult_dev function. */
__global__ void kernel_mat_mult(Matrix result, Matrix A, Matrix B) {
    int row, col, sum, idx;
    row = blockIdx.y * blockDim.y + threadIdx.y;
    col = blockIdx.x * blockDim.x + threadIdx.x;
    sum = 0;
    // This is the inner-most loop of the host matrix multiplication function.
    for (idx=0; idx < (A.height); idx++) {
        sum += A.elements[row * A.width + idx] * B.elements[idx * B.width + col];
    }
    result.elements[row * result.width + col] = sum;
}

/* Matrix multiply A by B on the GPU, storing the product into result. */
void mat_mult_dev(Matrix *result, Matrix *A, Matrix *B) {
    int size;
    Matrix A_dev, B_dev, C_dev;

    /* When we copy the input matrices to the device, we only copy the elements
     * and not the entire structure. We do this to avoid the complexity of
     * having to find the size structures with variable length arrays.
     */
    A_dev.width = A->width;
    A_dev.height = A->height;
    size = sizeof(float) * A->width * A->height;
    cudaMalloc(&A_dev.elements, size);
    cudaMemcpy(A_dev.elements, A->elements, size, cudaMemcpyHostToDevice);

    B_dev.width = B->width;
    B_dev.height = B->height;
    size = sizeof(float) * B->width * B->height;
    cudaMalloc(&B_dev.elements, size);
    cudaMemcpy(B_dev.elements, B->elements, size, cudaMemcpyHostToDevice);

    C_dev.width = result->width;
    C_dev.height = result->height;
    size = sizeof(float) * result->width * result->height;
    cudaMalloc(&C_dev.elements, size);

    /* Since the kernel only uses one block, once the number of cells in the
     * result matrix exceeds the number of threads that can be in a block,
     * things will fail. Thouh we don't do so in this program, one can find the
     * max number of threads in a block that our card supports by: 
     *     cudaDeviceProp devProps = cudaGetDeviceProperties(&devProps, 0);
     *     printf("Max threads per block: %d", devProps.maxThreadsPerBlock);
     */
    dim3 dimGrid(1,1,1);
    dim3 dimBlock(result->width, result->height, 1);
    kernel_mat_mult<<<dimGrid,dimBlock>>>(C_dev, A_dev, B_dev);
    
    // retrieve the result from the GPU
    cudaMemcpy(result->elements, C_dev.elements, size, cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(A_dev.elements);
    cudaFree(B_dev.elements);
    cudaFree(C_dev.elements);
}


/* Print the matrix so that each row is on a new line, and each value in a row
 * is separated by a space. */
void print_mat(Matrix *mat) {
    int row, col;
    for (row = 0; row < (mat->height); row++) {
        for (col = 0; col < (mat->width); col++) {
            printf("%lf ", mat->elements[row * mat->width + col]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(void) {
    /* Initialize the matrix that we will be multiplying with itself */
    Matrix matrix = {8, 8}; // Set the width and height
    float elements[] = {1, 2, 3, 4, 5, 6, 7, 8, 
                        2, 3, 4, 5, 6, 7, 8, 9, 
                        3, 4, 5, 6, 7, 8, 9, 10, 
                        4, 5, 6, 7, 8, 9, 10, 11, 
                        5, 6, 7, 8, 9, 10, 11, 12, 
                        6, 7, 8, 9, 10, 11, 12, 13, 
                        7, 8, 9, 10, 11, 12, 13, 14, 
                        8, 9, 10, 11, 12, 13, 14, 15};
    matrix.elements = elements; // Set the elements

    /* Allocate space for the result vectors. */
    Matrix cpu_result = {8, 8};
    Matrix gpu_result = {8, 8};
    cpu_result.elements = (float *) malloc(sizeof(float) * cpu_result.width * cpu_result.height);
    gpu_result.elements = (float *) malloc(sizeof(float) * gpu_result.width * gpu_result.height);

    printf("CPU result:\n");
    mat_mult_host(&cpu_result, &matrix, &matrix);
    print_mat(&cpu_result);

    printf("GPU result:\n");
    mat_mult_dev(&gpu_result, &matrix, &matrix);
    print_mat(&gpu_result);
    return 0;
}
