// cuda.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

__global__ void vector_multiply_row_device(float * a, float * b, float * c, int m)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int j;
    for (j=0; j<m; j++)
    {
        a[idx] += b[j+idx*m]*c[j];
    }
}
int main(void)
{
    float *a_h, *b_h, *c_h;            // pointers to host memory
    float *a_d, *b_d, *c_d;            // pointers to device memory
    int i, j, m, n;

    printf("Please give m and n: ");
    scanf("%d %d",&m,&n);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // allocate arrays on host
    if ( (a_h=(float *)malloc(m*sizeof(float))) == NULL )
        perror("memory allocation for a");
    if ( (b_h=(float *)malloc(m*n*sizeof(float))) == NULL )
        perror("memory allocation for b");
    if ( (c_h=(float *)malloc(n*sizeof(float))) == NULL )
        perror("memory allocation for c");

    // allocate array on device 
    cudaMalloc((void **) &a_d, n*sizeof(float));
    cudaMalloc((void **) &b_d, m*n*sizeof(float));
    cudaMalloc((void **) &c_d, n*sizeof(float));

    // initialization of host data
    printf("Initializing matrix B and vector c\n");
    for (j=0; j<n; j++)
        c_h[j] = 2.0;
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            b_h[i*n+j] = i;

    printf("Vector c:\n");
    for (j=0; j<n; j++)
        printf("c_h[%d] = %f\n", j, c_h[j]);

    printf("Matrix B:\n");
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            printf("b_h[%d] = %f\n", i*n+j, b_h[i*n+j]);

    printf("Initializing a to 0\n");
    for(i=0; i<n; i++)
        a_h[i] = 0.0;

    // copy data from host to device
    cudaMemcpy(a_d, a_h, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, m*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 4;
    int numBlocks = m;

    vector_multiply_row_device <<< blockSize, numBlocks >>> (a_d, b_d, c_d, m);

    cudaMemcpy(a_h, a_d, m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_h, b_d, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_h, c_d, m*sizeof(float), cudaMemcpyDeviceToHost);

    gettimeofday(&end, NULL);
    printf("Elapsed time: %ldus\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

    printf("Vector a_h:\n");
    for (j=0; j<n; j++)
        printf("a_h[%d] = %f\n", j, a_h[j]);

    // cleanup
    free(a_h); free(b_h); free(c_h); cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); 
}
