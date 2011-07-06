#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

void mxv(int m, int n, double * restrict a, double * restrict b, double * restrict c);

int main(int argc, char * argv[])
{
    double *a,*b,*c;
    int i, j, m, n;

    printf("Please give m and n: ");
    scanf("%d %d",&m,&n);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    if ( (a=(double *)malloc(m*sizeof(double))) == NULL )
        perror("memory allocation for a");
    if ( (b=(double *)malloc(m*n*sizeof(double))) == NULL )
        perror("memory allocation for b");
    if ( (c=(double *)malloc(n*sizeof(double))) == NULL )
        perror("memory allocation for c");

    printf("Initializing matrix B and vector c\n");
    for (j=0; j<n; j++)
        c[j] = 2.0;
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            b[i*n+j] = i;

    printf("Vector c:\n");
    for (j=0; j<n; j++)
        printf("c[%d] = %f\n", j, c[j]);

    printf("Matrix B:\n");
    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            printf("b[%d] = %f\n", i*n+j, b[i*n+j]);

    printf("Executing mxv function for m = %d n = %d\n",m,n);
    (void) mxv(m, n, a, b, c);
    gettimeofday(&end, NULL);
    printf("Elapsed time: %ldus\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

    printf("Vector a:\n");
    for (j=0; j<n; j++)
        printf("a[%d] = %f\n", j, a[j]);

    free(a);free(b);free(c);
    return(0);
}

void mxv(int m, int n, double * restrict a, double * restrict b, double * restrict c)
{
    int i, j;

    for (i=0; i<m; i++)
    {
        a[i] = 0.0;
        for (j=0; j<n; j++)
            a[i] += b[i*n+j]*c[j];
    }
}
