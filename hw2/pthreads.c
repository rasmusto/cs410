#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 5

void * mxv(void *threadid);

/*typedef struct str_thdata
{
    int m;
    int n;
    double * restrict a;
    double * restrict b;
    double * restrict c;
    void * t;
} thdata;*/

typedef struct str_thdata
{
    int thread_id;
} thdata;

int i, j, m, n;
double *a,*b,*c;

int main(int argc, char * argv[])
{


    printf("Please give m and n: ");
    scanf("%d %d",&m,&n);

    pthread_t threads[n];
    pthread_mutex_t mutexsum;

    int rc;
    long t;

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

    thdata data[5];
    for(i=0; i<n; i++){
        data[i].thread_id = i;
        printf("thread_id = %d\n", data[i].thread_id);
    }

    /*
    data.n = 0;
    data.m = m;
    data.a = a;
    data.b = b;
    data.c = c;
    for(i=0; i<n; i++)
        data.a[i] = 0.0;
    */
    for(i=0; i<n; i++)
        a[i] = 0.0;

    printf("Executing mxv function for m = %d n = %d\n",m,n);
    //create a thread for each row of the matrix
    for(i=0; i<n; i++) {
        rc = pthread_create(&threads[i], NULL, mxv, (void *) &data[i]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    //wait for them all to finish
    for(i=0; i<n; i++)
        pthread_join(threads[i], NULL);

    printf("Vector a:\n");
    for (j=0; j<n; j++)
        printf("a[%d] = %f\n", j, a[j]);

    free(a);free(b);free(c);
    pthread_exit(NULL);
    return(0);
}

void * mxv(void * ptr)
{
    thdata * data;
    data = (thdata *) ptr;
    for (j=0; j<m; j++)
    {
        //printf("modifying a[%d]\n", data->thread_id);
        a[data->thread_id] += b[j+data->thread_id*n]*c[j];
    }
    pthread_exit(NULL);
}
