CC=gcc
CFLAGS=-Wall -Wextra -g -std=c99
CUDAFLAGS=--compiler-options -Wall
default: sequential openmp pthreads cuda

sequential: sequential.c
	${CC} ${CFLAGS} -o sequential.o sequential.c

openmp: openmp.c
	${CC} ${CFLAGS} -fopenmp -o openmp.o openmp.c

pthreads: pthreads.c
	${CC} ${CFLAGS} -lpthread -o pthreads.o pthreads.c

cuda: cuda.cu
	nvcc ${CUDAFLAGS} -o cuda.o cuda.cu 

clean:
	rm -f *.o
