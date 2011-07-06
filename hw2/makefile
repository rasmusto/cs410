CC=gcc
CFLAGS=-Wall -Wextra -g -std=c99
default: sequential openmp pthreads

sequential: sequential.c
	${CC} ${CFLAGS} -o sequential.o sequential.c

openmp: openmp.c
	${CC} ${CFLAGS} -fopenmp -o openmp.o openmp.c

pthreads: pthreads.c
	${CC} ${CFLAGS} -lpthread -o pthreads.o pthreads.c

clean:
	rm -f *.o
