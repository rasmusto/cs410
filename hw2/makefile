CC=gcc
CFLAGS=-Wall -Wextra -g -std=c99
default: sequential openmp

sequential: sequential.c
	${CC} ${CFLAGS} -o sequential.o sequential.c

openmp: openmp.c
	${CC} ${CFLAGS} -fopenmp -o openmp.o openmp.c

clean:
	rm sequential
