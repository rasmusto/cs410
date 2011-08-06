/* Author: Christopher Mitchell <chrism@lclark.edu>
 * Date: 2011-07-15
 *
 * Compile with `gcc gol.c`.
 */

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep

#define WIDTH 60
#define HEIGHT 30

// The two boards 
int current[WIDTH * HEIGHT];
int next[WIDTH * HEIGHT];

const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};


void fill_board(int *board) {
    int i;
    for (i=0; i<WIDTH*HEIGHT; i++)
        board[i] = rand() % 2;
}

void print_board(int *board) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}

void step() {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make this board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + WIDTH) % WIDTH;
                ny = (y + offsets[i][1] + HEIGHT) % HEIGHT;
                if (current[ny * WIDTH + nx]) {
                    num_neighbors++;
                }
            }

            // apply the game of life rules to this cell
            next[y * WIDTH + x] = 0;
            if ((current[y * WIDTH + x] && num_neighbors==2) || num_neighbors==3) {
                next[y * WIDTH + x] = 1;
            }
        }
    }

}

void animate() {
    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    while (1) {
        print_board(current);
        step();
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, sizeof(int) * WIDTH * HEIGHT);
        nanosleep(&delay, &remaining);
    }
}


int main(void) {
    // Initialize the global "current".
    fill_board(current);
    animate();

    return 0;
}
