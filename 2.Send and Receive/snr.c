#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    char message[50];

    if (size < 2){
        printf("This program requires at least two processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0){
        strcpy(message, "Hello from process 0");
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message: %s\n", message);
    }
    else if (rank == 1){
        MPI_Recv(message, 50, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message: %s\n", message);
    }

    MPI_Finalize();
    return 0;
}
