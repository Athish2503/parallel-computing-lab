#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    char message[100];
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3)
    {
        if (rank == 0)
            printf("Run with exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // -------- Process 0 : Sender --------
    if (rank == 0)
    {
        sleep(2); // delay for clarity

        // Blocking send
        strcpy(message, "Message for Blocking Receiver");
        printf("P0: Sending message to P1 (BLOCKING)...\n");
        fflush(stdout);
        MPI_Send(message, strlen(message)+1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("P0: Blocking send completed.\n\n");
        fflush(stdout);

        // Non-blocking send
        strcpy(message, "Message for Non-Blocking Receiver");
        printf("P0: Sending message to P2 (NON-BLOCKING)...\n");
        fflush(stdout);
        MPI_Isend(message, strlen(message)+1, MPI_CHAR, 2, 1, MPI_COMM_WORLD, &request);

        printf("P0: Doing other work while sending...\n");
        fflush(stdout);
        sleep(3);

        MPI_Wait(&request, &status);
        printf("P0: Non-blocking send completed.\n");
        fflush(stdout);
    }

    // -------- Process 1 : Blocking Receiver --------
    if (rank == 1)
    {
        printf("P1: Waiting for message (BLOCKING)...\n");
        fflush(stdout);

        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        printf("P1: Received blocking message: %s\n", message);
        fflush(stdout);
    }

    // -------- Process 2 : Non-Blocking Receiver --------
    if (rank == 2)
    {
        printf("P2: Starting non-blocking receive...\n");
        fflush(stdout);

        MPI_Irecv(message, 100, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &request);

        for (int i = 1; i <= 5; i++)
        {
            printf("P2: Doing other work %d...\n", i);
            fflush(stdout);
            sleep(1);
        }

        MPI_Wait(&request, &status);
        printf("P2: Received non-blocking message: %s\n", message);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
