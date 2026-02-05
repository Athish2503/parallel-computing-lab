#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("\nProcess %d started\n", rank);
    fflush(stdout);

    /**************** WORKER PROCESSES ****************/
    if (rank != 0)
    {
        int send_data = rank * 10;

        // Artificial delay for fairness demo
        sleep(rank);

        printf("[Sender] Process %d sending data = %d\n",
               rank, send_data);
        fflush(stdout);

        MPI_Send(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    /**************** ROOT PROCESS ****************/
    else
    {
        int *recv_buffer = (int *)malloc((size - 1) * sizeof(int));
        MPI_Request *requests = (MPI_Request *)malloc((size - 1) * sizeof(MPI_Request));
        MPI_Status *statuses = (MPI_Status *)malloc((size - 1) * sizeof(MPI_Status));

        printf("\n[Root] Posting non-blocking receives...\n");
        fflush(stdout);

        for (int i = 1; i < size; i++)
        {
            MPI_Irecv(&recv_buffer[i - 1], 1, MPI_INT,
                      i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        int remaining = size - 1;

        while (remaining > 0)
        {
            int outcount;
            int *indices = (int *)malloc(remaining * sizeof(int));

            MPI_Waitsome(size - 1, requests,
                         &outcount, indices, statuses);

            printf("\n[Root] MPI_Waitsome returned %d messages\n", outcount);
            fflush(stdout);

            for (int i = 0; i < outcount; i++)
            {
                int idx = indices[i];
                int sender_rank = statuses[i].MPI_SOURCE;

                printf("[Root] Received data = %d from Process %d\n",
                       recv_buffer[idx], sender_rank);
                fflush(stdout);

                // Mark request as completed
                requests[idx] = MPI_REQUEST_NULL;
                remaining--;
            }

            free(indices);
        }

        printf("\n[Root] All messages received fairly using MPI_Waitsome\n");
        fflush(stdout);

        free(recv_buffer);
        free(requests);
        free(statuses);
    }

    MPI_Finalize();
    return 0;
}
