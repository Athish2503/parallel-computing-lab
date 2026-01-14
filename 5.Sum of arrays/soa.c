#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, n;
    int *array = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    int *local_array = NULL;
    int local_n;
    int local_sum = 0;
    int *gathered_sums = NULL;
    int total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter number of elements: ");
        fflush(stdout);
        scanf("%d", &n);

        array = (int *)malloc(n * sizeof(int));
        printf("Enter %d elements:\n", n);
        for (int i = 0; i < n; i++)
            scanf("%d", &array[i]);
    }

   
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

 
    local_n = n / size;
    if (rank < n % size)
        local_n++;

    local_array = (int *)malloc(local_n * sizeof(int));

    
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = n / size;
            if (i < n % size)
                sendcounts[i]++;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        gathered_sums = (int *)malloc(size * sizeof(int));
    }

    MPI_Scatterv(
        array,
        sendcounts,
        displs,
        MPI_INT,
        local_array,
        local_n,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /* Compute local sum */
    for (int i = 0; i < local_n; i++)
        local_sum += local_array[i];

    printf("Rank %d received %d elements, local sum = %d\n",
           rank, local_n, local_sum);

    /* Gather local sums */
    MPI_Gather(
        &local_sum,
        1,
        MPI_INT,
        gathered_sums,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /* Root computes final sum */
    if (rank == 0) {
        printf("\nGathered sums:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d\n", i, gathered_sums[i]);
            total_sum += gathered_sums[i];
        }
        printf("\nTotal sum = %d\n", total_sum);

        free(array);
        free(sendcounts);
        free(displs);
        free(gathered_sums);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
