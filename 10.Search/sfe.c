#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0, target = 0;
    int *arr = NULL;

    if (rank == 0)
    {
        printf("Enter the number of array elements: ");
        fflush(stdout);
        scanf("%d", &n);

        if (n <= 0)
        {
            printf("Error: Array size must be positive.\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        arr = (int *)malloc(n * sizeof(int));

        printf("Enter the array elements:\n");
        fflush(stdout);
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &arr[i]);
        }

        printf("Enter the element to search for: ");
        fflush(stdout);
        scanf("%d", &target);

        printf("\nInput Array: ");
        for (int i = 0; i < n; i++)
            printf("%d ", arr[i]);
        printf("\nTarget Element: %d\n\n", target);
        fflush(stdout);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int base = n / size;
    int remainder = n % size;

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_n = sendcounts[rank];
    int *local_arr = (int *)malloc(local_n * sizeof(int));

    MPI_Scatterv(arr, sendcounts, displs, MPI_INT,
                 local_arr, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d received %d elements: ", rank, local_n);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local_arr[i]);
    printf("\n");
    fflush(stdout);

    /* Find local index (global index = displs[rank] + i) */
    int local_index = -1;
    for (int i = 0; i < local_n; i++)
    {
        if (local_arr[i] == target)
        {
            local_index = displs[rank] + i; /* compute global index */
            break;
        }
    }

    if (local_index != -1)
        printf("Process %d search result: FOUND at index %d\n", rank, local_index);
    else
        printf("Process %d search result: NOT FOUND\n", rank);
    fflush(stdout);

    /* Reduce a flag indicating whether the element was found */
    int local_found = (local_index != -1) ? 1 : 0;
    int global_found = 0;
    MPI_Reduce(&local_found, &global_found,
               1, MPI_INT, MPI_LOR,
               0, MPI_COMM_WORLD);

    /* Reduce the global index to find the first (lowest) index where target appears.
       Use INT_MAX as a sentinel so ranks that didn't find it don't influence the min. */
    int reduce_local_index = (local_index == -1) ? INT_MAX : local_index;
    int global_index = INT_MAX;
    MPI_Reduce(&reduce_local_index, &global_index,
               1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (global_found)
            printf("\nFINAL RESULT: Element %d FOUND in the array at index %d.\n", target, global_index);
        else
            printf("\nFINAL RESULT: Element %d NOT FOUND in the array.\n", target);
        fflush(stdout);
    }

    free(local_arr);
    free(sendcounts);
    free(displs);
    if (rank == 0)
        free(arr);

    MPI_Finalize();
    return 0;
}
