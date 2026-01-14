#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size, n;
    int *arr = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    int local_n;
    int *local_arr;
    int local_sum = 0, total_sum = 0;
    float average;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process inputs number of elements
    if (rank == 0)
    {
        printf("Enter number of elements: ");
        fflush(stdout);
        scanf("%d", &n);

        arr = (int *)malloc(n * sizeof(int));
        printf("Enter %d elements:\n", n);
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &arr[i]);
        }

        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int base = n / size;
        int rem = n % size;

        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }

        // Print distribution on root
        printf("Root process %d computed distribution:\n", rank);
        for (int i = 0; i < size; i++)
        {
            printf("  sendcounts[%d] = %d, displs[%d] = %d\n", i, sendcounts[i], i, displs[i]);
        }
        fflush(stdout);
    }

    // Broadcast sendcounts to all processes
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received local_n = %d\n", rank, local_n);
    fflush(stdout);

    local_arr = (int *)malloc(local_n * sizeof(int));

    // Scatter elements unevenly
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT,
                 local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (local_n > 0)
    {
        printf("Process %d received elements:", rank);
        for (int i = 0; i < local_n; i++)
        {
            printf(" %d", local_arr[i]);
        }
        printf("\n");
    }
    else
    {
        printf("Process %d received no elements\n", rank);
    }
    fflush(stdout);

    // Local sum
    for (int i = 0; i < local_n; i++)
    {
        local_sum += local_arr[i];
    }
    printf("Process %d computed local_sum = %d\n", rank, local_sum);
    fflush(stdout);

    // Reduce to get total sum
    printf("Process %d sending local_sum %d to root\n", rank, local_sum);
    fflush(stdout);
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root calculates average
    if (rank == 0)
    {
        average = (float)total_sum / n;
        printf("\nTotal Sum = %d", total_sum);
        printf("\nAverage = %.2f\n", average);

        free(arr);
        free(sendcounts);
        free(displs);
    }

    free(local_arr);

    MPI_Finalize();
    return 0;
}
