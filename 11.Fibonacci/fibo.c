#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, N;
    int *fib = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter N: ");
        fflush(stdout);
        scanf("%d", &N);

        fib = (int *)malloc(N * sizeof(int));
        fib[0] = 0;
        fib[1] = 1;

        for (int i = 2; i < N; i++)
            fib[i] = fib[i - 1] + fib[i - 2];
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        fib = (int *)malloc(N * sizeof(int));

    MPI_Bcast(fib, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk = N / size;
    int start = rank * chunk;
    int end = start + chunk;

    // Fix for remainder
    if (rank == size - 1)
        end = N;

    int local_sum = 0, local_odd = 0, local_even = 0;

    printf("Process %d handles indices [%d - %d)\n", rank, start, end);
    fflush(stdout);

    for (int i = start; i < end; i++)
    {
        printf("Process %d -> fib[%d] = %d\n", rank, i, fib[i]);
        fflush(stdout);

        local_sum += fib[i];
        if (fib[i] % 2 == 0)
            local_even++;
        else
            local_odd++;
    }

    printf("\n=== Process %d Results ===\n", rank);
    printf("Process %d: Sum = %d, Odd Count = %d, Even Count = %d\n", rank, local_sum, local_odd, local_even);
    fflush(stdout);

    int total_sum, total_odd, total_even;

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_odd, &total_odd, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_even, &total_even, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < N; i++) {
            printf("%d ", fib[i]);
        }
        printf("\n");

        printf("\n====== FINAL RESULT (ROBUST) ======\n");
        printf("Total Sum of %d Fibonacci Numbers = %d\n", N, total_sum);
        printf("Total Odd Numbers = %d\n", total_odd);
        printf("Total Even Numbers = %d\n", total_even);
        printf("==================================\n");
        fflush(stdout);
    }

    free(fib);
    MPI_Finalize();
    return 0;
}
