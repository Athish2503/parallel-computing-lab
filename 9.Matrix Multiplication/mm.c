#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void printMatrix(int *mat, int rows, int cols, const char *name)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%4d ", mat[i * cols + j]);
        printf("\n");
    }
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int R, C, K;
    int *A = NULL, *B = NULL, *Cmat = NULL;

    /* Input on rank 0 only */
    if (rank == 0)
    {
        printf("Enter rows and columns of Matrix A (min 3x3): ");
        fflush(stdout);
        scanf("%d %d", &R, &C);

        printf("Enter columns of Matrix B: ");
        fflush(stdout);
        scanf("%d", &K);

        if (R < 3 || C < 3 || K < 3)
        {
            printf("ERROR: Minimum matrix size is 3x3\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A = malloc(R * C * sizeof(int));
        B = malloc(C * K * sizeof(int));
        Cmat = malloc(R * K * sizeof(int));

        printf("\nEnter Matrix A (row-wise):\n");
        fflush(stdout);
        for (int i = 0; i < R * C; i++)
            scanf("%d", &A[i]);

        printf("\nEnter Matrix B (row-wise):\n");
        fflush(stdout);
        for (int i = 0; i < C * K; i++)
            scanf("%d", &B[i]);

        printMatrix(A, R, C, "Matrix A");
        printMatrix(B, C, K, "Matrix B");
    }

    /* Broadcast dimensions */
    MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Broadcast Matrix B */
    if (rank != 0)
        B = malloc(C * K * sizeof(int));

    MPI_Bcast(B, C * K, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d received Matrix B\n", rank);
    fflush(stdout);

    /* Row distribution logic */
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int rows_per_proc = R / size;
    int extra = R % size;

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        int rows_assigned = rows_per_proc + (i < extra);
        sendcounts[i] = rows_assigned * C;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    /* Rank 0 prints distribution plan */
    if (rank == 0)
    {
        printf("\nRow distribution plan:\n");
        for (int i = 0; i < size; i++)
        {
            int start_row = displs[i] / C;
            int end_row = start_row + (sendcounts[i] / C) - 1;
            printf("Process %d -> rows %d to %d\n", i, start_row, end_row);
        }
        fflush(stdout);
    }

    int local_rows = sendcounts[rank] / C;
    int *local_A = malloc(local_rows * C * sizeof(int));
    int *local_C = malloc(local_rows * K * sizeof(int));

    MPI_Scatterv(A, sendcounts, displs, MPI_INT,
                 local_A, sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    int global_row_offset = displs[rank] / C;

    MPI_Barrier(MPI_COMM_WORLD);
    printf("\nProcess %d received rows %d to %d of Matrix A\n",
           rank,
           global_row_offset,
           global_row_offset + local_rows - 1);

    printf("Process %d local Matrix A:\n", rank);
    for (int i = 0; i < local_rows; i++)
    {
        printf("Row %d: ", global_row_offset + i);
        for (int j = 0; j < C; j++)
            printf("%d ", local_A[i * C + j]);
        printf("\n");
    }
    fflush(stdout);

    /* Local matrix multiplication */
    for (int i = 0; i < local_rows; i++)
    {
        int global_i = global_row_offset + i;

        for (int j = 0; j < K; j++)
        {
            int sum = 0;

            printf("\n[Process %d] Computing C[%d][%d]\n",
                   rank, global_i, j);

            for (int k = 0; k < C; k++)
            {
                int a = local_A[i * C + k];
                int b = B[k * K + j];
                sum += a * b;

                printf("  A[%d][%d]=%d * B[%d][%d]=%d | sum=%d\n",
                       global_i, k, a, k, j, b, sum);
            }

            local_C[i * K + j] = sum;
            printf("  => C[%d][%d] = %d\n", global_i, j, sum);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("\nProcess %d finished computation\n", rank);
    fflush(stdout);

    /* Prepare gather info */
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (sendcounts[i] / C) * K;
        displs[i] = (displs[i] / C) * K;
    }

    MPI_Gatherv(local_C, sendcounts[rank], MPI_INT,
                Cmat, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
        printMatrix(Cmat, R, K, "Result Matrix C");

    /* Cleanup */
    free(local_A);
    free(local_C);
    free(B);
    free(sendcounts);
    free(displs);

    if (rank == 0)
    {
        free(A);
        free(Cmat);
    }

    MPI_Finalize();
    return 0;
}
