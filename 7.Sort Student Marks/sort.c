#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Comparator for qsort */
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/* Merge two sorted arrays */
void merge(int *a, int na, int *b, int nb, int *result)
{
    int i = 0, j = 0, k = 0;

    while (i < na && j < nb)
        result[k++] = (a[i] < b[j]) ? a[i++] : b[j++];

    while (i < na) result[k++] = a[i++];
    while (j < nb) result[k++] = b[j++];
}

int main(int argc, char *argv[])
{
    int rank, size, n = 0;
    int *marks = NULL, *local_marks = NULL, *sorted = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setvbuf(stdout, NULL, _IONBF, 0);

    /* Root input handling */
    if (rank == 0)
    {
        printf("Enter number of students: ");
        fflush(stdout);

        if (scanf("%d", &n) != 1 || n <= 0)
        {
            fprintf(stderr, "Invalid number of students\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        marks = malloc(n * sizeof(int));
        if (!marks)
        {
            fprintf(stderr, "Memory allocation failed\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Enter %d student marks:\n", n);
        fflush(stdout);

        for (int i = 0; i < n; i++)
        {
            if (scanf("%d", &marks[i]) != 1)
            {
                fprintf(stderr, "Invalid mark input\n");
                fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        printf("\nOriginal Marks: ");
        for (int i = 0; i < n; i++)
            printf("%d ", marks[i]);

        printf("\n");
        fflush(stdout);
    }

    /* Broadcast n */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute distribution */
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    if (!sendcounts || !displs)
    {
        fprintf(stderr, "Distribution allocation failed\n");
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base = n / size;
    int rem  = n % size;

    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    if (sendcounts[rank] > 0)
    {
        local_marks = malloc(sendcounts[rank] * sizeof(int));
        if (!local_marks)
        {
            fprintf(stderr, "Local marks allocation failed\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Scatter marks */
    MPI_Scatterv(marks, sendcounts, displs, MPI_INT,
                 local_marks, sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Print received marks */
    for (int p = 0; p < size; p++)
    {
        if (rank == p && sendcounts[rank] > 0)
        {
            printf("Process %d received: ", rank);
            fflush(stdout);

            for (int i = 0; i < sendcounts[rank]; i++)
                printf("%d ", local_marks[i]);

            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    
    if (sendcounts[rank] > 0 && local_marks != NULL)
        qsort(local_marks, sendcounts[rank], sizeof(int), compare);

    
    for (int p = 0; p < size; p++)
    {
        if (rank == p && sendcounts[rank] > 0)
        {
            printf("Process %d locally sorted: ", rank);
            fflush(stdout);

            for (int i = 0; i < sendcounts[rank]; i++)
                printf("%d ", local_marks[i]);

            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

   
    if (rank == 0)
    {
        sorted = malloc(n * sizeof(int));
        if (!sorted)
        {
            fprintf(stderr, "Sorted allocation failed\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(local_marks, sendcounts[rank], MPI_INT,
                sorted, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

   
    if (rank == 0)
    {
        int *temp = malloc(n * sizeof(int));
        int current_size = sendcounts[0];

        for (int i = 1; i < size; i++)
        {
            merge(sorted, current_size,
                  sorted + displs[i], sendcounts[i],
                  temp);

            current_size += sendcounts[i];

            for (int j = 0; j < current_size; j++)
                sorted[j] = temp[j];
        }

        printf("\nFinal Sorted Marks: ");
        fflush(stdout);

        for (int i = 0; i < n; i++)
            printf("%d ", sorted[i]);

        printf("\n");
        fflush(stdout);

        free(temp);
        free(temp);
    }

    if (local_marks) free(local_marks);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
