#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Merge two sorted subarrays */
void merge(int *arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

/* Recursive Merge Sort (Divide & Conquer) */
void mergeSort(int *arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);     // Divide
        mergeSort(arr, m + 1, r); // Divide
        merge(arr, l, m, r);      // Combine
    }
}

int main(int argc, char *argv[])
{
    int rank, size, n;
    int *arr = NULL;
    int *sendcounts, *displs;
    int *local_arr;
    int local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Input by root */
    if (rank == 0)
    {
        printf("Enter number of elements (minimum 15): ");
        fflush(stdout);
        scanf("%d", &n);

        if (n < 15)
        {
            printf("Error: Minimum 15 elements required.\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        arr = (int *)malloc(n * sizeof(int));
        printf("Enter %d elements:\n", n);
        fflush(stdout);
        for (int i = 0; i < n; i++)
            scanf("%d", &arr[i]);
    }

    /* Broadcast n */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute send counts */
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    int base = n / size;
    int rem = n % size;

    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    local_n = sendcounts[rank];
    local_arr = (int *)malloc(local_n * sizeof(int));

    /* Root prints the original array and planned splits before scattering */
    if (rank == 0)
    {
        printf("Root: original array (%d elements):", n);
        for (int i = 0; i < n; i++)
            printf(" %d", arr[i]);
        printf("\n");

        for (int r = 0; r < size; r++)
        {
            printf("Root: will send %d elements to rank %d:", sendcounts[r], r);
            for (int t = 0; t < sendcounts[r]; t++)
                printf(" %d", arr[displs[r] + t]);
            printf("\n");
        }
        fflush(stdout);
    }

    /* Ensure root's planning prints appear before scatter */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Divide data among processes */
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT,
                 local_arr, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    /* Print the split assigned to each rank */
    printf("Rank %d: received %d elements:", rank, local_n);
    for (int i = 0; i < local_n; i++)
        printf(" %d", local_arr[i]);
    printf("\n");
    fflush(stdout);

    /* Synchronize so all ranks finish printing their splits */
    MPI_Barrier(MPI_COMM_WORLD);

    /* LOCAL DIVIDE AND CONQUER SORT */
    if (local_n > 0)
        mergeSort(local_arr, 0, local_n - 1);

    /* Print after local sort */
    printf("Rank %d: after local sort (%d elements):", rank, local_n);
    for (int i = 0; i < local_n; i++)
        printf(" %d", local_arr[i]);
    printf("\n");
    fflush(stdout);

    /* Synchronize so all ranks finish local sorting before merging */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Header for tree merge (printed once) */
    if (rank == 0)
    {
        printf("\nTree-based merging starting...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    int step = 1;
    int active = 1; /* stay in loop to participate in barriers even after sending */
    while (step < size)
    {
        /* synchronize start of this merging step */
        MPI_Barrier(MPI_COMM_WORLD);

        if (active)
        {
            if (rank % (2 * step) == 0)
            {
                int partner = rank + step;
                if (partner < size)
                {
                    printf("Rank %d (step %d): waiting to receive from %d\n", rank, step, partner);
                    fflush(stdout);

                    int recv_n;
                    MPI_Status status;
                    MPI_Recv(&recv_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &status);

                    printf("Rank %d (step %d): about to receive %d elements from %d\n", rank, step, recv_n, partner);
                    fflush(stdout);

                    int *recv_arr = (int *)malloc(recv_n * sizeof(int));
                    MPI_Recv(recv_arr, recv_n, MPI_INT, partner, 0, MPI_COMM_WORLD, &status);

                    printf("Rank %d (step %d): received elements from %d:", rank, step, partner);
                    for (int i = 0; i < recv_n; i++)
                        printf(" %d", recv_arr[i]);
                    printf("\n");
                    fflush(stdout);

                    /* merge local_arr and recv_arr into merged */
                    int *merged = (int *)malloc((local_n + recv_n) * sizeof(int));
                    int i = 0, j = 0, k = 0;
                    while (i < local_n && j < recv_n)
                        merged[k++] = (local_arr[i] <= recv_arr[j]) ? local_arr[i++] : recv_arr[j++];
                    while (i < local_n)
                        merged[k++] = local_arr[i++];
                    while (j < recv_n)
                        merged[k++] = recv_arr[j++];

                    free(local_arr);
                    free(recv_arr);
                    local_arr = merged;
                    local_n += recv_n;

                    printf("Rank %d (step %d): after merge size %d:", rank, step, local_n);
                    for (int t = 0; t < local_n; t++)
                        printf(" %d", local_arr[t]);
                    printf("\n");
                    fflush(stdout);
                }
            }
            else
            {
                int partner = rank - step;
                /* send size and data to partner; remain in loop to hit barriers */
                MPI_Send(&local_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                MPI_Send(local_arr, local_n, MPI_INT, partner, 0, MPI_COMM_WORLD);

                active = 0; /* keep participating in barriers only */
            }
        }

        /* synchronize end of this merging step */
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            printf("-- Completed merging step %d --\n", step);
            fflush(stdout);
        }

        step *= 2;
    }

    /* Final synchronization before printing the result */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Output */
    if (rank == 0)
    {
        printf("\nSorted Array:\n");
        fflush(stdout);
        for (int i = 0; i < local_n; i++)
            printf("%d ", local_arr[i]);
        printf("\n");
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
