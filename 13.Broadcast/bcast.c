#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("\nTotal Processes: %d\n", size);
    fflush(stdout);

    /********************* 1. BROADCAST *********************/
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        data = 100;
        printf("\n[Broadcast] Root process %d broadcasting data = %d\n", rank, data);
        fflush(stdout);
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[Broadcast] Process %d received data = %d\n", rank, data);
    fflush(stdout);

    /********************* 2. MULTICAST *********************/
    MPI_Barrier(MPI_COMM_WORLD);

    int multicast_data;

    if (rank == 0)
    {
        multicast_data = 200;
        printf("\n[Multicast] Root process %d sending data = %d to selected processes\n", rank, multicast_data);
        fflush(stdout);

        // Multicast to processes 1 and 2 only
        if (size > 1)
            MPI_Send(&multicast_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        if (size > 2)
            MPI_Send(&multicast_data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1 || rank == 2)
    {
        MPI_Recv(&multicast_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Multicast] Process %d received data = %d\n", rank, multicast_data);
        fflush(stdout);
    }
    else
    {
        printf("[Multicast] Process %d did NOT receive any data\n", rank);
        fflush(stdout);
    }

    /********************* 3. REDUCE *********************/
    MPI_Barrier(MPI_COMM_WORLD);  

    int local_value = rank + 1;  // each process contributes
    int global_sum = 0;

    printf("\n[Reduce] Process %d local value = %d\n", rank, local_value);
    fflush(stdout);

    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("[Reduce] Root process %d received global sum = %d\n", rank, global_sum);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
