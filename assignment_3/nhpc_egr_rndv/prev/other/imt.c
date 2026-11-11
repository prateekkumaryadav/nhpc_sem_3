#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX_SIZE (1024 * 1024 * 32)
#define ITERATIONS 100
#define WARMUP 20

// Comparison function for qsort (median computation)
int compare_doubles_mpi(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

double compute_median_mpi(double *latencies, int count) {
    qsort(latencies, count, sizeof(double), compare_doubles_mpi);
    if (count % 2 == 0) {
        return (latencies[count / 2 - 1] + latencies[count / 2]) / 2.0;
    } else {
        return latencies[count / 2];
    }
}

int main(int argc, char **argv) {
  int size, rank, src_rank = 0, dst_rank = 1;
  char *src = NULL, *dst = NULL;
  int i, msg_size;
  MPI_Status status;
  struct timespec start, end;
  double latencies[ITERATIONS];

  if (argc > 1) src_rank = atoi(argv[1]);
  if (argc > 2) dst_rank = atoi(argv[2]);

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == src_rank) {
    printf("\n=======================================================\n");
    printf("    Unmodified MPI Median Latency Benchmark            \n");
    printf("=======================================================\n");
    printf("%14s   %12s   %18s\n", "Size (bytes)", "Iterations", "Median Latency (us)");
    printf("-------------------------------------------------------\n");
  }

  src = (char *)malloc(MAX_SIZE);
  dst = (char *)malloc(MAX_SIZE);
  if (src == NULL || dst == NULL) {
     perror("cannot allocate memory\n");
     exit(1);
  }

  // Initial synchronization barrier before warmup
  MPI_Barrier(MPI_COMM_WORLD);

  // Warmup iterations starting from 1 byte
  for (msg_size = 1; msg_size <= MAX_SIZE; msg_size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (i = 0; i < WARMUP; i++) {
       if (rank == src_rank) {
         MPI_Send(src, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD);
         MPI_Recv(dst, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &status);
       } else if (rank == dst_rank) {
         MPI_Recv(dst, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);
         MPI_Send(src, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD);
       }
    }
  }

  // Initial synchronization barrier before benchmark
  MPI_Barrier(MPI_COMM_WORLD);
  for (msg_size = 1; msg_size <= MAX_SIZE; msg_size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < ITERATIONS; i++) {
       if (rank == src_rank) {
         clock_gettime(CLOCK_MONOTONIC, &start);

         MPI_Send(src, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD);
         MPI_Recv(dst, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &status);

         clock_gettime(CLOCK_MONOTONIC, &end);

         double rtt_us = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;
         latencies[i] = rtt_us / 2.0;

       } else if (rank == dst_rank) {
         MPI_Recv(dst, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);
         MPI_Send(src, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD);
       }
    }

    if (rank == src_rank) {
      double median_lat = compute_median_mpi(latencies, ITERATIONS);
      printf("%14d   %12d   %18.2f\n", msg_size, ITERATIONS, median_lat);
    }
  }

  free(src);
  free(dst);
  MPI_Finalize();
  return 0;
}