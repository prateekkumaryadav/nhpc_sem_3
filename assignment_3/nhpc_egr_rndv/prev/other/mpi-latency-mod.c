/*
 * mpi-latency-mod.c
 *
 * Modified MPI latency benchmark that measures:
 *   Phase 1: Eager protocol (chunked 128KB RDMA Send/Recv)
 *   Phase 2: Rendezvous protocol (RDMA Write with Immediate)
 *
 * Both protocols are benchmarked for all sizes from 4KB to 4MB.
 * Results are printed side-by-side for comparison.
 *
 * Compile:
 *   mpicc -O3 -std=gnu99 mpi-latency-mod.c myverbs.c -o verbs-latency -libverbs
 *
 * Run:
 *   mpirun -np 2 ./verbs-latency
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "myverbs.h"

#define MAX_SIZE   (1024*1024*4)   /* 4 MB */
#define ITERATIONS 100
#define WARMUP     10
#define PAGE_SIZE  4096

int main(int argc, char **argv)
{
    int mpi_size, rank, src_rank = 0, dst_rank = 1;
    char *src = NULL, *dst = NULL;
    int i, idx;
    int msg_size;
    struct timeval start, end;
    float total_time;
    float eager_latency[32];   /* enough for 4KB .. 4MB in powers of 2 */

    if (argc > 1) src_rank = atoi(argv[1]);
    if (argc > 2) dst_rank = atoi(argv[2]);

    /* Allocate page-aligned memory */
    if (posix_memalign((void **)&src, PAGE_SIZE, MAX_SIZE) != 0 ||
        posix_memalign((void **)&dst, PAGE_SIZE, MAX_SIZE) != 0) {
        perror("cannot allocate aligned memory");
        exit(1);
    }
    memset(src, 'A', MAX_SIZE);
    memset(dst, 0,   MAX_SIZE);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == src_rank)
        printf("Latency benchmark between src_rank=%d and dst_rank=%d\n",
               src_rank, dst_rank);

    /* Initialize RDMA verbs: open device, create QP, connect, pre-post recvs */
    myverbs_init(rank, src_rank, dst_rank);

    /* =========================================================
     *  PHASE 1:  EAGER PROTOCOL BENCHMARK
     * ========================================================= */

    /* Warmup */
    for (msg_size = PAGE_SIZE; msg_size <= MAX_SIZE; msg_size *= 2) {
        for (i = 0; i < WARMUP; i++) {
            if (rank == src_rank) {
                myverbs_send_eager(src, msg_size, dst_rank);
                myverbs_recv_eager(dst, msg_size, dst_rank);
            } else if (rank == dst_rank) {
                myverbs_recv_eager(dst, msg_size, src_rank);
                myverbs_send_eager(src, msg_size, src_rank);
            }
        }
    }

    /* Timed runs */
    idx = 0;
    for (msg_size = PAGE_SIZE; msg_size <= MAX_SIZE; msg_size *= 2) {
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start, NULL);
        for (i = 0; i < ITERATIONS; i++) {
            if (rank == src_rank) {
                myverbs_send_eager(src, msg_size, dst_rank);
                myverbs_recv_eager(dst, msg_size, dst_rank);
            } else if (rank == dst_rank) {
                myverbs_recv_eager(dst, msg_size, src_rank);
                myverbs_send_eager(src, msg_size, src_rank);
            }
        }
        gettimeofday(&end, NULL);

        total_time  = (end.tv_sec - start.tv_sec) * 1000000.0f
                    + (end.tv_usec - start.tv_usec);
        total_time /= 2.0f;                /* half-RTT = one-way latency */
        total_time /= (float)ITERATIONS;
        eager_latency[idx++] = total_time;
    }

    /* =========================================================
     *  PHASE 2:  RENDEZVOUS PROTOCOL BENCHMARK
     * ========================================================= */

    /* Warmup */
    for (msg_size = PAGE_SIZE; msg_size <= MAX_SIZE; msg_size *= 2) {
        for (i = 0; i < WARMUP; i++) {
            if (rank == src_rank) {
                myverbs_send_rendezvous(src, msg_size, dst_rank);
                myverbs_recv_rendezvous(dst, msg_size, dst_rank);
            } else if (rank == dst_rank) {
                myverbs_recv_rendezvous(dst, msg_size, src_rank);
                myverbs_send_rendezvous(src, msg_size, src_rank);
            }
        }
    }

    /* Print header */
    if (rank == src_rank)
        printf("\n%10s   %15s   %15s\n",
               "Size (B)", "Eager (us)", "Rendezvous (us)");

    /* Timed runs */
    idx = 0;
    for (msg_size = PAGE_SIZE; msg_size <= MAX_SIZE; msg_size *= 2) {
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start, NULL);
        for (i = 0; i < ITERATIONS; i++) {
            if (rank == src_rank) {
                myverbs_send_rendezvous(src, msg_size, dst_rank);
                myverbs_recv_rendezvous(dst, msg_size, dst_rank);
            } else if (rank == dst_rank) {
                myverbs_recv_rendezvous(dst, msg_size, src_rank);
                myverbs_send_rendezvous(src, msg_size, src_rank);
            }
        }
        gettimeofday(&end, NULL);

        total_time  = (end.tv_sec - start.tv_sec) * 1000000.0f
                    + (end.tv_usec - start.tv_usec);
        total_time /= 2.0f;
        total_time /= (float)ITERATIONS;

        if (rank == src_rank)
            printf("%10d   %15.2f   %15.2f\n",
                   msg_size, eager_latency[idx], total_time);
        idx++;
    }

    myverbs_finalize();
    free(src);
    free(dst);
    MPI_Finalize();
    return 0;
}