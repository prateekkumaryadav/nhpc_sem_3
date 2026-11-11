#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "myverbs.h"

#define MAX_SIZE (1024*1024*4)
#define START_SIZE (4096)
#define ITERATIONS 100
#define WARMUP     10

int main(int argc, char **argv) {
  int size, rank, src_rank=0, dst_rank=1;
  char *src=NULL, *dst=NULL;
  /* 0 = Base MPI, 1 = Verbs Eager, 2 = Verbs Rendezvous */
  int i, mode = 0; 
  struct timeval start, end;
  float total_time=0;

  if(argc > 1) src_rank = atoi(argv[1]);
  if(argc > 2) dst_rank = atoi(argv[2]);
  if(argc > 3) mode = atoi(argv[3]);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Allocate 4KB page-aligned buffers */
  posix_memalign((void**)&src, 4096, MAX_SIZE);
  posix_memalign((void**)&dst, 4096, MAX_SIZE);

  if (src == NULL || dst == NULL) {
     perror("cannot allocate memory\n");
     MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* Pre-initialize RDMA channel and pre-register user memory buffers upfront */
  if (mode > 0) {
      int peer = (rank == src_rank) ? dst_rank : src_rank;
      myverbs_init(rank, peer, src, dst, MAX_SIZE);
  }

  #define DO_TRANSFER(s, d, len, peer) do { \
      if (mode == 1) { \
          if (rank == src_rank) { myverbs_send_eager(s, len, peer); myverbs_recv_eager(d, len, peer); } \
          else { myverbs_recv_eager(d, len, peer); myverbs_send_eager(s, len, peer); } \
      } else if (mode == 2) { \
          if (rank == src_rank) { myverbs_send_rendezvous(s, len, peer); myverbs_recv_rendezvous(d, len, peer); } \
          else { myverbs_recv_rendezvous(d, len, peer); myverbs_send_rendezvous(s, len, peer); } \
      } else { \
          if (rank == src_rank) { \
              MPI_Send(s, len, MPI_CHAR, peer, 1, MPI_COMM_WORLD); \
              MPI_Recv(d, len, MPI_CHAR, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); \
          } else { \
              MPI_Recv(d, len, MPI_CHAR, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); \
              MPI_Send(s, len, MPI_CHAR, peer, 1, MPI_COMM_WORLD); \
          } \
      } \
  } while(0)

  /* Warmup */
  for(size = START_SIZE; size <= MAX_SIZE; size *= 2) {
    for (i = 0; i < WARMUP; i++) {
        int peer = (rank == src_rank) ? dst_rank : src_rank;
        DO_TRANSFER(src, dst, size, peer);
    }
  }

  /* Benchmark */
  if(rank == src_rank)
    printf("Mode: %s | Latency Benchmark (src=%d, dst=%d)\n%10s   %10s\n",
           (mode==0 ? "Base MPI" : (mode==1 ? "Verbs Eager" : "Verbs Rendezvous (Pre-registered)")),
           src_rank, dst_rank, "Size", "Latency (usecs)");

  for(size = START_SIZE; size <= MAX_SIZE; size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&start, NULL);
    for (i = 0; i < ITERATIONS; i++) {
        int peer = (rank == src_rank) ? dst_rank : src_rank;
        DO_TRANSFER(src, dst, size, peer);
    }
    gettimeofday(&end, NULL);

    total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / 2.0;
    total_time /= ITERATIONS;

    if(rank == src_rank)
      printf("%10d   %10.2f\n", size, total_time);
  }

  if (mode > 0) {
      myverbs_cleanup();
  }

  free(src);
  free(dst);
  MPI_Finalize();
  return 0;
}