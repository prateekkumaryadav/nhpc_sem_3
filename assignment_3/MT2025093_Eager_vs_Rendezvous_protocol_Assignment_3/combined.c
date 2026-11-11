#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <mpi.h>

#define CHUNK_SIZE (128 * 1024)
#define MAX_CHUNKS 32
#define INLINE_THRESHOLD 256
#define IB_PORT 1

#define MAX_SIZE (1024 * 1024 * 4)
#define START_SIZE (4096)
#define ITERATIONS 100
#define WARMUP 10

/* =========================================================================
   1. STRUCTS & GLOBAL STATE
   ========================================================================= */

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    
    char *send_buf[2];
    char *recv_buf[2];
    struct ibv_mr *send_mr[2];
    struct ibv_mr *recv_mr[2];

    struct ibv_mr *user_src_mr;
    struct ibv_mr *user_dst_mr;

    uint32_t remote_qpn;
    uint16_t remote_lid;
    uint64_t remote_dst_addr;
    uint32_t remote_dst_rkey;
} __attribute__((aligned(64)));

struct qp_info {
    uint32_t qpn;
    uint16_t lid;
    uint64_t dst_addr;
    uint32_t dst_rkey;
};

static struct rdma_context rdma_ctx;

/* =========================================================================
   2. VERBS PROTOCOL IMPLEMENTATION
   ========================================================================= */

static void setup_qp_state(struct ibv_qp *qp, uint16_t remote_lid, uint32_t remote_qpn) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT,
        .pkey_index = 0,
        .port_num = IB_PORT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote_lid;
    attr.ah_attr.port_num = IB_PORT;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
}

void myverbs_init(int rank, int peer_rank, void *src_buf, void *dst_buf, size_t max_buf_size) {
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    rdma_ctx.ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);

    rdma_ctx.pd = ibv_alloc_pd(rdma_ctx.ctx);
    rdma_ctx.cq = ibv_create_cq(rdma_ctx.ctx, 512, NULL, NULL, 0);

    struct ibv_qp_init_attr qp_init_attr = {
        .send_cq = rdma_ctx.cq,
        .recv_cq = rdma_ctx.cq,
        .cap = {
            .max_send_wr = 256,
            .max_recv_wr = 256,
            .max_send_sge = 1,
            .max_recv_sge = 1,
            .max_inline_data = INLINE_THRESHOLD
        },
        .qp_type = IBV_QPT_RC
    };
    rdma_ctx.qp = ibv_create_qp(rdma_ctx.pd, &qp_init_attr);

    for (int k = 0; k < 2; k++) {
        posix_memalign((void**)&rdma_ctx.send_buf[k], 4096, CHUNK_SIZE);
        posix_memalign((void**)&rdma_ctx.recv_buf[k], 4096, CHUNK_SIZE);
        rdma_ctx.send_mr[k] = ibv_reg_mr(rdma_ctx.pd, rdma_ctx.send_buf[k], CHUNK_SIZE, IBV_ACCESS_LOCAL_WRITE);
        rdma_ctx.recv_mr[k] = ibv_reg_mr(rdma_ctx.pd, rdma_ctx.recv_buf[k], CHUNK_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }

    rdma_ctx.user_src_mr = ibv_reg_mr(rdma_ctx.pd, src_buf, max_buf_size, IBV_ACCESS_LOCAL_WRITE);
    rdma_ctx.user_dst_mr = ibv_reg_mr(rdma_ctx.pd, dst_buf, max_buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    struct ibv_port_attr port_attr;
    ibv_query_port(rdma_ctx.ctx, IB_PORT, &port_attr);

    struct qp_info my_info = {
        .qpn = rdma_ctx.qp->qp_num,
        .lid = port_attr.lid,
        .dst_addr = (uint64_t)dst_buf,
        .dst_rkey = rdma_ctx.user_dst_mr->rkey
    };
    struct qp_info peer_info;

    MPI_Sendrecv(&my_info, sizeof(my_info), MPI_BYTE, peer_rank, 0,
                 &peer_info, sizeof(peer_info), MPI_BYTE, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    rdma_ctx.remote_qpn = peer_info.qpn;
    rdma_ctx.remote_lid = peer_info.lid;
    rdma_ctx.remote_dst_addr = peer_info.dst_addr;
    rdma_ctx.remote_dst_rkey = peer_info.dst_rkey;

    setup_qp_state(rdma_ctx.qp, peer_info.lid, peer_info.qpn);
}

void myverbs_cleanup() {
    for (int k = 0; k < 2; k++) {
        ibv_dereg_mr(rdma_ctx.send_mr[k]);
        ibv_dereg_mr(rdma_ctx.recv_mr[k]);
        free(rdma_ctx.send_buf[k]);
        free(rdma_ctx.recv_buf[k]);
    }
    ibv_dereg_mr(rdma_ctx.user_src_mr);
    ibv_dereg_mr(rdma_ctx.user_dst_mr);
    ibv_destroy_qp(rdma_ctx.qp);
    ibv_destroy_cq(rdma_ctx.cq);
    ibv_dealloc_pd(rdma_ctx.pd);
    ibv_close_device(rdma_ctx.ctx);
}

void myverbs_send_eager(const void *buf, int count, int dest_rank) {
    int bytes_left = count;
    const char *src_ptr = (const char *)buf;
    int buf_idx = 0;
    int inflight = 0;

    while (bytes_left > 0 || inflight > 0) {
        while (bytes_left > 0 && inflight < 2) {
            int send_bytes = (bytes_left > CHUNK_SIZE) ? CHUNK_SIZE : bytes_left;
            memcpy(rdma_ctx.send_buf[buf_idx], src_ptr, send_bytes);

            struct ibv_sge list = {
                .addr = (uint64_t)rdma_ctx.send_buf[buf_idx],
                .length = send_bytes,
                .lkey = rdma_ctx.send_mr[buf_idx]->lkey
            };
            struct ibv_send_wr wr = {
                .sg_list = &list,
                .num_sge = 1,
                .opcode = IBV_WR_SEND,
                .send_flags = IBV_SEND_SIGNALED,
                .next = NULL
            }, *bad_wr;

            ibv_post_send(rdma_ctx.qp, &wr, &bad_wr);
            bytes_left -= send_bytes;
            src_ptr += send_bytes;
            buf_idx ^= 1;
            inflight++;
        }

        struct ibv_wc wc;
        if (ibv_poll_cq(rdma_ctx.cq, 1, &wc) > 0) {
            inflight--;
        }
    }
}

void myverbs_recv_eager(void *buf, int count, int src_rank) {
    int num_chunks = (count + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_chunks == 0) num_chunks = 1;

    struct ibv_sge sge[MAX_CHUNKS];
    struct ibv_recv_wr wr[MAX_CHUNKS];
    struct ibv_recv_wr *bad_wr;

    int bytes_left = count;
    int buf_idx = 0;

    for (int i = 0; i < num_chunks; i++) {
        int recv_bytes = (bytes_left > CHUNK_SIZE) ? CHUNK_SIZE : bytes_left;

        sge[i].addr = (uint64_t)rdma_ctx.recv_buf[buf_idx];
        sge[i].length = CHUNK_SIZE;
        sge[i].lkey = rdma_ctx.recv_mr[buf_idx]->lkey;

        wr[i].sg_list = &sge[i];
        wr[i].num_sge = 1;
        wr[i].next = (i < num_chunks - 1) ? &wr[i + 1] : NULL;

        bytes_left -= recv_bytes;
        buf_idx ^= 1;
    }

    ibv_post_recv(rdma_ctx.qp, &wr[0], &bad_wr);

    char *dst_ptr = (char *)buf;
    for (int i = 0; i < num_chunks; i++) {
        struct ibv_wc wc;
        while (ibv_poll_cq(rdma_ctx.cq, 1, &wc) < 1);

        int chunk_idx = i % 2;
        memcpy(dst_ptr, rdma_ctx.recv_buf[chunk_idx], wc.byte_len);
        dst_ptr += wc.byte_len;
    }
}

void myverbs_send_rendezvous(const void *buf, int count, int dest_rank) {
    struct ibv_sge list = {
        .addr = (uint64_t)buf,
        .length = count,
        .lkey = rdma_ctx.user_src_mr->lkey
    };
    struct ibv_send_wr wr = {
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED,
        .imm_data = htonl(count),
        .next = NULL,
        .wr = {
            .rdma = {
                .remote_addr = rdma_ctx.remote_dst_addr,
                .rkey = rdma_ctx.remote_dst_rkey
            }
        }
    }, *bad_wr;

    ibv_post_send(rdma_ctx.qp, &wr, &bad_wr);

    struct ibv_wc wc;
    while (ibv_poll_cq(rdma_ctx.cq, 1, &wc) < 1);
}

void myverbs_recv_rendezvous(void *buf, int count, int src_rank) {
    struct ibv_recv_wr rwr = { .num_sge = 0, .next = NULL }, *bad_rwr;
    ibv_post_recv(rdma_ctx.qp, &rwr, &bad_rwr);

    struct ibv_wc wc;
    while (ibv_poll_cq(rdma_ctx.cq, 1, &wc) < 1);
}

/* =========================================================================
   3. LATENCY MEASUREMENT BENCHMARK RUNNER
   ========================================================================= */

#define DO_TRANSFER(s, d, len, peer) do { \
    if (mode == 1) { \
        if (rank == src_rank) { myverbs_send_eager(s, len, peer); myverbs_recv_eager(d, len, peer); } \
        else { myverbs_recv_eager(d, len, peer); myverbs_send_eager(s, len, peer); } \
    } else if (mode == 2) { \
        if (rank == src_rank) { \
            myverbs_send_rendezvous(s, len, peer); \
            myverbs_recv_rendezvous(d, len, peer); \
        } else { \
            myverbs_recv_rendezvous(d, len, peer); \
            myverbs_send_rendezvous(s, len, peer); \
        } \
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

int main(int argc, char **argv) {
    int size, rank, comm_size, src_rank = 0, dst_rank = 1;
    char *src = NULL, *dst = NULL;
    /* 0 = Base MPI, 1 = Verbs Eager, 2 = Verbs Rendezvous */
    int i, mode = 0; 
    struct timeval start, end;
    float total_time = 0;

    if (argc > 1) src_rank = atoi(argv[1]);
    if (argc > 2) dst_rank = atoi(argv[2]);
    if (argc > 3) mode = atoi(argv[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    posix_memalign((void**)&src, 4096, MAX_SIZE);
    posix_memalign((void**)&dst, 4096, MAX_SIZE);

    if (src == NULL || dst == NULL) {
        perror("cannot allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (mode > 0) {
        int peer = (rank == src_rank) ? dst_rank : src_rank;
        myverbs_init(rank, peer, src, dst, MAX_SIZE);
    }

    /* Warmup iterations */
    for (size = START_SIZE; size <= MAX_SIZE; size *= 2) {
        for (i = 0; i < WARMUP; i++) {
            int peer = (rank == src_rank) ? dst_rank : src_rank;
            DO_TRANSFER(src, dst, size, peer);
        }
    }

    /* Latency Benchmark Loop */
    if (rank == src_rank) {
        printf("Mode: %s | Latency Benchmark (src=%d, dst=%d)\n%10s   %10s\n",
               (mode == 0 ? "Base MPI" : (mode == 1 ? "Verbs Eager" : "Verbs Rendezvous (Pre-registered)")),
               src_rank, dst_rank, "Size", "Latency (usecs)");
    }

    for (size = START_SIZE; size <= MAX_SIZE; size *= 2) {
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start, NULL);
        for (i = 0; i < ITERATIONS; i++) {
            int peer = (rank == src_rank) ? dst_rank : src_rank;
            DO_TRANSFER(src, dst, size, peer);
        }
        gettimeofday(&end, NULL);

        total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / 2.0;
        total_time /= ITERATIONS;

        if (rank == src_rank) {
            printf("%10d   %10.2f\n", size, total_time);
        }
    }

    if (mode > 0) {
        myverbs_cleanup();
    }

    free(src);
    free(dst);
    MPI_Finalize();
    return 0;
}