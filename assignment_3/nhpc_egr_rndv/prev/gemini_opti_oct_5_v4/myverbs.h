#ifndef MYVERBS_H
#define MYVERBS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <mpi.h>
#include <emmintrin.h>

#define CHUNK_SIZE (128 * 1024)
#define MAX_CHUNKS 512            /* Increased to support larger buffers up to 64MB */
#define INLINE_THRESHOLD 256      /* Reduced to safe hardware maximum */
#define IB_PORT 1

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

extern struct rdma_context rdma_ctx;

void myverbs_init(int rank, int peer_rank, void *src_buf, void *dst_buf, size_t max_buf_size);
void myverbs_cleanup();

void myverbs_send_eager(const void *buf, int count, int dest_rank);
void myverbs_recv_eager(void *buf, int count, int src_rank);

void myverbs_send_rendezvous(const void *buf, int count, int dest_rank);
void myverbs_recv_rendezvous(void *buf, int count, int src_rank);

#endif