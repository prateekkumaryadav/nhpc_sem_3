#ifndef MYVERBS_H
#define MYVERBS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <mpi.h>

#define CHUNK_SIZE (128 * 1024) /* 128 KB eager chunk size */
#define IB_PORT 1

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    
    /* Task 1: Pre-allocated Eager Buffers & MRs */
    char *send_buf;
    char *recv_buf;
    struct ibv_mr *send_mr;
    struct ibv_mr *recv_mr;

    /* Task 2: Pre-registered User Buffers & MRs (A priori) */
    struct ibv_mr *user_src_mr;
    struct ibv_mr *user_dst_mr;

    /* Remote Peer Information */
    uint32_t remote_qpn;
    uint16_t remote_lid;
    
    /* Remote User Buffer Information for Rendezvous Zero-Copy */
    uint64_t remote_dst_addr;
    uint32_t remote_dst_rkey;
};

struct qp_info {
    uint32_t qpn;
    uint16_t lid;
    uint64_t dst_addr;
    uint32_t dst_rkey;
};

extern struct rdma_context rdma_ctx;

void myverbs_init(int rank, int peer_rank, void *src_buf, void *dst_buf, size_t max_buf_size);
void myverbs_cleanup();

/* Task 1: Eager Protocol */
void myverbs_send_eager(const void *buf, int count, int dest_rank);
void myverbs_recv_eager(void *buf, int count, int src_rank);

/* Task 2: Optimized Rendezvous Protocol (Pre-registered) */
void myverbs_send_rendezvous(const void *buf, int count, int dest_rank);
void myverbs_recv_rendezvous(void *buf, int count, int src_rank);

#endif