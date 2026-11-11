#ifndef MYVERBS_H
#define MYVERBS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <mpi.h>

/*
128 KB eager chunk size
*/ 
#define CHUNK_SIZE (128 * 1024) 
#define IB_PORT 1

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    
    /* Eager buffers */
    char *send_buf;
    char *recv_buf;
    struct ibv_mr *send_mr;
    struct ibv_mr *recv_mr;
    
    /* Peer details */
    uint32_t remote_qpn;
    uint16_t remote_lid;
};

struct qp_info {
    uint32_t qpn;
    uint16_t lid;
};

struct rkey_msg {
    uint64_t addr;
    uint32_t rkey;
    uint32_t len;
};

extern struct rdma_context rdma_ctx;

void myverbs_init(int rank, int peer_rank);
void myverbs_cleanup();

/* Task 1: Eager Protocol */
void myverbs_send_eager(const void *buf, int count, int dest_rank);
void myverbs_recv_eager(void *buf, int count, int src_rank);

/* Task 2: Rendezvous Protocol */
void myverbs_send_rendezvous(const void *buf, int count, int dest_rank);
void myverbs_recv_rendezvous(void *buf, int count, int src_rank);

#endif