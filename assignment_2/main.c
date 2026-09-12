/*
 * build:  gcc -O2 -std=gnu99 -o rdma_bench rdma_bench_fixed.c -libverbs
 * server: ./rdma_bench -s <port>
 * client: ./rdma_bench -c <ip> <port>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#define MAX_SIZE     (4 * 1024 * 1024)
#define ITERS        100
#define WARMUP       10
#define PAGE_SIZE    4096
#define RX_DEPTH     16
#define GID_IDX      0

struct pp_dest {
    uint16_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;
    uint64_t addr;
    uint32_t rkey;
};

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void oob_send_all(int fd, const void *buf, size_t len) {
    size_t off = 0;
    const char *p = buf;
    while (off < len) {
        ssize_t s = send(fd, p + off, len - off, 0);
        if (s <= 0) { perror("oob send"); exit(1); }
        off += s;
    }
}

static void oob_recv_all(int fd, void *buf, size_t len) {
    size_t off = 0;
    char *p = buf;
    while (off < len) {
        ssize_t s = recv(fd, p + off, len - off, 0);
        if (s <= 0) { perror("oob recv"); exit(1); }
        off += s;
    }
}

static int oob_listen_accept(int port) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = { .sin_family = AF_INET,
                                 .sin_port = htons(port),
                                 .sin_addr.s_addr = INADDR_ANY };
    if (bind(ls, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); exit(1); }
    listen(ls, 1);
    printf("OOB listening on port %d...\n", port);
    int fd = accept(ls, NULL, NULL);
    close(ls);
    return fd;
}

static int oob_connect(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
    inet_pton(AF_INET, ip, &addr.sin_addr);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("connect"); exit(1); }
    return fd;
}

struct rdma_ctx {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    struct ibv_cq      *cq;
    struct ibv_qp      *qp;
    struct ibv_mr      *mr;
    char               *buf;
    int                 port_num;
    int                 is_server;

    struct ibv_sge      recv_sge;
    struct ibv_recv_wr  recv_wr;
    struct ibv_sge      send_sge;
    struct ibv_send_wr  send_wr;
    struct ibv_send_wr  write_wr;
};

static struct ibv_context *open_first_device(void) {
    int n;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) {
        fprintf(stderr, "No RDMA devices found. Is an IB/RoCE NIC + driver present?\n");
        exit(1);
    }
    struct ibv_context *ctx = ibv_open_device(list[0]);
    if (!ctx) { fprintf(stderr, "ibv_open_device failed\n"); exit(1); }
    ibv_free_device_list(list);
    return ctx;
}

static void modify_qp_init(struct ibv_qp *qp, int port_num) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ;
    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp, &attr, flags)) { perror("modify_qp INIT"); exit(1); }
}

static void modify_qp_rtr(struct ibv_qp *qp, int port_num, struct pp_dest *rem,
                           enum ibv_mtu mtu, int my_psn_unused) {
    (void)my_psn_unused;
    struct ibv_qp_attr attr = {0};
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = mtu;
    attr.dest_qp_num        = rem->qpn;
    attr.rq_psn             = rem->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12;
    attr.ah_attr.is_global  = 0;
    attr.ah_attr.dlid       = rem->lid;
    attr.ah_attr.sl         = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num   = port_num;

    if (rem->lid == 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = rem->gid;
        attr.ah_attr.grh.sgid_index = GID_IDX;
        attr.ah_attr.grh.hop_limit = 1;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp, &attr, flags)) { perror("modify_qp RTR"); exit(1); }
}

static void modify_qp_rts(struct ibv_qp *qp, uint32_t my_psn) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.sq_psn        = my_psn;
    attr.max_rd_atomic = 1;
    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp, &attr, flags)) { perror("modify_qp RTS"); exit(1); }
}

static void setup_rdma(struct rdma_ctx *r, int oob_fd, int is_server) {
    r->is_server = is_server;
    r->port_num  = 1;
    r->ctx = open_first_device();

    struct ibv_port_attr port_attr;
    if (ibv_query_port(r->ctx, r->port_num, &port_attr)) { perror("query_port"); exit(1); }

    r->pd = ibv_alloc_pd(r->ctx);
    if (!r->pd) { fprintf(stderr, "alloc_pd failed\n"); exit(1); }

    if (posix_memalign((void **)&r->buf, PAGE_SIZE, MAX_SIZE)) { perror("posix_memalign"); exit(1); }
    memset(r->buf, is_server ? 0x00 : 0xAB, MAX_SIZE);

    r->cq = ibv_create_cq(r->ctx, RX_DEPTH * 4, NULL, NULL, 0);
    if (!r->cq) { fprintf(stderr, "create_cq failed\n"); exit(1); }

    r->mr = ibv_reg_mr(r->pd, r->buf, MAX_SIZE,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!r->mr) { fprintf(stderr, "reg_mr failed\n"); exit(1); }

    struct ibv_qp_init_attr qattr = {0};
    qattr.send_cq = r->cq;
    qattr.recv_cq = r->cq;
    qattr.qp_type = IBV_QPT_RC;
    qattr.cap.max_send_wr  = RX_DEPTH;
    qattr.cap.max_recv_wr  = RX_DEPTH;
    qattr.cap.max_send_sge = 1;
    qattr.cap.max_recv_sge = 1;

    r->qp = ibv_create_qp(r->pd, &qattr);
    if (!r->qp) { fprintf(stderr, "create_qp failed\n"); exit(1); }

    modify_qp_init(r->qp, r->port_num);

    union ibv_gid my_gid = {0};
    if (port_attr.lid == 0) {
        if (ibv_query_gid(r->ctx, r->port_num, GID_IDX, &my_gid)) {
            perror("query_gid"); exit(1);
        }
    }

    struct pp_dest local = {0};
    local.lid  = port_attr.lid;
    local.qpn  = r->qp->qp_num;
    local.psn  = lrand48() & 0xffffff;
    local.gid  = my_gid;
    local.addr = (uint64_t)(uintptr_t)r->buf;
    local.rkey = r->mr->rkey;

    struct pp_dest remote;
    if (is_server) {
        oob_send_all(oob_fd, &local, sizeof(local));
        oob_recv_all(oob_fd, &remote, sizeof(remote));
    } else {
        oob_recv_all(oob_fd, &remote, sizeof(remote));
        oob_send_all(oob_fd, &local, sizeof(local));
    }

    enum ibv_mtu mtu = port_attr.active_mtu < IBV_MTU_1024 ? port_attr.active_mtu : IBV_MTU_1024;

    modify_qp_rtr(r->qp, r->port_num, &remote, mtu, local.psn);
    modify_qp_rts(r->qp, local.psn);

    r->recv_sge = (struct ibv_sge){ .addr = (uint64_t)(uintptr_t)r->buf, .length = 0, .lkey = r->mr->lkey };
    r->recv_wr = (struct ibv_recv_wr){ .wr_id = 1, .sg_list = &r->recv_sge, .num_sge = 1 };

    r->send_sge = (struct ibv_sge){ .addr = (uint64_t)(uintptr_t)r->buf, .length = 0, .lkey = r->mr->lkey };
    r->send_wr = (struct ibv_send_wr){ .wr_id = 2, .sg_list = &r->send_sge, .num_sge = 1,
                                       .opcode = IBV_WR_SEND, .send_flags = IBV_SEND_SIGNALED };

    r->write_wr = (struct ibv_send_wr){ .wr_id = 3, .sg_list = &r->send_sge, .num_sge = 1,
                                        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM, .send_flags = IBV_SEND_SIGNALED,
                                        .wr = { .rdma = { .remote_addr = remote.addr, .rkey = remote.rkey } } };
}

static inline void post_recv(struct rdma_ctx *r, size_t len) {
    r->recv_sge.length = len;
    struct ibv_recv_wr *bad;
    if (ibv_post_recv(r->qp, &r->recv_wr, &bad)) { perror("post_recv"); exit(1); }
}

static inline void post_send(struct rdma_ctx *r, size_t len) {
    r->send_sge.length = len;
    struct ibv_send_wr *bad;
    if (ibv_post_send(r->qp, &r->send_wr, &bad)) { perror("post_send"); exit(1); }
}

static inline void post_write_imm(struct rdma_ctx *r, size_t len) {
    r->send_sge.length = len;
    r->write_wr.imm_data = htonl((uint32_t)len);
    struct ibv_send_wr *bad;
    if (ibv_post_send(r->qp, &r->write_wr, &bad)) { perror("post_write"); exit(1); }
}

static inline void poll_n(struct rdma_ctx *r, int n) {
    int got = 0;
    struct ibv_wc wc[4];
    while (got < n) {
        int ne = ibv_poll_cq(r->cq, n - got, wc);
        if (ne < 0) { fprintf(stderr, "poll_cq failed\n"); exit(1); }
        for (int i = 0; i < ne; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "completion error: %s (wr_id=%llu)\n",
                        ibv_wc_status_str(wc[i].status), (unsigned long long)wc[i].wr_id);
                exit(1);
            }
        }
        got += ne;
    }
}

static void server_loop(struct rdma_ctx *r) {
    for (size_t size = 1; size <= MAX_SIZE; size *= 2) {
        /* Phase 1: pure SendRecv echo, WARMUP + ITERS times */
        for (int i = 0; i < WARMUP + ITERS; i++) {
            post_recv(r, size);
            poll_n(r, 1);
            post_send(r, size);
            poll_n(r, 1);
        }
        /* Phase 2: pure RDMA-Write-with-Immediate echo, WARMUP + ITERS times */
        for (int i = 0; i < WARMUP + ITERS; i++) {
            post_recv(r, size);
            poll_n(r, 1);
            post_write_imm(r, size);
            poll_n(r, 1);
        }
    }
}

static void client_loop(struct rdma_ctx *r) {
    printf("%-10s%16s%16s%16s%16s\n",
           "Msg size", "SendRecv(us)", "RDMAWrite(us)", "MRreg(us)", "MRdereg(us)");

    for (size_t size = 1; size <= MAX_SIZE; size *= 2) {
        /* Phase 1: SendRecv - warmup then timed, matcheing server's Phase 1 count */
        for (int i = 0; i < WARMUP; i++) {
            post_recv(r, size);
            post_send(r, size);
            poll_n(r, 2);
        }
        double t0 = now_us();
        for (int i = 0; i < ITERS; i++) {
            post_recv(r, size);
            post_send(r, size);
            poll_n(r, 2);
        }
        double t1 = now_us();
        double sr_us = (t1 - t0) / ITERS / 2.0;

        /* Phase 2: RDMA Write - warmup then timed, matches server's Phase 2 count */
        for (int i = 0; i < WARMUP; i++) {
            post_recv(r, size);
            post_write_imm(r, size);
            poll_n(r, 2);
        }
        double t2 = now_us();
        for (int i = 0; i < ITERS; i++) {
            post_recv(r, size);
            post_write_imm(r, size);
            poll_n(r, 2);
        }
        double t3 = now_us();
        double wr_us = (t3 - t2) / ITERS / 2.0;

        size_t reg_len = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        if (reg_len == 0) reg_len = PAGE_SIZE;

        void *tmp;
        if (posix_memalign(&tmp, PAGE_SIZE, reg_len)) { perror("posix_memalign"); exit(1); }
        memset(tmp, 0, reg_len);

        double reg_sum = 0, dereg_sum = 0;
        for (int i = 0; i < ITERS; i++) {
            double r0 = now_us();
            struct ibv_mr *mr = ibv_reg_mr(r->pd, tmp, reg_len,
                                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
            double r1 = now_us();
            if (!mr) { fprintf(stderr, "reg_mr failed for size %zu\n", size); exit(1); }
            reg_sum += (r1 - r0);

            double d0 = now_us();
            ibv_dereg_mr(mr);
            double d1 = now_us();
            dereg_sum += (d1 - d0);
        }
        free(tmp);

        double reg_avg = reg_sum / ITERS;
        double dereg_avg = dereg_sum / ITERS;

        printf("%-10zu%16.2f%16.2f%16.2f%16.2f\n",
               size, sr_us, wr_us, reg_avg, dereg_avg);
    }
}

int main(int argc, char **argv) {
    srand48(getpid());

    if (argc >= 3 && strcmp(argv[1], "-s") == 0) {
        int port = atoi(argv[2]);
        int oob = oob_listen_accept(port);
        struct rdma_ctx r;
        setup_rdma(&r, oob, 1);
        server_loop(&r);
        close(oob);
    } else if (argc >= 4 && strcmp(argv[1], "-c") == 0) {
        int oob = oob_connect(argv[2], atoi(argv[3]));
        struct rdma_ctx r;
        setup_rdma(&r, oob, 0);
        client_loop(&r);
        close(oob);
    } else {
        fprintf(stderr, "Usage:\n  %s -s <port>\n  %s -c <ip> <port>\n", argv[0], argv[0]);
        return 1;
    }
    return 0;
}