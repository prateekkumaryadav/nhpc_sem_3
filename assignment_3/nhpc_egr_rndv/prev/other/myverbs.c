/*
 * myverbs.c - RDMA Verbs implementation for MPI latency benchmark
 *
 * Task 1: Eager protocol  - memcpy to pre-registered 128KB buffer, IBV_WR_SEND
 * Task 2: Rendezvous protocol - register user buffer, exchange handle via
 *         channel semantics, then RDMA Write with Immediate
 *
 * Key design decisions:
 *   - Separate send_cq and recv_cq to prevent completion mixing
 *   - MTU 4096 to match hardware capability (mlx4_0 active_mtu=4096)
 *   - Receive buffers always reposted immediately after consumption
 *   - MPI_Barrier at end of init to prevent RNR race
 *   - Error checking on all verbs calls
 */

#include <mpi.h>
#include <infiniband/verbs.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define EAGER_BUF_SIZE (128 * 1024)  /* 128KB chunk size for eager */

/* ---- Global RDMA state ---- */
static struct ibv_context *ctx;
static struct ibv_pd      *pd;
static struct ibv_cq      *send_cq;
static struct ibv_cq      *recv_cq;
static struct ibv_qp      *qp;

static struct ibv_mr *eager_send_mr;
static struct ibv_mr *eager_recv_mr;
static char          *eager_send_buf;
static char          *eager_recv_buf;

static int my_rank;   /* cached for error messages */

/* Structure for exchanging memory handles in Rendezvous */
struct mr_info {
    uint64_t addr;
    uint32_t rkey;
    uint32_t size;    /* so receiver knows the payload size */
    uint32_t pad;     /* keep alignment */
};

/* ================================================================
 * Helper: post a single receive WR on the eager recv buffer
 * ================================================================ */
static void post_eager_recv(void)
{
    struct ibv_sge sge;
    sge.addr   = (uint64_t)eager_recv_buf;
    sge.length = EAGER_BUF_SIZE;
    sge.lkey   = eager_recv_mr->lkey;

    struct ibv_recv_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sge;
    wr.num_sge = 1;

    int ret = ibv_post_recv(qp, &wr, &bad_wr);
    if (ret) {
        fprintf(stderr, "Rank %d: ibv_post_recv failed (%d)\n", my_rank, ret);
        exit(1);
    }
}

/* ================================================================
 * Helper: poll a CQ until one completion arrives; abort on error
 * ================================================================ */
static void poll_cq(struct ibv_cq *cq, const char *op_name)
{
    struct ibv_wc wc;
    int ret;

    do {
        ret = ibv_poll_cq(cq, 1, &wc);
    } while (ret == 0);

    if (ret < 0) {
        fprintf(stderr, "Rank %d [%s]: ibv_poll_cq returned %d\n",
                my_rank, op_name, ret);
        exit(1);
    }
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "Rank %d [%s]: WC error %s (%d), wr_id=%lu\n",
                my_rank, op_name,
                ibv_wc_status_str(wc.status), wc.status,
                (unsigned long)wc.wr_id);
        exit(1);
    }
}

/* ================================================================
 * myverbs_init  –  open device, create QP, exchange addresses,
 *                  transition QP to RTS, pre-post receives
 * ================================================================ */
void myverbs_init(int rank, int src_rank, int dst_rank)
{
    int ret;
    my_rank = rank;

    /* 1. Open first IB device */
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Rank %d: no IB devices found\n", rank);
        exit(1);
    }
    ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!ctx) { perror("ibv_open_device"); exit(1); }

    /* 2. Allocate PD */
    pd = ibv_alloc_pd(ctx);
    if (!pd) { perror("ibv_alloc_pd"); exit(1); }

    /* 3. Create separate Send and Recv CQs to avoid completion mixing */
    send_cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);
    recv_cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);
    if (!send_cq || !recv_cq) { perror("ibv_create_cq"); exit(1); }

    /* 4. Create RC Queue Pair */
    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq          = send_cq;
    qp_attr.recv_cq          = recv_cq;
    qp_attr.qp_type          = IBV_QPT_RC;
    qp_attr.cap.max_send_wr  = 128;
    qp_attr.cap.max_recv_wr  = 128;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) { perror("ibv_create_qp"); exit(1); }

    /* 5. Allocate & register eager buffers (page-aligned) */
    if (posix_memalign((void **)&eager_send_buf, 4096, EAGER_BUF_SIZE) ||
        posix_memalign((void **)&eager_recv_buf, 4096, EAGER_BUF_SIZE)) {
        perror("posix_memalign eager bufs"); exit(1);
    }
    memset(eager_send_buf, 0, EAGER_BUF_SIZE);
    memset(eager_recv_buf, 0, EAGER_BUF_SIZE);

    eager_send_mr = ibv_reg_mr(pd, eager_send_buf, EAGER_BUF_SIZE,
                               IBV_ACCESS_LOCAL_WRITE);
    eager_recv_mr = ibv_reg_mr(pd, eager_recv_buf, EAGER_BUF_SIZE,
                               IBV_ACCESS_LOCAL_WRITE  |
                               IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ);
    if (!eager_send_mr || !eager_recv_mr) {
        perror("ibv_reg_mr eager"); exit(1);
    }

    /* 6. Query local port LID */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(ctx, 1, &port_attr);
    if (ret) { perror("ibv_query_port"); exit(1); }

    uint32_t my_lid = port_attr.lid;
    uint32_t my_qpn = qp->qp_num;

    /* 7. Exchange LID + QPN with peer via MPI */
    uint32_t local_info[2]  = { my_lid, my_qpn };
    uint32_t remote_info[2] = { 0, 0 };
    int peer = (rank == src_rank) ? dst_rank : src_rank;

    MPI_Sendrecv(local_info,  2, MPI_UINT32_T, peer, 0,
                 remote_info, 2, MPI_UINT32_T, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    uint32_t remote_lid = remote_info[0];
    uint32_t remote_qpn = remote_info[1];

    if (rank == src_rank)
        fprintf(stderr, "Rank %d: local LID=%u QPN=%u  <->  remote LID=%u QPN=%u\n",
                rank, my_lid, my_qpn, remote_lid, remote_qpn);

    /* 8. QP transition: RESET -> INIT */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = 1;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE  |
                               IBV_ACCESS_REMOTE_READ  |
                               IBV_ACCESS_REMOTE_WRITE;
        ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                            IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS);
        if (ret) { fprintf(stderr, "Rank %d: QP->INIT failed (%d)\n", rank, ret); exit(1); }
    }

    /* 9. QP transition: INIT -> RTR */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state              = IBV_QPS_RTR;
        attr.path_mtu              = IBV_MTU_4096;   /* matches hw active_mtu */
        attr.dest_qp_num           = remote_qpn;
        attr.rq_psn                = 0;
        attr.max_dest_rd_atomic    = 1;
        attr.min_rnr_timer         = 12;
        attr.ah_attr.is_global     = 0;
        attr.ah_attr.dlid          = (uint16_t)remote_lid;
        attr.ah_attr.sl            = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num      = 1;
        ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret) { fprintf(stderr, "Rank %d: QP->RTR failed (%d)\n", rank, ret); exit(1); }
    }

    /* 10. QP transition: RTR -> RTS */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state      = IBV_QPS_RTS;
        attr.timeout        = 14;
        attr.retry_cnt      = 7;
        attr.rnr_retry      = 7;   /* infinite RNR retry (backed by barrier below) */
        attr.sq_psn         = 0;
        attr.max_rd_atomic  = 1;
        ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret) { fprintf(stderr, "Rank %d: QP->RTS failed (%d)\n", rank, ret); exit(1); }
    }

    /* 11. Pre-post a receive so we're ready for the first message */
    post_eager_recv();

    /* 12. Barrier: both sides must have receives posted before anyone sends */
    MPI_Barrier(MPI_COMM_WORLD);
}

/* ================================================================
 * TASK 1  –  EAGER PROTOCOL
 *
 * Copy user data into pre-registered 128KB buffer in chunks,
 * IBV_WR_SEND each chunk (channel semantics).
 * ================================================================ */
void myverbs_send_eager(char *buf, int size, int target_rank)
{
    int offset = 0;
    while (offset < size) {
        int chunk = size - offset;
        if (chunk > EAGER_BUF_SIZE) chunk = EAGER_BUF_SIZE;

        /* copy user data -> pre-pinned send buffer */
        memcpy(eager_send_buf, buf + offset, chunk);

        struct ibv_sge sge;
        sge.addr   = (uint64_t)eager_send_buf;
        sge.length = (uint32_t)chunk;
        sge.lkey   = eager_send_mr->lkey;

        struct ibv_send_wr wr, *bad_wr;
        memset(&wr, 0, sizeof(wr));
        wr.opcode     = IBV_WR_SEND;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        int ret = ibv_post_send(qp, &wr, &bad_wr);
        if (ret) {
            fprintf(stderr, "Rank %d: eager send post failed (%d)\n", my_rank, ret);
            exit(1);
        }

        /* wait for send completion on send_cq */
        poll_cq(send_cq, "eager_send");

        offset += chunk;
    }
}

void myverbs_recv_eager(char *buf, int size, int source_rank)
{
    int offset = 0;
    while (offset < size) {
        int chunk = size - offset;
        if (chunk > EAGER_BUF_SIZE) chunk = EAGER_BUF_SIZE;

        /* wait for receive completion on recv_cq */
        poll_cq(recv_cq, "eager_recv");

        /* copy from pre-pinned recv buffer -> user buffer */
        memcpy(buf + offset, eager_recv_buf, chunk);

        /* immediately repost the receive buffer for the next chunk */
        post_eager_recv();

        offset += chunk;
    }
}

/* ================================================================
 * TASK 2  –  RENDEZVOUS PROTOCOL
 *
 * Sender: register user buffer, send handle (addr+rkey) via channel
 *         semantics, then wait for the receiver to RDMA-Write-With-
 *         Immediate the data and receive the completion notification.
 *
 * Per the assignment: "Once the handle (rkey and address) is ready,
 * you can do then do RDMA write with immediate to perform the
 * transfer."
 *
 * Flow:
 *   Sender                           Receiver
 *   ------                           --------
 *   reg_mr(user_buf)
 *   SEND handle  -------->          recv handle
 *   wait recv_cq (imm)              reg_mr(user_buf)
 *                <--------          RDMA_WRITE_WITH_IMM into sender's buf
 *   got imm data = done             wait send_cq (write done)
 *   dereg_mr                        SEND 0-byte sync  -------->
 *                                   wait send_cq (sync done)
 *                                   dereg_mr
 *   wait recv_cq (sync)
 *   repost recv
 *   dereg_mr
 *
 * Actually, let's simplify the rendezvous to be clean and correct:
 *
 *   Sender:
 *     1) Register user buf with REMOTE_WRITE access
 *     2) SEND {addr, rkey, size} to receiver  (channel)
 *     3) Poll send_cq (handle sent)
 *     4) Poll recv_cq (wait for 0-byte ACK from receiver = transfer done)
 *     5) Repost recv buffer
 *     6) Deregister MR
 *
 *   Receiver:
 *     1) Poll recv_cq (get handle)
 *     2) Repost recv buffer (for future use)
 *     3) RDMA WRITE WITH IMMEDIATE: write user data into sender's registered buffer
 *        The immediate data acts as an implicit notification to the sender
 *     4) Poll send_cq (write completed)
 *     5) SEND 0-byte ACK to sender (so sender can deregister safely)
 *     6) Poll send_cq (ack sent)
 *
 * Wait, RDMA Write with Immediate puts a recv completion on the
 * *remote* side. So the sender will get a recv_cq completion when
 * the write arrives. But the receiver also needs to know the sender's
 * buffer addr/rkey. Let me re-think this to match the assignment exactly:
 *
 * The assignment says the SENDER sends the handle, and the RECEIVER
 * does RDMA Write with Immediate to push data to the sender. This means
 * the receiver has the data and writes it remotely.
 *
 * But in the ping-pong benchmark, the sender has the data (src buf)
 * and the receiver needs the data (dst buf). So actually we need:
 *
 *   RECEIVER registers its dst buffer with REMOTE_WRITE, sends handle
 *   SENDER does RDMA Write with Immediate into receiver's buffer.
 *
 * OR, we keep the flow from the sender side: the sender tells the
 * receiver "here is my buffer, come get it" and the receiver does
 * RDMA READ. But the assignment says "RDMA write with immediate."
 *
 * Let me re-read: "Once the handle (rkey and address) is ready,
 * you can do then do RDMA write with immediate to perform the transfer."
 *
 * Most natural interpretation for rendezvous with RDMA Write:
 *   - Receiver advertises its buffer (where data should land)
 *   - Sender does RDMA Write with Immediate into that buffer
 *   - The Immediate generates a recv completion on the receiver side
 *     to notify completion
 *
 * BUT in a ping-pong, roles flip. On the "send" call, we are the
 * initiator. On "recv" we are the target. So:
 *
 *   myverbs_send_rendezvous(buf, size):
 *     - Has the data to send
 *     - Needs to know WHERE to write it
 *     - Wait for receiver's handle (recv_cq)
 *     - RDMA Write with Immediate to receiver's buffer
 *     - Poll send_cq
 *
 *   myverbs_recv_rendezvous(buf, size):
 *     - Register buf for REMOTE_WRITE
 *     - Send handle to sender (channel)
 *     - Poll recv_cq for Immediate notification (data arrived)
 *     - Deregister
 *
 * This is cleaner. Let's implement this.
 * ================================================================ */

void myverbs_send_rendezvous(char *buf, int size, int target_rank)
{
    /*
     * Step 1: Wait for the receiver's handle (receiver registered its
     *         buffer and sent us {addr, rkey}).
     */
    poll_cq(recv_cq, "rndv_send: recv handle");

    struct mr_info remote_info;
    memcpy(&remote_info, eager_recv_buf, sizeof(remote_info));

    /* Repost receive for next message (sync ack or next iteration) */
    post_eager_recv();

    /*
     * Step 2: Register our source buffer for LOCAL_WRITE (needed as
     *         scatter/gather source).
     */
    struct ibv_mr *src_mr = ibv_reg_mr(pd, buf, size,
                                       IBV_ACCESS_LOCAL_WRITE);
    if (!src_mr) {
        fprintf(stderr, "Rank %d: rndv send reg_mr failed\n", my_rank);
        exit(1);
    }

    /*
     * Step 3: RDMA Write With Immediate into receiver's buffer.
     *         The immediate data generates a RECV completion on the
     *         receiver's recv_cq, notifying that the write is done.
     */
    struct ibv_sge sge;
    sge.addr   = (uint64_t)buf;
    sge.length = (uint32_t)size;
    sge.lkey   = src_mr->lkey;

    struct ibv_send_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.opcode               = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.sg_list              = &sge;
    wr.num_sge              = 1;
    wr.send_flags           = IBV_SEND_SIGNALED;
    wr.imm_data             = htonl(size);
    wr.wr.rdma.remote_addr  = remote_info.addr;
    wr.wr.rdma.rkey         = remote_info.rkey;

    int ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        fprintf(stderr, "Rank %d: rndv RDMA write post failed (%d)\n", my_rank, ret);
        exit(1);
    }

    /* Step 4: Wait for the write to complete locally */
    poll_cq(send_cq, "rndv_send: rdma write");

    /* Step 5: Deregister */
    ibv_dereg_mr(src_mr);
}

void myverbs_recv_rendezvous(char *buf, int size, int source_rank)
{
    /*
     * Step 1: Register our receive buffer with REMOTE_WRITE access
     *         so the sender can RDMA-write into it.
     */
    struct ibv_mr *dst_mr = ibv_reg_mr(pd, buf, size,
                                       IBV_ACCESS_LOCAL_WRITE  |
                                       IBV_ACCESS_REMOTE_WRITE);
    if (!dst_mr) {
        fprintf(stderr, "Rank %d: rndv recv reg_mr failed\n", my_rank);
        exit(1);
    }

    /*
     * Step 2: Send our handle {addr, rkey} to the sender via channel
     *         semantics (IBV_WR_SEND using pre-registered eager buffer).
     */
    struct mr_info info;
    info.addr = (uint64_t)buf;
    info.rkey = dst_mr->rkey;
    info.size = (uint32_t)size;
    info.pad  = 0;
    memcpy(eager_send_buf, &info, sizeof(info));

    struct ibv_sge sge;
    sge.addr   = (uint64_t)eager_send_buf;
    sge.length = sizeof(info);
    sge.lkey   = eager_send_mr->lkey;

    struct ibv_send_wr wr, *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.opcode     = IBV_WR_SEND;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.send_flags = IBV_SEND_SIGNALED;

    int ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        fprintf(stderr, "Rank %d: rndv handle send failed (%d)\n", my_rank, ret);
        exit(1);
    }

    /* Wait for the handle to be sent */
    poll_cq(send_cq, "rndv_recv: send handle");

    /*
     * Step 3: Wait for RDMA Write with Immediate to arrive.
     *         This generates a RECV completion with the immediate data.
     *         We need a receive WR posted for the immediate — the
     *         eager recv buffer is already posted from init / previous repost.
     */
    poll_cq(recv_cq, "rndv_recv: wait rdma_write_imm");

    /* Repost receive buffer for next operation */
    post_eager_recv();

    /* Step 4: Data is now in buf (the sender wrote directly). Deregister. */
    ibv_dereg_mr(dst_mr);
}

/* ================================================================
 * Cleanup
 * ================================================================ */
void myverbs_finalize(void)
{
    if (eager_send_mr) ibv_dereg_mr(eager_send_mr);
    if (eager_recv_mr) ibv_dereg_mr(eager_recv_mr);
    if (qp)            ibv_destroy_qp(qp);
    if (send_cq)       ibv_destroy_cq(send_cq);
    if (recv_cq)       ibv_destroy_cq(recv_cq);
    if (pd)            ibv_dealloc_pd(pd);
    if (ctx)           ibv_close_device(ctx);
    free(eager_send_buf);
    free(eager_recv_buf);
}