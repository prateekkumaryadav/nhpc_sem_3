/*
 * tcp_bench.c
 * code.c
 *
 * Simple TCP socket based Latency / Bandwidth micro-benchmark.
 *
 * Runs a ping-pong test for message sizes 1 byte .. 4 MiB (doubling each
 * step). For every size:
 *      - a set of WARMUP iterations are done first and NOT measured
 *      - a set of measured iterations follow, and the average one-way
 *        latency (usecs) and bandwidth (MB/s) are printed
 *
 * Large messages are not sent in one shot -- they are broken into chunks
 * of CHUNK_SIZE bytes (configurable through the environment variable
 * TCP_BENCH_CHUNK_SIZE, default 16384) and sent/received chunk by chunk,
 * exactly like real world large-message transfers.
 *
 * Build:
 *      gcc -O2 -o tcp_bench tcp_bench.c
 *      gcc -O2 -o tcp_bench code.c
 *
 * Run:
 *      Server:  ./tcp_bench -s <port>
 *      Client:  ./tcp_bench -c <server_ip> <port>
 *
 * Env:
 *      TCP_BENCH_CHUNK_SIZE   chunk size (bytes) used to break a message
 *                              into multiple send()/recv() calls.
 *                              default = 16384 (16 KiB)
 * 
 * 
 * 
 * gcc -O2 -std=c99 -o tcp_bench code.c
 * 
 * [student4@master MT2025093]$ gcc -O2 -std=gnu99 -o tcp_bench test.c
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define MIN_MSG_SIZE   1
#define MAX_MSG_SIZE   (4 * 1024 * 1024)   /* 4 MiB */
#define DEFAULT_CHUNK  (16 * 1024)         /* 16 KiB */

/* ------------------------------------------------------------------ */
/* Utility                                                             */
/* ------------------------------------------------------------------ */

static size_t g_chunk_size = DEFAULT_CHUNK;

static double now_usec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* Send 'len' bytes from buf, split into chunks of g_chunk_size bytes. */
static int send_chunked(int sock, const char *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        size_t want = (len - sent) < g_chunk_size ? (len - sent) : g_chunk_size;
        ssize_t n = send(sock, buf + sent, want, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

/* Receive exactly 'len' bytes into buf, split into chunks of g_chunk_size. */
static int recv_chunked(int sock, char *buf, size_t len)
{
    size_t got = 0;
    while (got < len) {
        size_t want = (len - got) < g_chunk_size ? (len - got) : g_chunk_size;
        ssize_t n = recv(sock, buf + got, want, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        got += (size_t)n;
    }
    return 0;
}

/* Choose iteration / warmup counts based on message size, so the whole
 * sweep finishes in reasonable time for large sizes while still getting
 * statistically stable numbers for small sizes. */
static void get_iter_counts(size_t size, int *iterations, int *warmup)
{
    if (size <= 8 * 1024) {
        *iterations = 1000;
        *warmup     = 100;
    } else if (size <= 256 * 1024) {
        *iterations = 200;
        *warmup     = 20;
    } else {
        *iterations = 50;
        *warmup     = 5;
    }
}

static void set_socket_opts(int sock)
{
    int one = 1;
    /* Disable Nagle's algorithm -- essential for accurate latency numbers */
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

/* ------------------------------------------------------------------ */
/* Server side                                                        */
/* ------------------------------------------------------------------ */

static void run_server(const char *ip, int port)
{
    int listen_fd, conn_fd;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    int one = 1;

    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); exit(1); }

    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (ip == NULL || strlen(ip) == 0 || strcmp(ip, "0.0.0.0") == 0)
        addr.sin_addr.s_addr = INADDR_ANY;
    else
        inet_pton(AF_INET, ip, &addr.sin_addr);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(1);
    }

    if (listen(listen_fd, 1) < 0) { perror("listen"); exit(1); }

    printf("Server listening on port %d ...\n", port);
    fflush(stdout);

    conn_fd = accept(listen_fd, (struct sockaddr *)&addr, &addr_len);
    if (conn_fd < 0) { perror("accept"); exit(1); }

    set_socket_opts(conn_fd);
    printf("Client connected: %s\n", inet_ntoa(addr.sin_addr));
    fflush(stdout);

    char *buf = malloc(MAX_MSG_SIZE);
    if (!buf) { perror("malloc"); exit(1); }

    for (size_t size = MIN_MSG_SIZE; size <= MAX_MSG_SIZE; size *= 2) {
        int iterations, warmup;
        get_iter_counts(size, &iterations, &warmup);
        int total = iterations + warmup;

        for (int i = 0; i < total; i++) {
            /* Echo server: receive the message, then send it right back */
            if (recv_chunked(conn_fd, buf, size) < 0) {
                fprintf(stderr, "server: recv failed at size %zu iter %d\n", size, i);
                free(buf);
                close(conn_fd);
                close(listen_fd);
                exit(1);
            }
            if (send_chunked(conn_fd, buf, size) < 0) {
                fprintf(stderr, "server: send failed at size %zu iter %d\n", size, i);
                free(buf);
                close(conn_fd);
                close(listen_fd);
                exit(1);
            }
        }
    }

    free(buf);
    close(conn_fd);
    close(listen_fd);
}

/* ------------------------------------------------------------------ */
/* Client side                                                        */
/* ------------------------------------------------------------------ */

static void run_client(const char *ip, int port)
{
    int sock;
    struct sockaddr_in addr;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); exit(1); }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid IP address: %s\n", ip);
        exit(1);
    }

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect"); exit(1);
    }

    set_socket_opts(sock);

    printf("Connected to server %s:%d\n", ip, port);
    printf("Chunk size (TCP_BENCH_CHUNK_SIZE) = %zu bytes\n\n", g_chunk_size);
    printf("%-12s%20s%22s\n", "Msg size", "Time (usecs)", "Bandwidth (MB/s)");
    fflush(stdout);

    char *sbuf = malloc(MAX_MSG_SIZE);
    char *rbuf = malloc(MAX_MSG_SIZE);
    if (!sbuf || !rbuf) { perror("malloc"); exit(1); }
    memset(sbuf, 0xAB, MAX_MSG_SIZE);

    for (size_t size = MIN_MSG_SIZE; size <= MAX_MSG_SIZE; size *= 2) {
        int iterations, warmup;
        get_iter_counts(size, &iterations, &warmup);

        /* Warmup: same send/recv pattern, but not timed */
        for (int i = 0; i < warmup; i++) {
            if (send_chunked(sock, sbuf, size) < 0 ||
                recv_chunked(sock, rbuf, size) < 0) {
                fprintf(stderr, "client: warmup failed at size %zu\n", size);
                exit(1);
            }
        }

        /* Timed measurement */
        double t_start = now_usec();
        for (int i = 0; i < iterations; i++) {
            if (send_chunked(sock, sbuf, size) < 0 ||
                recv_chunked(sock, rbuf, size) < 0) {
                fprintf(stderr, "client: measured iter failed at size %zu\n", size);
                exit(1);
            }
        }
        double t_end = now_usec();

        double total_usec   = t_end - t_start;
        /* round-trip covers "send + reply", so one-way latency = RTT / 2 */
        double avg_rtt_usec  = total_usec / iterations;
        double one_way_usec  = avg_rtt_usec / 2.0;

        double one_way_sec  = one_way_usec / 1e6;
        double bandwidth_mb = (one_way_sec > 0.0)
                               ? ((double)size / (1024.0 * 1024.0)) / one_way_sec
                               : 0.0;

        printf("%-12zu%20.2f%22.2f\n", size, one_way_usec, bandwidth_mb);
        fflush(stdout);
    }

    free(sbuf);
    free(rbuf);
    close(sock);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  Server:  %s -s <port> [bind_ip]\n"
        "  Client:  %s -c <server_ip> <port>\n\n"
        "Environment:\n"
        "  TCP_BENCH_CHUNK_SIZE   chunk size in bytes used to split large\n"
        "                         messages into multiple send/recv calls\n"
        "                         (default: %d)\n",
        prog, prog, DEFAULT_CHUNK);
}

int main(int argc, char *argv[])
{
    const char *chunk_env = getenv("TCP_BENCH_CHUNK_SIZE");
    if (chunk_env != NULL) {
        long v = atol(chunk_env);
        if (v > 0)
            g_chunk_size = (size_t)v;
    }

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-s") == 0) {
        if (argc < 3) { usage(argv[0]); return 1; }
        int port = atoi(argv[2]);
        const char *bind_ip = (argc >= 4) ? argv[3] : "0.0.0.0";
        run_server(bind_ip, port);
    } else if (strcmp(argv[1], "-c") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        const char *ip = argv[2];
        int port = atoi(argv[3]);
        run_client(ip, port);
    } else {
        usage(argv[0]);
        return 1;
    }

    return 0;
}