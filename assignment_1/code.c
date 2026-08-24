/*
 * TCP latency/bandwidth ping-pong benchmark
 * build:  gcc -O2 -o tcp_bench_min tcp_bench_min.c
 * build: gcc -O2 -std=gnu99 -o code code.c
 * server: ./code -s <port>
 * client: ./code -c <ip> <port>
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

#define MAX_SIZE   (4 * 1024 * 1024) // 4MB
#define ITERS      100
#define WARMUP     10

static size_t chunk_size = 65536;

static void send_all(int fd, char *buf, size_t len) { // send/recv exactly 'len' bytes, broken into chunk_size pieces
    size_t off = 0;
    while (off < len) {
        size_t n = (len - off < chunk_size) ? len - off : chunk_size;
        ssize_t s = send(fd, buf + off, n, 0);
        if (s <= 0) { perror("send"); exit(1); }
        off += s;
    }
}

static void recv_all(int fd, char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        size_t n = (len - off < chunk_size) ? len - off : chunk_size;
        ssize_t s = recv(fd, buf + off, n, 0);
        if (s <= 0) { perror("recv"); exit(1); }
        off += s;
    }
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void run_server(int port) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = { .sin_family = AF_INET,
                                 .sin_port = htons(port),
                                 .sin_addr.s_addr = INADDR_ANY };
    bind(ls, (struct sockaddr *)&addr, sizeof(addr));
    listen(ls, 1);
    printf("Listening on port %d...\n", port);

    int fd = accept(ls, NULL, NULL);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    char *buf = malloc(MAX_SIZE);
    for (size_t size = 1; size <= MAX_SIZE; size *= 2)
        for (int i = 0; i < WARMUP + ITERS; i++) {
            recv_all(fd, buf, size);   // echo the message back
            send_all(fd, buf, size);
        }

    free(buf);
    close(fd);
    close(ls);
}

static void run_client(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
    inet_pton(AF_INET, ip, &addr.sin_addr);
    connect(fd, (struct sockaddr *)&addr, sizeof(addr));

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    printf("%-12s%18s%22s\n", "Msg size", "Time (usecs)", "Bandwidth (MB/s)");

    char *buf = malloc(MAX_SIZE);
    memset(buf, 0xAB, MAX_SIZE);

    for (size_t size = 1; size <= MAX_SIZE; size *= 2) {
        for (int i = 0; i < WARMUP; i++) { // warmup
            send_all(fd, buf, size);
            recv_all(fd, buf, size);
        }

        double t0 = now_us();
        for (int i = 0; i < ITERS; i++) {
            send_all(fd, buf, size);
            recv_all(fd, buf, size);
        }
        double t1 = now_us();

        double one_way_us = (t1 - t0) / ITERS / 2.0;
        double bw_mbs = (size / 1048576.0) / (one_way_us / 1e6);

        printf("%-12zu%18.2f%22.2f\n", size, one_way_us, bw_mbs);
    }

    free(buf);
    close(fd);
}

int main(int argc, char **argv) {
    const char *cs = getenv("CHUNK_SIZE");
    if (cs) chunk_size = atol(cs);

    if (argc >= 3 && strcmp(argv[1], "-s") == 0) {
        run_server(atoi(argv[2]));
    } else if (argc >= 4 && strcmp(argv[1], "-c") == 0) {
        run_client(argv[2], atoi(argv[3]));
    } else {
        fprintf(stderr, "Usage:\n  %s -s <port>\n  %s -c <ip> <port>\n",
                argv[0], argv[0]);
        return 1;
    }
    return 0;
}