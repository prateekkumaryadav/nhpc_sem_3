#ifndef MYVERBS_H
#define MYVERBS_H

void myverbs_init(int rank, int src_rank, int dst_rank);
void myverbs_send_eager(char *buf, int size, int target_rank);
void myverbs_recv_eager(char *buf, int size, int source_rank);
void myverbs_send_rendezvous(char *buf, int size, int target_rank);
void myverbs_recv_rendezvous(char *buf, int size, int source_rank);
void myverbs_finalize(void);

#endif