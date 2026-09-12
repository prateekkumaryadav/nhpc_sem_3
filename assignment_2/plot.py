import matplotlib.pyplot as plt

# Data extracted from benchmark output
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
send_recv = [8.82, 8.82, 8.80, 9.19, 8.94, 19.21, 8.87, 8.98, 8.85, 8.87, 8.85, 12.06, 11.56, 11.60, 12.98, 15.87, 21.69, 33.27, 56.46, 102.83, 195.61, 381.49, 761.67]
rdma_write = [8.84, 9.10, 8.84, 8.86, 9.08, 9.20, 8.90, 9.19, 8.81, 8.80, 8.81, 8.81, 8.81, 8.81, 8.81, 11.02, 14.86, 25.74, 48.29, 94.67, 187.72, 373.31, 744.34]
mr_reg = [31.78, 31.28, 30.62, 30.60, 29.73, 29.12, 28.63, 28.26, 27.64, 27.20, 27.06, 27.69, 26.76, 26.90, 27.18, 27.39, 28.14, 29.51, 32.23, 44.94, 69.61, 109.20, 184.50]
mr_dereg = [33.14, 33.06, 32.39, 32.28, 31.66, 31.22, 30.99, 30.99, 30.33, 29.93, 29.81, 29.70, 29.50, 29.49, 29.73, 29.85, 29.87, 30.41, 30.97, 35.98, 43.12, 53.03, 71.05]

plt.figure(figsize=(10, 6))
plt.plot(sizes, send_recv, label='Send/Recv', marker='o')
plt.plot(sizes, rdma_write, label='RDMA Write', marker='s')
plt.plot(sizes, mr_reg, label='MR Register', marker='^')
plt.plot(sizes, mr_dereg, label='MR Deregister', marker='d')

# Log-log scale to handle the massive range in message sizes
plt.xscale('log', base=2)
plt.yscale('log')

plt.xlabel('Message Size (Bytes)')
plt.ylabel('Latency (us)')
plt.title('RDMA Latency vs Message Size')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Save as PNG or display
plt.savefig('rdma_benchmark_plot.png')
plt.show()