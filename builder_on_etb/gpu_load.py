import numpy as np
from numba import cuda
import time

# Define a simple kernel function to add two arrays
@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Initialize data
N = 1024*1024*1024*5  # Size of the arrays
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

# Transfer data to the GPU
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
c_device = cuda.to_device(c)

# Define the number of threads and blocks
threads_per_block = 256
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

# Run the kernel in a loop to stress the GPU
start_time = time.time()
for _ in range(1000):  # Adjust the number of iterations as needed
    add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, c_device)

# Copy the result back to the host
c = c_device.copy_to_host()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")