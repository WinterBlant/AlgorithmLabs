import numpy as np
import time


def generate_random_matrix(length):
    rng = np.random.default_rng()
    return rng.integers(10, size=(length, length))


elapsed_time = []
for n in range(1, 2001):
    matrixA = generate_random_matrix(n)
    matrixB = generate_random_matrix(n)
    t0 = time.perf_counter()
    matrixC = np.matmul(matrixA, matrixB)
    elapsed_time.append(time.perf_counter() - t0)
    print("finished " + str(n) + " x " + str(n))
with open("matrix.txt", 'a') as f:
    f.write(str(elapsed_time) + '\n')
