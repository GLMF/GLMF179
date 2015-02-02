import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
        #include <stdio.h>
        __global__ void add(int *a, int *b, int *result)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;

            result[index] = a[index] + b[index];
        }
""")

add = mod.get_function('add')

if __name__ == '__main__':
    a = np.ones(400).astype(np.int32)
    b = np.ones(400).astype(np.int32)
    result = np.zeros(400).astype(np.int32)

    add(cuda.In(a), cuda.In(b), cuda.Out(result), block=(100, 1, 1), grid=(4, 1))

    print(result)
