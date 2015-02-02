import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
        #include <stdio.h>
        __device__ int *add(int *a, int *b)
       {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            __shared__ int result[400];

            result[index] = a[index] + b[index];
            return result;
        }

        __global__ void run(int *a, int *b, int *result)
        {
            result = add(a, b);
        }
""")

run = mod.get_function('run')

if __name__ == '__main__':
    a = np.ones(400).astype(np.int32)
    b = np.ones(400).astype(np.int32)
    result = np.zeros(400).astype(np.int32)

    run(cuda.In(a), cuda.In(b), cuda.Out(result), block=(100, 1, 1), grid=(4, 1))

    print(result)
