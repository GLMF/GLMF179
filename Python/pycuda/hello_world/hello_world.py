import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
        #include <stdio.h>
        __global__ void hello_world(void)
        {
            printf("Hello world from block (%d, %d) thread (%d, %d, %d)\\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
        }
""")

hello_world = mod.get_function('hello_world')

hello_world(block=(1, 2, 3), grid=(2, 1))
