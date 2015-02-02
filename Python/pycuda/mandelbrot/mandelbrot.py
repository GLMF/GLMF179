import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import pylab as pl
from pycuda.elementwise import ElementwiseKernel

mandelbrot_total = ElementwiseKernel(
    """
        int x_axis_min, int x_axis_max, int y_axis_min, int y_axis_max, int height, 
        int width, int iterations, int *img_gpu
    """,
    """
        float Z_real, Z_imag;
        float c_real, c_imag;

        c_real = x_axis_min + (y_axis_max - y_axis_min) * (float) (i % width) / height; 
        c_imag = y_axis_min + (x_axis_max - x_axis_min) * (float) (i / width) / width;
        Z_real = 0;
        Z_imag = 0;
        img_gpu[i] = 255;

        for (int n = 0; n < iterations; n++) 
        {
            float Z_real_old = Z_real;
            Z_real = Z_real * Z_real - Z_imag * Z_imag + c_real;
            Z_imag = 2.0 * Z_real_old * Z_imag + c_imag;

            if (Z_real * Z_real + Z_imag * Z_imag > 4)
            {
                img_gpu[i] = n;
                break;
            }
        }
        """)
 
def fractal(x_axis, y_axis, iterations, height=500, width=750):
    img = np.zeros(height * width).astype(np.int32)

    img_gpu = gpuarray.to_gpu(img)

    mandelbrot_total(np.int16(x_axis['min']), np.int16(x_axis['max']), np.int16(y_axis['min']), 
        np.int16(y_axis['max']), np.int16(height), np.int16(width), np.int16(iterations), img_gpu)

    img_gpu.get(img)
 
    return img.reshape(height, width)

if __name__ == '__main__':
    image = fractal({'min': -2, 'max': 1}, {'min': -1, 'max': 1}, 100)
    pl.imshow(image, cmap='prism')
    pl.show()
