cimport cython
import numpy as np
cimport numpy as np


def mandelbrot(double x, double y, int iterations):
    cdef:
        int i
        complex c, z

    c = complex(x, y)
    Z = complex(0, 0)

    for i in range(iterations):
        Z = Z ** 2 + c
        if abs(Z) >= 2:
            return i

    return 255


def fractal(x_axis, y_axis, int iterations, int height=500, int width=750):
    cdef:
        double pixel_size_x, pixel_size_y
        double real_x, real_y
        int x, y
        np.ndarray img

    img = np.zeros((height, width))

    pixel_size_x = (x_axis['max'] - x_axis['min']) / width
    pixel_size_y = (y_axis['max'] - y_axis['min']) / height

    for x in range(width):
        real_x = x_axis['min'] + x * pixel_size_x
        for y in range(height):
            real_y = y_axis['min'] + y * pixel_size_y
            img[y, x] = mandelbrot(real_x, real_y, iterations)

    return img
