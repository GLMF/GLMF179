from numba import cuda
import numpy as np
import pylab as pl


@cuda.jit('int32(float64, float64, int32)', device=True)
def mandelbrot(x, y, iterations):
    c = complex(x, y)
    Z = complex(0, 0)

    for i in range(iterations):
        Z = Z * Z + c
        if Z.real * Z.real + Z.imag * Z.imag >= 4:
            return i

    return 255

@cuda.autojit
def fractal(x_axis_min, x_axis_max, y_axis_min, y_axis_max, iterations, img):
    height = img.shape[0]
    width = img.shape[1]

    pixel_size_x = (x_axis_max - x_axis_min) / width
    pixel_size_y = (y_axis_max - y_axis_min) / height

    for x in range(width):
        real_x = x_axis_min + x * pixel_size_x
        for y in range(height):
            real_y = y_axis_min + y * pixel_size_y
            img[y, x] = mandelbrot(real_x, real_y, iterations)

if __name__ == '__main__':
    image = np.zeros((500, 750)).astype(np.uint8)
    fractal(-2, 1, -1, 1, 100, image)
    pl.imshow(image, cmap='prism')
    pl.show()
