import numpy as np
import pylab as pl
import numba


@numba.autojit
def mandelbrot(x, y, iterations):
    c = complex(x, y)
    Z = complex(0, 0)

    for i in range(iterations):
        Z = Z ** 2 + c
        if abs(Z) >= 2:
            return i

    return 255


@numba.autojit
def fractal(x_axis, y_axis, iterations, height=500, width=750):
    img = np.zeros((height, width))

    pixel_size_x = (x_axis['max'] - x_axis['min']) / width
    pixel_size_y = (y_axis['max'] - y_axis['min']) / height

    for x in range(width):
        real_x = x_axis['min'] + x * pixel_size_x
        for y in range(height):
            real_y = y_axis['min'] + y * pixel_size_y
            img[y, x] = mandelbrot(real_x, real_y, iterations)

    return img


if __name__ == '__main__':
    image = fractal({'min': -2, 'max': 1}, {'min': -1, 'max': 1}, 100, 500, 750)
    pl.imshow(image, cmap='prism')
    pl.show()
