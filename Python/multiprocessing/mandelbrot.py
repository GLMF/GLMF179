import numpy as np
import pylab as pl
from multiprocessing import Process, Queue

def mandelbrot(x, y, iterations):
    c = complex(x, y)
    Z = complex(0, 0)

    for i in range(iterations):
        Z = Z ** 2 + c
        if abs(Z) >= 2:
            return i

    return 255


def fractal(x_axis, y_axis, iterations, pile, pos, height=500, width=750):
    img = np.zeros((height, width))

    pixel_size_x = (x_axis['max'] - x_axis['min']) / width
    pixel_size_y = (y_axis['max'] - y_axis['min']) / height

    for x in range(width):
        real_x = x_axis['min'] + x * pixel_size_x
        for y in range(height):
            real_y = y_axis['min'] + y * pixel_size_y
            img[y, x] = mandelbrot(real_x, real_y, iterations)

    pile.put((img, pos))


if __name__ == '__main__':
    pile = Queue()
    t = []
    for i in range(6):
        t.append(Process(target=fractal, args=({'min': -2 + 0.5 * i, 'max': -1.5 + 0.5 * i}, {'min': -1, 'max': 1}, 100, pile, i, 500, 125)))

    for i in range(6):
        t[i].start()

    image_tmp = [None] * 6
    p = [None] * 6
    for i in range(6):
        image_tmp[i], p[i] = pile.get()
    img = [None] * 6
    for i in range(6):
        img[p[i]] = image_tmp[i]
    image = np.hstack((img[0], img[1]))
    for i in range(2, 6):
        image = np.hstack((image, img[i]))

    for i in range(6):
        t[i].join()

    pl.imshow(image, cmap='prism')
    pl.show()
