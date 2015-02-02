import numpy as np
import pylab as pl


if __name__ == '__main__':
    img = np.zeros((100, 200))

    for i in range(100):
        img[i] = img[i] + 0.1 * i

    pl.imshow(img)
    pl.show()
