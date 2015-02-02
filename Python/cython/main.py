import mandelbrot
import pylab as pl

if __name__ == '__main__':
    image = mandelbrot.fractal({'min': -2, 'max': 1}, {'min': -1, 'max': 1}, 100)
    pl.imshow(image, cmap='prism')
    pl.show()
