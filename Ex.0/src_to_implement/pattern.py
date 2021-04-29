import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = int(resolution)
        self.tile_size = int(tile_size)
        self.tile_count = int(self.resolution / self.tile_size)
        self.dividable = int(resolution / 2)
        self.output = []

    def draw(self):
        if self.dividable % self.tile_size != 0:
            print("the resolution is not divisible by 2")
            return

        tile_zero = np.zeros((self.tile_size, self.tile_size))
        tile_one = np.ones((self.tile_size, self.tile_size))

        row_zero = np.concatenate((tile_zero, tile_one), axis=1)
        row_one = np.concatenate((tile_one, tile_zero), axis=1)

        row_zero = np.tile(row_zero, self.tile_count // 2)
        row_one = np.tile(row_one, self.tile_count // 2)

        row_tile = np.concatenate((row_zero, row_one), axis=0)

        pattern = np.tile(row_tile, (self.tile_count // 2, 1))

        self.output = pattern

        return pattern.copy()

    def show(self):
        plt.imshow(self.output, cmap='binary_r')
        plt.show()


class Circle:
    def __init__(self, res, radius, pos):
        self.res = int(res)
        self.radius = int(radius)
        self.pos = tuple(pos)
        self.output = []

    def draw(self):
        x, y = np.mgrid[:self.res, :self.res]

        dist_from_center = (x - self.pos[1]) ** 2 + (y - self.pos[0]) ** 2
        circle = dist_from_center <= self.radius ** 2
        self.output = circle

        return circle.copy()

    def show(self):
        plt.imshow(self.output, cmap="binary_r")
        plt.show()


class Spectrum(object):

    def __init__(self, resolution):
        self.resolution = resolution
        self.solution = None

    def draw(self):
        self.solution = self.get_pic(self.resolution)

    def show(self):
        plt.imshow(self.solution)
        plt.show()

    # used in get_pic for computing gradients for each color
    def compute_color(self, i, j, color):
        size = len(i)
        red = color[0]
        gre = color[1]
        blu = color[2]

        # how far is it from the corner? used for coloring;
        d = np.sqrt(j ** 2 + i ** 2) / (np.sqrt(2) * size)

        # color of pixes
        r = d + red * (1 - d)
        g = d + gre * (1 - d)
        b = d + blu * (1 - d)

        # fix overflows
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)

        return (np.array([r, g, b])).T

    # size -> picture function
    # combines together colors from compute_color
    def get_pic(self, size):
        x = np.arange(size)
        y = np.arange(size)
        xx, yy = np.meshgrid(x, y)

        B = np.flip(self.compute_color(xx, yy, [0, 0, 255]), axis=0)
        R = np.flip(np.flip(self.compute_color(xx, yy, [255, 0, 0]), axis=0), axis=1)
        Y = np.flip(self.compute_color(xx, yy, [255, 255, 0]), axis=1)
        b = (self.compute_color(xx, yy, [0, 255, 255]))

        return np.flip(R * B * b * Y * np.sqrt(2), axis=0)
