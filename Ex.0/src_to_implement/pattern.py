import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self,res,tile):
        self.res = int(res)
        self.tile = int(tile)
        self.tile_count = int((res/tile))
        self.dividable = int(res/2)
        self.output = []


    def draw(self):
        if self.dividable % 2 != 0:
            print("the resolution is not divisible by 2")
            return

        black = np.zeros((self.tile, self.tile))
        white = np.ones((self.tile, self.tile))
        row_b = [np.concatenate([black, white] * int(self.res / self.tile / 2), axis=1)]
        row_w = [np.concatenate([white, black] * int(self.res / self.tile / 2), axis=1)]
        row_final = np.concatenate((row_b + row_w) * int(self.res / self.tile / 2), axis=0)

        self.output = row_final
        return row_final.copy()

    def show(self):
        plt.imshow(self.output, cmap = "gray")
        plt.show()


class Circle:
    def __init__(self, res, radius, pos):
        self.res = int(res)
        self.radius = int(radius)
        self.pos = tuple(pos)
        self.output = []

    def draw(self):

        x_axis = np.linspace(0, self.res)
        y_axis = np.linspace(0, self.res)

        x, y = np.meshgrid(x_axis, y_axis)

        dist_from_center = np.sqrt((x- self.pos[0])**2 + (y-self.pos[1])**2)
        circle = dist_from_center <= self.radius
        self.output = circle

        return circle.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
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
        

















