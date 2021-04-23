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
        print(row_final)
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

        x, y = np.meshgrid[x_axis, y_axis]

        print(x)

        print(y)







