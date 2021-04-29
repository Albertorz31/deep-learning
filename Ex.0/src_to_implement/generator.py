import json
import numpy as np
import matplotlib.pyplot as plt
import os.path
from PIL import Image


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = os.path.join(file_path)
        self.label_path = os.path.join(label_path)
        self.batch_size = int(batch_size)
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.how_many_call = 0

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.labels = None
        self.batch = None
        # TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)]
        images.sort()
        total_images_count = len(images)

        beginning = (self.how_many_call * self.batch_size) % total_images_count

        if (beginning + self.batch_size) > total_images_count:
            taken_images = images[beginning:]
            how_many_remaning = self.batch_size - len(taken_images)
            taken_images += images[:how_many_remaning]
        else:
            taken_images = images[beginning:beginning + self.batch_size]

        f = open(self.label_path)
        data = json.load(f)

        batch = []
        labels = []

        for image in taken_images:
            image_array = np.load(image)
            # We interpolation the data of  the images using the fuction resize regarding (height, width and channel)
            image_array_resized = np.array(Image.fromarray(image_array).resize(self.image_size[:2]))
            image_array_resized = self.augment(image_array_resized)
            batch.append(image_array_resized)
            # Append the labels of array
            label_of_the_image = data[image.split("/")[-1][:-4]]
            labels.append(label_of_the_image)

        self.how_many_call += 1

        labels = np.asarray(labels)
        batch = np.asarray(batch)
        self.labels = labels
        self.batch = batch

        if self.shuffle:
            self.batch, self.labels = self.shuffle_it()

        if self.rotation:
            for i in range(len(self.batch)):
                img = self.batch[i]
                self.batch[i] = self.augment(img)

        return self.batch, self.labels

    def shuffle_it(self):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        self.batch = self.batch[indices]
        self.labels = self.labels[indices]
        return self.batch, self.labels

    def augment(self, img):
        if self.mirroring:
            img = np.fliplr(img)

        if self.rotation:
            # Rotate images by 90, 180, 270 degrees.
            value = np.random.randint(0, 3)
            if value == 0:
                img = np.rot90(img, k=1)
            elif value == 1:
                img = np.rot90(img, k=2)
            elif value == 2:
                img = np.rot90(img, k=3)
            else:
                print("There is a problem with rotation")
                raise Exception
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        f = open(self.label_path)
        data = json.load(f)

        name = self.class_dict[data[x]]
        return name

    def show(self):
        fig = plt.figure(figsize=(8, 8))
        primo = 1
        numbers = []
        columns = 0
        rows = 0

        for n in range(2, self.batch_size):
            if self.batch_size % n == 0:
                primo = 0

        if primo == 0:
            for divisor in range(1, self.batch_size):
                if (self.batch_size % divisor) == 0:
                    numbers.append(divisor)
        else:
            aux = self.batch_size + 1
            for divisor in range(1, aux):
                if (aux % divisor) == 0:
                    numbers.append(divisor)

        list_sort = sorted(numbers)
        half = int(len(list_sort) / 2)

        if len(list_sort) % 2 == 0:
            columns = int((list_sort[half] + list_sort[half - 1]) / 2)
        else:
            columns = list_sort[int(half)]

        rows = int(self.batch_size / columns)

        ax = []
        for i in range(columns * rows):
            img = self.batch[i]
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(self.class_dict[self.labels[i]], fontdict={'size': 18})  # set title
            plt.axis('off')
            plt.imshow(img)
        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.385, wspace=0.235)
        plt.show()
