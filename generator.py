import json
import numpy as np
import matplotlib.pyplot as plt
import os.path
from PIL import Image
from random import randrange


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
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.labels = None
        self.batch = None

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        # TODO: implement constructor

    def next(self):
        images = []
        # we need to have the number of images in a batch
        for f in os.listdir(self.file_path):
            images.append(os.path.join(self.file_path, f))

        images.sort()
        total_images_count = len(images)
        index = randrange(total_images_count)

        if index + self.batch_size > total_images_count:
            aux = index - self.batch_size
            select_images = images[aux:index]
        else:
            select_images = images[index:index + self.batch_size]

        batch = []
        labels = []

        f = open(self.label_path)
        data = json.load(f)

        for picture in select_images:
            picture_array = np.load(picture)
            # We interpolation the data of  the images using the fuction resize regarding (height, width and channel)
            image_array_resized = np.array(Image.fromarray(picture_array).resize(self.image_size[:2]))
            batch.append(image_array_resized)
            # Append the labels of array
            label_of_the_image = data[picture.split("/")[-1][:-4]]
            labels.append(label_of_the_image)

        labels = np.asarray(labels)
        batch = np.asarray(batch)
        self.labels = labels
        self.batch = batch
        print(len(self.batch))

        if self.shuffle:
            self.batch, self.labels = self.shuffeF()

        if self.mirroring==True or self.rotation==True:
            for i in range(len(self.batch)):
                img = self.batch[i]
                self.batch[i] = self.augment(img)

        return images, labels

    def shuffeF(self):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        self.batch = self.batch[indices]
        self.labels = self.labels[indices]
        return (self.batch, self.labels)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # Mirror images randomly
        if (self.mirroring==True):
            img = np.fliplr(img)
            return img

        if (self.rotation==True):
            # Rotate images by 90, 180, 270 degrees.
            value = np.random.randint(0, 3)
            if (value == 0):
                img = np.rot90(img, k=1)
            elif (value == 1):
                img = np.rot90(img, k=2)
            elif (value == 2):
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

        # Comprobamos si el total de elementos es par
        if len(list_sort) % 2 == 0:
            columns = int((list_sort[half] + list_sort[half - 1]) / 2)
        else:
            columns = list_sort[int(half)]

        rows = int(self.batch_size / columns)

        ax=[]
        for i in range(columns*rows):
            img = self.batch[i]
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(self.class_dict[self.labels[i]], fontdict={'size': 18})  # set title
            plt.axis('off')
            plt.imshow(img)
        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.385, wspace=0.235)
        plt.show()

