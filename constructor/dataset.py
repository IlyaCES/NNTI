import os

import numpy as np
from PIL import Image


class Dataset(object):

    @property
    def x_train(self):
        return np.array(self._x_train)

    @property
    def y_train(self):
        return np.array(self._y_train)

    @property
    def x_test(self):
        return np.array(self._x_test)

    @property
    def y_test(self):
        return np.array(self._y_test)

    @property
    def labels(self):
        return self._labels

    @property
    def grayscale(self):
        return self._grayscale

    def load_data(self):
        self._load_x()
        self._load_x(True)
        self._load_y()
        self._load_y(True)

    def __init__(self, train_path, test_path, labels, grayscale=False, resize_shape=False):
        self.train_path = train_path
        self.test_path = test_path
        self._labels = labels
        self._x_train = []
        self._y_train = []
        self._x_test = []
        self._y_test = []
        self._grayscale = grayscale
        self._resize_shape = resize_shape

    def _load_x(self, test=False):
        path = self.test_path if test else self.train_path
        x = self._x_test if test else self._x_train
        size = {}

        for class_folder in os.listdir(path):
            for img in os.listdir(path + '/' + class_folder):
                file_path = path + '/' + class_folder + '/' + img
                img = Image.open(file_path)
                img = img.convert('L') if self._grayscale else img.convert('RGB')
                if self._resize_shape:
                    img = img.resize(self._resize_shape)
                else:
                    size[img.size] = 1
                img_data = np.asarray(img, dtype=np.int16)
                x.append(img_data)
            if len(size) > 1:
                x.clear()
                raise Exception('If reshape_size is not specified all images should be the same size')

    def _load_y(self, test=False):
        path = self.test_path if test else self.train_path
        y = self._y_test if test else self._y_train
        for class_folder in self._labels:
            y.extend([class_folder] * len(os.listdir(path + '/' + class_folder)))

