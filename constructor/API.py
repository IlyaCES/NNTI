import os
import pickle
import shutil

import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.utils import to_categorical

from constructor.dataset import Dataset
from constructor.model import Model


class NNConstructorAPI:

    def __init__(self):
        self.model = Model()
        self.dataset = None
        self._create_model_dir()

    @staticmethod
    def _create_model_dir():
        """Makes model dir in root folder"""

        if not os.path.exists('models'):
            os.makedirs('models')

    def set_data(self, path, grayscale=False, resize_shape=False):
        """Set dataset path

        :param path: Path to dataset
        :param grayscale: loads images as RGB if False, grayscaled if True (default = False)
        :param resize_shape: Images will be scaled to resize_shape if specified, if False images should be teh same size
        :type resize_shape: (int, int)
        :raises ValueError: Path should include 2 folders: train and test
        :raises ValueError: Train and test have different label folders
        """

        if frozenset(os.listdir(path)) != {'train', 'test'}:
            raise ValueError('Path should include 2 folders: train and test')
        test_labels = os.listdir(path + '/test')
        train_labels = os.listdir(path + '/train')
        if set(test_labels) != set(train_labels):
            raise ValueError('Train and test have different label folders')
        self.dataset = Dataset(path + '/train', path + '/test', train_labels, grayscale, resize_shape)

    def set_optimizer(self, **params):
        """Sets optimizer for training

        :param algorithm: Adam, SGD, RMSProp, Adagrad or Adadelta
        :param learning_rate: learning rate for chosen algorithm
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param momentum: momentum for SGD
        :param rho: rho for RMSProp or Adadelta
        :raises ValueError: If wrong algorithm name is passed
        """

        if params['algorithm'] == 'Adam':
            self.model.optimizer = optimizers.Adam(learning_rate=params['learning_rate'],
                                                   beta_1=params['beta_1'],
                                                   beta_2=params['beta_2'])
        elif params['algorithm'] == 'SGD':
            self.model.optimizer = optimizers.SGD(learning_rate=params['learning_rate'],
                                                  momentum=params['momentum'])
        elif params['algorithm'] == 'RMSProp':
            self.model.optimizer = optimizers.RMSprop(learning_rate=params['learning_rate'],
                                                      rho=params['rho'])
        elif params['algorithm'] == 'Adagrad':
            self.model.optimizer = optimizers.Adagrad(learning_rate=params['learning_rate'])
        elif params['algorithm'] == 'Adadelta':
            self.model.optimizer = optimizers.Adadelta(learning_rate=params['learning_rate'],
                                                       rho=params['rho'])
        else:
            raise ValueError()

    def get_activation(self, name):
        """Returns activation function

        :param name: relu, sigmoid or tanh
        :raises ValueError: If function name is incorrect
        """

        if name == 'relu':
            return activations.relu
        elif name == 'sigmoid':
            return activations.sigmoid
        elif name == 'tanh':
            return activations.tanh
        else:
            raise ValueError('Incorrect activation function')

    def add_conv(self, filters=32, kernel_size=(3, 3), activation='relu'):
        """Adds convolutional layer to model

        :param filters: num of filter for conv layer (default = 32)
        :param kernel_size: filter shape (defalut = (3, 3))
        :type kernel_size: (int, int)
        :param activation: activation function for layer. relu, sigmoid or tanh (default = relu)
        """

        layer = Conv2D(filters=filters, kernel_size=kernel_size)
        layer.activation = self.get_activation(activation)
        self.model.add_layer(layer)
        return layer

    def add_dense(self, units=64, activation='relu'):
        """Adds dense layer to model

        :param units: number of neurons in layer (default = 64)
        :param activation: activation function for layer. relu, sigmoid or tanh (default = relu)
        """

        layer = Dense(units)
        layer.activation = self.get_activation(activation)
        self.model.add_layer(layer)
        return layer

    def add_max_pooling(self, pool_size=(2, 2)):
        """Adds pooling layer to model

        :param pool_size: pool_size (default = (2, 2))
        :type pool_size: (int, int)
        """

        layer = MaxPooling2D(pool_size=pool_size)
        self.model.add_layer(layer)
        return layer

    def add_flatten(self):
        """Adds flatten layer"""

        layer = Flatten()
        self.model.add_layer(layer)
        return layer

    def add_dropout(self, rate=0.5):
        """Adds dropout layer

        :param rate: probability to deactivate neuron (default = 0.5)
        """

        layer = Dropout(rate)
        self.model.add_layer(layer)
        return layer

    def delete_layer(self, layer):
        """Deletes layer
        :param layer: layer to delete
        :type layer: int or layer obj
        """

        self.model.delete_layer(layer)

    def save_model(self, name):
        """Saves model at ./models/NAME/

        :param name: Model name, also used as a folder name
        :raises ValueError: If ./models/NAME already exists
        """

        path = 'models/' + name
        if os.path.exists(path):
            raise ValueError('Model directory ' + name + ' already exists')
        os.makedirs(path)
        self.model.save(path, name)

    def load_model(self, name):
        """Loads model

        :param name: Name of the model
        :raises ValueError: If model does not exist
        """

        path = 'models/' + name
        if os.path.exists(path):
            with open(path + '/' + name + '.pkl', 'rb') as model_file:
                self.model = pickle.load(model_file)
                self.model.load(path)
        else:
            raise ValueError('Model ' + name + ' does not exist')

    def delete_model(self, name):
        """Delete model
        :param name: name of model to delete
        :raises ValueError: If model doesn't exist
        """

        path = 'models/' + name
        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            raise ValueError('Model ' + name + ' does not exist')

    def build(self):
        """Builds model"""

        if len(self.model.layers) == 0:
            raise ValueError('you have trained empty model. No results')

        self.dataset.load_data()

        if type(self.model.layers[0]) is Conv2D:
            if self.dataset.grayscale:
                self.model.add_layer(Input(shape=(28, 28, 1)), 0)
            else:
                self.model.add_layer(Input(shape=(28, 28, 3)), 0)
        elif type(self.model.layers[0]) is Dense:
            self.model.add_layer(Flatten(), 0)
        self.model.add_layer(Dense(len(self.dataset.labels), activation='softmax'))
        self.model.build()

    def fit(self, batch_size=32, epochs=5, callbacks=None):
        """Stars training
        :param batch_size: batch size (default = 32)
        :param epochs: number of epochs (default = 5)
        :param callbacks: callback for keras fit method
        """

        y_train = to_categorical(self.dataset.y_train, num_classes=len(self.dataset.labels))
        y_test = to_categorical(self.dataset.y_test, num_classes=len(self.dataset.labels))

        train_shape = self.dataset.x_train.shape
        test_shape = self.dataset.x_test.shape
        if self.dataset.grayscale:
            x_train = np.asarray(self.dataset.x_train).reshape(*train_shape, 1)
            x_test = np.asarray(self.dataset.x_test).reshape(*test_shape, 1)
        else:
            x_train = np.asarray(self.dataset.x_train)
            x_test = np.asarray(self.dataset.x_test)

        self.model.batch_size = batch_size
        self.model.epochs = epochs
        self.model.fit((x_train / 255, y_train), (x_test / 255, y_test), callbacks=callbacks)
