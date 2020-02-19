import pickle

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback


class Model(object):
    def __init__(self):
        self.batch_size = None
        self.name = None
        self.layers = []
        self.optimizer = None
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
        self.best_loss = None
        self.best_accuracy = None
        self.epochs = None
        self._model = None

    def add_layer(self, layer, index=None):
        """Adds next layer to model at index (default last)

        :param layer: Layer to add
        :param index: Default = None, insertion index
        """

        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)
        print(self.layers)

    def delete_layer(self, layer):
        """Deletes layer from model

        :param layer: Layer to delete
        :type layer: layer obj or int
        """

        self.layers.remove(layer)

    def swap_layers(self):
        pass

    def save(self, path, name):
        """Saves model as 2 files: NAME.h5 and NAME.pkl at path

        :param path: path where model should be saved
        :param name: name of .h5 and .pkl files
        """

        self.name = name

        if self._model is None:
            raise ValueError('Model does not exist')
        self._model.save(path + '/' + name + '.h5')
        self._model = None
        self.layers = None

        with open(path + '/' + name + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Loads model

        :param path: path where model is saved
        """

        self._model = load_model(path + '/' + self.name + '.h5')
        self.layers = self._model.layers

    def build(self):
        """Builds model"""
        self._model = Sequential()

        for layer in self.layers:
            self._model.add(layer)

        if self.optimizer is None:
            self.optimizer = optimizers.Adam()  # raise ValueError('Optimizer is not setted')
        self._model.compile(optimizer=self.optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self, train_data, validation_data, callbacks=None):
        """Fit

        :param train_data: (x_train, y_train)
        :param validation_data: (x_valid, y_valid)
        """
        callbacks = [LossAndAccuracyUpdate(model=self)] + callbacks
        history = self._model.fit(*train_data,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  verbose=1,
                                  validation_data=validation_data,
                                  callbacks=callbacks)

        # self.loss = history.history['loss']
        # self.val_loss = history.history['val_loss']
        # self.accuracy = history.history['accuracy']
        # self.val_accuracy = history.history['val_accuracy']
        print(self.loss, self.val_loss, self.accuracy, self.val_accuracy)


class LossAndAccuracyUpdate(Callback):
    def __init__(self, model):
        super(Callback, self).__init__()
        self.my_model = model

    def on_epoch_end(self, epoch, logs=None):
        self.my_model.loss.append(logs['loss'])
        self.my_model.val_loss.append(logs['val_loss'])
        self.my_model.accuracy.append(logs['accuracy'])
        self.my_model.val_accuracy.append(logs['val_accuracy'])