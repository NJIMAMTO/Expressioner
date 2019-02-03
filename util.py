import os
import math
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Concatenate, Lambda
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class PlotHistory(Callback):
    def __init__(self, path, interval=50):
        super().__init__()
        self.interval = interval
        self.plot_save_dir = path

    def on_train_begin(self, logs={}):
        self._reset_hists()
        if not os.path.exists(self.plot_save_dir):
            os.mkdir(self.plot_save_dir)

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if epoch % self.interval == 0 and epoch != 0:
            self._plot(self.acc, self.val_acc,
                       self.loss, self.val_loss,
                       self.plot_save_dir + "full")

            tmp = len(self.loss)
            filename = "{}-{}".format(tmp-self.interval, tmp)
            self._plot(self.acc[-self.interval:],
                       self.val_acc[-self.interval:],
                       self.loss[-self.interval:],
                       self.val_loss[-self.interval:],
                       self.plot_save_dir + filename)
            print("plot to {}".format(self.plot_save_dir))

    def _plot(self, acc, val_acc, loss, val_loss, figname):
        x_ax = np.arange(1, len(acc) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(x_ax, acc, label="acc")
        plt.plot(x_ax, val_acc, label="val_acc")
        plt.title("accuracy")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x_ax, loss, label="loss")
        plt.plot(x_ax, val_loss, label="val_loss")
        plt.title("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figname, dpi=200)
        plt.close()

    def _reset_hists(self):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []


class ShowConfmat(Callback):
    def __init__(self, data_val, label_val, batch_size, interval=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data_val = data_val
        self.label_val = np.argmax(label_val, axis=1)
        self.classes = self.label_val.max() + 1
        self.interval = interval
        self.max_acc = 0
        print("class: ", self.classes)

    def on_epoch_end(self, epoch, logs={}):
        val_acc = logs.get('val_acc')
        if (self.interval is None) and self.max_acc < val_acc:
            self.max_acc = val_acc
            self.confmat()

        elif (self.interval is not None) and (epoch + 1) % self.interval == 0:
            self.confmat()

    def confmat(self):
        pred = self.model.predict(self.data_val, batch_size=self.batch_size)
        pred = np.argmax(pred, axis=1)
        conf_mat = confusion_matrix(self.label_val, pred)
        conf_mat_per = conf_mat / conf_mat.sum(axis=1).reshape((self.classes, 1)) * 100
        print('\n', conf_mat, '\n')
        print(conf_mat_per, '\n')


class AudioSequence(Sequence):
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])

        return X, y


class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.min, self.max = feature_range
        self.min_rev = None
        self.max_rev = None

    def fit(self, X, axis=None):
        result = self._process(X, axis, self.min, self.max)
        self.min_rev = X.min(axis=axis)
        self.max_rev = X.max(axis=axis)
        return result

    def rev(self, X, axis=None):
        judge = (self.min_rev is None) or (self.max_rev is None)
        assert judge is False, "can't reverse"
        result = self._process(X, axis, self.min_rev, self.max_rev)
        return result

    def _process(self, X, axis, range_min, range_max):
        X_std = (X - X.min(axis=axis)) / (X.max(axis=axis) - X.min(axis=axis))
        X_scaled = X_std * (range_max - range_min) + range_min
        return X_scaled


def make_parallel(model, gpu_count):
    """
    model parallel
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:]*0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))

        return Model(input=model.inputs, output=merged)
