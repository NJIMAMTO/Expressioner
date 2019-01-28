<<<<<<< HEAD

# coding: utf-8

#モジュールの読み込み
import sys 
import math
import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import Sequence
"""
class AudioSequence(Sequence):
    def __init__(self, x, y, batch_size):
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
"""

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Lerning', 'Test'], loc='lower right')
    #plt.show()

    # 損失の履歴をプロット
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Lerning', 'Test'], loc="upper right")
    plt.show()


#CSVファイルの読み込み
args = sys.argv
dataset_pass_not = args[1]
dataset_pass     = args[2]
df_n        = pd.read_csv(dataset_pass_not,index_col=0)
df          =  pd.read_csv(dataset_pass,index_col=0)
##################################################
#df_n    = df_n.iloc[:,0:10]
#df      = df.iloc[:,0:10]
##################################################
#目的変数の追加
df_n['lavel'] = 0
df['lavel'] = 1


#欠損値の除去
df_n = df_n.dropna(how='any')
df = df.dropna(how='any')

#無表情データと有表情データを2000ずつランダム抽出し結合
df_concat = pd.concat([df_n.sample(n=2000), df.sample(n=2000)])

#説明変数と目的変数の設定
x = pd.DataFrame(df_concat.drop("lavel",axis=1))
y =  pd.DataFrame(df_concat["lavel"])

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

print("x_train: {}\n x_test: {}".format(x_train.shape, x_test.shape))

y_train = keras.utils.to_categorical(y_train,2)
y_test = keras.utils.to_categorical(y_test,2)

#ニューラルネットワークモデルの設定
model_input = keras.layers.Input(shape=(16,))
#model_input = keras.layers.Input(shape=(10,))
x = model_input
x = Dense(12)(x)
x = keras.layers.Activation('relu')(x)

x = Dense(2)(x)
x = keras.layers.Activation('sigmoid')(x)
#x = keras.layers.Activation('softmax')(x)
model = keras.Model(model_input, x)

model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
"""
model.summary()
keras.utils.plot_model(model, "./test.png", show_shapes=True)
print(K.floatx())
batch_size = 50
train_gen = AudioSequence(x_train, y_train, batch_size)
"""

#ニューラルネットワークの学習
#history = model.fit_generator(train_gen, epochs=80,verbose=1,validation_data=(x_test, y_test))
history = model.fit(x_train, y_train,
                    batch_size=50,
                    epochs=80,
                    verbose=1,
                    validation_data=(x_test, y_test))
#history = model.fit(x_train, y_train,batch_size=20,epochs=200,verbose=1)

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])

#historyをエクセルファイルに出力
DF = pd.DataFrame(history.history)
DF = DF.ix[:,['acc','val_acc','loss','val_loss']]
DF.index = DF.index + 1

output_name = '5_50_80'
DF.to_excel('/media/mokugyo/ボリューム/Experiment_20180127/学習結果/' + output_name + '.xlsx')


# 学習履歴をプロット
=======

# coding: utf-8

#モジュールの読み込み
import sys 
import math
import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import Sequence

"""
class AudioSequence(Sequence):
    def __init__(self, x, y, batch_size):
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
"""

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Lerning', 'Test'], loc='lower right')
    #plt.show()

    # 損失の履歴をプロット
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Lerning', 'Test'], loc="upper right")
    plt.show()

#CSVファイルの読み込み
args = sys.argv
dataset_pass_not = args[1]
dataset_pass     = args[2]
df_n        = pd.read_csv(dataset_pass_not,index_col=0)
df          =  pd.read_csv(dataset_pass,index_col=0)
##################################################
#df_n    = df_n.iloc[:,0:10]
#df      = df.iloc[:,0:10]
##################################################

#目的変数の追加
df_n['lavel'] = 0
df['lavel'] = 1
#欠損値の除去
df_n = df_n.dropna(how='any')
df = df.dropna(how='any')
#無表情データと有表情データを2000ずつランダム抽出し結合
df_concat = pd.concat([df_n.sample(n=2000), df.sample(n=2000)])
#説明変数と目的変数の設定
x = pd.DataFrame(df_concat.drop("lavel",axis=1))
y =  pd.DataFrame(df_concat["lavel"])
#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

print("x_train: {}\n x_test: {}".format(x_train.shape, x_test.shape))

y_train = keras.utils.to_categorical(y_train,2)
y_test = keras.utils.to_categorical(y_test,2)

#ニューラルネットワークモデルの設定
model_input = keras.layers.Input(shape=(16,))
x = model_input
x = Dense(12)(x)
x = keras.layers.Activation('relu')(x)

x = Dense(2)(x)
x = keras.layers.Activation('sigmoid')(x)
model = keras.Model(model_input, x)

model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

"""
model.summary()
keras.utils.plot_model(model, "./test.png", show_shapes=True)
print(K.floatx())
batch_size = 50
train_gen = AudioSequence(x_train, y_train, batch_size)
"""

#ニューラルネットワークの学習
#history = model.fit_generator(train_gen, epochs=80,verbose=1,validation_data=(x_test, y_test))
history = model.fit(x_train, y_train,
                    batch_size=50,
                    epochs=80,
                    verbose=1,
                    validation_data=(x_test, y_test))
#history = model.fit(x_train, y_train,batch_size=20,epochs=200,verbose=1)

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])

#historyをエクセルファイルに出力
DF = pd.DataFrame(history.history)
DF = DF.ix[:,['acc','val_acc','loss','val_loss']]
DF.index = DF.index + 1
#output_name = '表情のみ_補正あり'
#DF.to_excel('/media/mokugyo/ボリューム/Experiment_20180127/学習結果/' + output_name + '.xlsx')


# 学習履歴をプロット
>>>>>>> ff5904bbd0937c0dabefb62545fc2d18209cd224
plot_history(history)