
# coding: utf-8

#モジュールの読み込み
import sys 

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Lerning', 'Test'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Lerning', 'Test'], loc="upper left")
    plt.show()

#CSVファイルの読み込み
args = sys.argv
dataset_pass_not = args[1]
dataset_pass = args[2]
df_n = pd.read_csv(dataset_pass_not,index_col=0)
df =  pd.read_csv(dataset_pass,index_col=0)

#目的変数の追加
df_n['lavel'] = 0
df['lavel'] = 1

#無表情データと有表情データを2000ずつランダム抽出し結合
df_concat = pd.concat([df_n.sample(n=2000), df.sample(n=2000)])

#説明変数と目的変数の設定
x = pd.DataFrame(df_concat.drop("lavel",axis=1))
y =  pd.DataFrame(df_concat["lavel"])

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = keras.utils.to_categorical(y_train,2)
y_test = keras.utils.to_categorical(y_test,2)


#ニューラルネットワークモデルの設定
model = Sequential()
#model.add(Dense(output_dim=24, input_dim=34, activation='sigmoid'))
#model.add(Dense(output_dim=2, input_dim=24, activation='sigmoid'))
model.add(Dense(24, input_dim=34, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=40,epochs=100,verbose=1,validation_data=(x_test, y_test))
#history = model.fit(x_train, y_train,batch_size=20,epochs=200,verbose=1)

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])

# 学習履歴をプロット
plot_history(history)
