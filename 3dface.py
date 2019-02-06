# coding: utf-8

#モジュールの読み込み
import sys 
import math
import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.layers.core import Dense, Activation
import librosa
from util import ShowConfmat

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Lerning'], loc='lower right')
    #plt.show()

    # 損失の履歴をプロット
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Lerning'], loc="upper right")
    plt.show()

path  = "./3dface_v5/"

files = [["V0S_A","V2L_A","V2L_Ar"],
        ["V0S_D","V2L_D","V2L_Dr"],
        ["V0S_F","V2L_F","V2L_Fr"],
        ["V0S_H","V2L_H","V2L_Hr"],
        ["V0S_S","V2L_S","V2L_Sr"],
        ["V0S_U","V2L_U","V2L_Ur"]]

for ii in range(0,3):
    df_t = []
    df_l = []
    for ll in range(0,6):
        #各視点ごとにデータをまとめる
        file_path = path + files[ll][ii] + ".CSV"
        df = pd.read_csv(file_path,index_col=0)

        #学習データと検証データを結合前に分離しておく
        df_test     = df.sample(40)
        df_learn    = df.drop(df_test.index)

        df_t.append(df_test)
        df_l.append(df_learn)

        print(file_path + ":now loading")

    if ii == 0:
        df_V0S_t = pd.concat(df_t, ignore_index=True)
        df_V0S_l = pd.concat(df_l, ignore_index=True)
    elif ii == 1:
        df_V2L_t = pd.concat(df_t, ignore_index=True)
        df_V2L_l = pd.concat(df_l, ignore_index=True)
    elif ii == 2:
        df_V2Lr_t = pd.concat(df_t, ignore_index=True)
        df_V2Lr_l = pd.concat(df_l, ignore_index=True)

df_test = pd.concat([df_V0S_t, df_V2L_t, df_V2Lr_t], ignore_index=True)
df_learn = pd.concat([df_V0S_l, df_V2L_l, df_V2Lr_l], ignore_index=True)

#欠損値の除去
df_test = df_test.dropna(how='any')
df_learn = df_learn.dropna(how='any')

#ピッチ角に応じて顔半分の特徴点のみを抽出
cols_right = [ "class",
                "right_eyebrow_23x","right_eyebrow_23y","right_eyebrow_25x","right_eyebrow_25y","right_eyebrow_27x","right_eyebrow_27y",
                "right_eye45_x","right_eye45_y",
                "right_lip55_x","right_lip55_y",
                "lip_center_upper52_x","lip_center_upper52_y","lip_center_lower58_x","lip_center_lower58_y"]
cols_left = [  "class",
                "left_eyebrow_18x","left_eyebrow_18y","left_eyebrow_20x","left_eyebrow_20y","left_eyebrow_22x","left_eyebrow_22y",
                "left_eye38_x","left_eye38_y",
                "left_lip49_x","left_lip49_y",
                "lip_center_upper52_x","lip_center_upper52_y","lip_center_lower58_x","lip_center_lower58_y"]

rename_cols = [ "class",
                "1","2","3","4","5","6",    #eyebrow
                "7","8",                    #eye   
                "9","10",                   #lip
                "11","12","13","14"         #lipcenter
                ]

dft_right = df_test[df_test["rot_y"] >= 0]
dft_right = dft_right[cols_right]
dft_right.columns = rename_cols
dft_left = df_test[df_test["rot_y"] < 0]
dft_left = dft_left[cols_left]
dft_left.columns = rename_cols

dfl_right = df_learn[df_learn["rot_y"] >= 0]
dfl_right = dfl_right[cols_right]
dfl_right.columns = rename_cols
dfl_left = df_learn[df_learn["rot_y"] < 0]
dfl_left = dfl_left[cols_left]
dfl_left.columns = rename_cols

test     = pd.concat([dft_right, dft_left], ignore_index=True)
learn    = pd.concat([dfl_right, dfl_left], ignore_index=True)

#説明変数と目的変数の設定
x_train = pd.DataFrame(learn.drop("class",axis=1))
y_train =  pd.DataFrame(learn["class"])
y_train = keras.utils.to_categorical(y_train,num_classes=6) #class -> onehot_vec

x_test = pd.DataFrame(test.drop("class",axis=1))
y_test =  pd.DataFrame(test["class"])
y_test = keras.utils.to_categorical(y_test, num_classes=6) #class -> onehot_vec

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

print("x_train: {}\n x_test: {}".format(x_train.shape, x_test.shape))

#ニューラルネットワークモデルの設定
batch_size = 64
show_conf = ShowConfmat(x_test, y_test, batch_size)

model = Sequential()
model.add(Dense(70, input_shape=(14,)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(70))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(70))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(70))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(70))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(70))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(6))
model.add(Activation('softmax'))


model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=9,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[show_conf])
"""
# 学習モデルの保存
model_json = model.to_json()
with open('keras_model2.json', 'w') as f_model:
    f_model.write(model_json)
model.save_weights('keras_model2.hdf5')
"""
model.save('model_v5.h5')
"""
#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])
"""

#混同行列を算出
predict_class = model.predict_classes(x_train, verbose=0)
true_class = np.argmax(y_train,1)
print(confusion_matrix(true_class, predict_class))

"""
#historyをエクセルファイルに出力
DF = pd.DataFrame(history.history)
DF = DF.ix[:,['acc','val_acc','loss','val_loss']]
DF.index = DF.index + 1

output_name = '5_50_80'
DF.to_excel('/media/mokugyo/ボリューム/Experiment_20180127/学習結果/' + output_name + '.xlsx')
"""

# 学習履歴をプロット
plot_history(history)