
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

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import OneHotEncoder

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


path = "/media/mokugyo/ボリューム/3Dface"
files = ["F_Angry","F_Disgust","F_Fear","F_Happy","F_Neutral","F_Surprise","F_Unhappy",
    "M_Angry","M_Disgust","M_Fear","M_Happy","M_Neutral","M_Surprise","M_Unhappy"]
V_4 = ["V0S","V2L","V0S_r","V2L_r"]
V_6 = ["V0S","V2L","V4L","V0S_r","V2L_r","V4L_r"]

dfs = []

for ii in range(0,14):
    for yy in range(0,4):
        file_path = path + "/" + files[ii] + "/" + V_4[yy] + ".csv"
        print(file_path)
        df          =  pd.read_csv(file_path,index_col=0)
        dfs.append(df)  
big_frame = pd.concat(dfs, ignore_index=True)
#big_frame      = big_frame.iloc[:,0:11]

#欠損値の除去
big_frame = big_frame.dropna(how='any')

#ピッチ角に応じて顔半分の特徴点のみを抽出
cols_right = ["class",
            "23-46","25-46","27-46",        #right_eyebrow
            "38-42","45-47",                #eye
            "49-34","52-34","55-34","58-34" #mouth
            ] 
cols_left = ["class",
            "18-37","20-37","22-37",        #left_eyebrow
            "38-42","45-47",                #eye
            "49-34","52-34","55-34","58-34" #mouth
            ] 
rename_cols = ["class",
                "1","2","3",    #eyebrow
                "4","5",        #eye
                "6","7","8","9" #mouth
                ]
df_right = big_frame[big_frame["rot_y"] >= 0]
df_right = df_right[cols_right]
df_right.columns = rename_cols

df_left = big_frame[big_frame["rot_y"] < 0]
df_left = df_left[cols_left]
df_left.columns = rename_cols

big_frame = pd.concat([df_left, df_right], ignore_index=True)
print(big_frame)
#説明変数と目的変数の設定
x = pd.DataFrame(big_frame.drop("class",axis=1))
y =  pd.DataFrame(big_frame["class"])
y = keras.utils.to_categorical(y) #class -> onehot_vec

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

print("x_train: {}\n x_test: {}".format(x_train.shape, x_test.shape))

#ニューラルネットワークモデルの設定
model = Sequential()
model.add(Dense(12, input_shape=(9,)))
model.add(Activation('relu'))

model.add(Dense(7))
model.add(Activation('softmax'))

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,
                    batch_size=200,
                    epochs=400,
                    verbose=1,
                    validation_data=(x_test, y_test))

"""
#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])
"""

#混同行列を算出
predict_class = model.predict_classes(x_test, verbose=0)
true_class = np.argmax(y_test,1)
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