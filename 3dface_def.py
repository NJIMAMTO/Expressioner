
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

import optuna

def create_model(n_layer, activation, mid_units, dropout_rate):
    model = Sequential()

    #入力層
    model.add(Dense(mid_units, input_shape=(9,),activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    #中間層
    for ii in range(n_layer):
        model.add(Dense(mid_units, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    #出力層
    model.add(Dense(7, activation=activation))

    return model

def objective(trial):
    #=======================データロード=======================#
    path = "/media/mokugyo/ボリューム/3Dface"
    files = ["F_Angry","F_Disgust","F_Fear","F_Happy","F_Neutral","F_Surprise","F_Unhappy",
        "M_Angry","M_Disgust","M_Fear","M_Happy","M_Neutral","M_Surprise","M_Unhappy"]
    V_4 = ["V0S","V2L","V0S_r","V2L_r"]
    V_6 = ["V0S","V2L","V4L","V0S_r","V2L_r","V4L_r"]

    dfs = []

    for ii in range(0,14):
        for yy in range(0,4):
            file_path = path + "/" + files[ii] + "/" + V_4[yy] + ".csv"
            df          =  pd.read_csv(file_path,index_col=0)
            dfs.append(df)  
    big_frame = pd.concat(dfs, ignore_index=True)
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

    #説明変数と目的変数の設定
    x = pd.DataFrame(big_frame.drop("class",axis=1))
    y =  pd.DataFrame(big_frame["class"])
    y = keras.utils.to_categorical(y) #class -> onehot_vec

    #説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

    #データの整形
    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)
    #=======================データロードここまで=======================#

    # 調整したいハイパーパラメータの設定
    n_layer = trial.suggest_int('n_layer', 1, 20) # 追加する層を1-5から選ぶ
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 5, 100, 1)) # ユニット数
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1) # ドロップアウト率
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid']) # 活性化関数
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop']) # 最適化アルゴリズム

    # 学習モデルの構築と学習の開始
    model = create_model(n_layer, activation, mid_units, dropout_rate)
    
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(x_train, y_train, 
                        verbose=1,
                        epochs=50,
                        validation_data=(x_test, y_test),
                        batch_size=64)
    #混同行列を算出
    predict_class = model.predict_classes(x_test, verbose=0)
    true_class = np.argmax(y_test,1)
    print(confusion_matrix(true_class, predict_class))

    # 学習モデルの保存
    model_json = model.to_json()
    with open('keras_model.json', 'w') as f_model:
        f_model.write(model_json)
    model.save_weights('keras_model.hdf5')
    
    # 最小値探索なので
    return -np.amax(history.history['val_acc'])

def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    print('best_params')
    print(study.best_params)
    print('-1 x best_value')
    print(-study.best_value)

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))


if __name__ == '__main__':
    main()