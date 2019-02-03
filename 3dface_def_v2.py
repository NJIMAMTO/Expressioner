
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
from util import ShowConfmat
import optuna

import time
import datetime

import uuid

trials = 0

#=======================初回データロード=======================#
path  = "./3dface_v4/"
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
cols_right = [  "class",
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

dfl_right = df_test[df_test["rot_y"] >= 0]
dfl_right = dfl_right[cols_right]
dfl_right.columns = rename_cols
dfl_left = df_test[df_test["rot_y"] < 0]
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

#=======================初回データロードここまで=======================#

def create_model(n_layer, activation, mid_units, dropout_rate):
    model = Sequential()

    #入力層
    model.add(Dense(mid_units, input_shape=(14,),activation=activation))
    model.add(BatchNormalization())

    #中間層
    for ii in range(n_layer):
        model.add(Dense(mid_units, activation=activation))
        model.add(BatchNormalization())

    #出力層
    model.add(Dense(6, activation="softmax"))

    return model

def objective(trial):
    # 試行にUUIDを設定
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr("uuid", trial_uuid)
    
    #説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
    global x_train
    global x_test
    global y_train
    global y_test

    #データの整形
    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)
    #=======================データロードここまで=======================#

    # 調整したいハイパーパラメータの設定
    n_layer = trial.suggest_int('n_layer', 1, 5) # 追加する層を1-5から選ぶ
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 5, 70, 1)) # ユニット数
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1) # ドロップアウト率
    activation = trial.suggest_categorical('activation', ['relu']) # 活性化関数
    optimizer = trial.suggest_categorical('optimizer', ['adam']) # 最適化アルゴリズム
    batch_size = 64
    show_conf = ShowConfmat(x_test, y_test, batch_size)

    #一試行あたりの実行時間測定
    start_1 = time.time()

    # 学習モデルの構築と学習の開始
    model = create_model(n_layer, activation, mid_units, dropout_rate)
    
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(x_train, y_train, 
                        verbose=0,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        callbacks=[show_conf])
    """
    #混同行列を算出
    predict_class = model.predict_classes(x_test, verbose=0)
    true_class = np.argmax(y_test,1)
    print(confusion_matrix(true_class, predict_class))
    """
    
    # 学習モデルの保存
    model_json = model.to_json()
    with open('keras_model.json', 'w') as f_model:
        f_model.write(model_json)
    model.save_weights('keras_model.hdf5')
    
    #実行時間表示
    global trials
    trials += 1
    print("trial = " + str(trials))
    
    elapsed_time = time.time() - start_1
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    dt_now = datetime.datetime.now()
    print(dt_now)
    #optunaは最小値探索なのでマイナスで返す
    return -np.amax(history.history['val_acc'])

def main():
    start = time.time()

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
    
    print(study.best_trial.user_attrs)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
    main()