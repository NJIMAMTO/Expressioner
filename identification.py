
# coding: utf-8

#モジュールの読み込み
import keras
import numpy as np
import pandas as pd
from keras.models import Model, load_model
import sys
from sklearn.model_selection import train_test_split
import glob

import keyboard
import time
#表情認識モデルをロード
path = "/home/mokugyo/git_project/Expressioner/model_v5.h5"
model = load_model(path)

#=======================データロード=======================#
for f_name in glob.glob("./Deep_for_recognition/*.csv"):
    split_filepath = f_name.split('/')
    df = pd.read_csv(f_name,index_col=0)
    print("load : "+f_name)
    
    print("次の処理に進む場合はキーを押してエンターを")
    a = input()
    #ピッチ角に応じて顔半分の特徴点のみを抽出
    rename_cols = ["1","2","3","4","5","6",     #eyebrow
                    "7","8",                    #eye   
                    "9","10",                   #lip
                    "11","12","13","14"         #lipcenter
                ]
    cols_right = ["right_eyebrow_23x","right_eyebrow_23y","right_eyebrow_25x","right_eyebrow_25y","right_eyebrow_27x","right_eyebrow_27y",
                "right_eye45_x","right_eye45_y",
                "right_lip55_x","right_lip55_y",
                "lip_center_upper52_x","lip_center_upper52_y","lip_center_lower58_x","lip_center_lower58_y"]
    df_right = df[df["rot_y"] >= 0]
    df_right = df_right[cols_right]
    df_right.columns = rename_cols

    cols_left = ["left_eyebrow_18x","left_eyebrow_18y","left_eyebrow_20x","left_eyebrow_20y","left_eyebrow_22x","left_eyebrow_22y",
                "left_eye38_x","left_eye38_y",
                "left_lip49_x","left_lip49_y",
                "lip_center_upper52_x","lip_center_upper52_y","lip_center_lower58_x","lip_center_lower58_y"]
    df_left = df[df["rot_y"] < 0]
    df_left = df_left[cols_left]
    df_left.columns = rename_cols

    #右視点・左視点のDataframeをひとつにまとめる
    x_recognition = pd.concat([df_right, df_left], ignore_index=True)
    #=======================データロードここまで=======================#
    y_predict = model.predict(x_recognition)
    for ii, pre in enumerate(y_predict):
        print(str('{:.2f}'.format(ii/30)) + " : "+ str(pre) + str(np.argmax(pre)))
    print("")
    print("次の処理に進む場合はキーを押してエンターを")
    a = input()
    """
    # ./Deep_for_recognitionと./SVM_for_learningで同名のファイルを検索
    for name in glob.glob("./SVM_for_learning*/" + split_filepath[2], recursive=False):
classes_num = {
                0:0,
                1:0,
                2:0,
                3:0,
                4:0,
                5:0
                }
classes_acc = {
                0:0,
                1:0,
                2:0,
                3:0,
                4:0,
                5:0
                }
for y_t, y_p in zip(y_test, y_predict):
    true_label = np.argmax(y_t)
    predict_label = np.argmax(y_p)
    if true_label == predict_label:
        classes_acc[true_label] += 1
    classes_num[true_label] += 1

print(classes_acc)
print(classes_num)
print(y_test[30], np.argmax(y_test[30]))
print(true_label)
print("")

for i in range(6):
    print("class {0}, acc {1}".format(i, classes_acc[i]/(classes_num[i] + sys.float_info.epsilon)))
"""