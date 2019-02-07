# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import glob

big_frame = []
for f_name in glob.glob("./SVM_for_learning/*.csv"):
    df = pd.read_csv(f_name,index_col=0)
    big_frame.append(df)
big_frame = pd.concat(big_frame, ignore_index=True)    

cols_right = ["right_eyebrow_23x","right_eyebrow_23y","right_eyebrow_25x","right_eyebrow_25y","right_eyebrow_27x","right_eyebrow_27y",
                "right_eye45_x","right_eye45_y",
                "right_lip55_x","right_lip55_y",
                "lip_center_upper52_x","lip_center_upper52_y",
                "lip_center_lower58_x","lip_center_lower58_y",

                "trans_x","trans_y","trans_z",
                "rot_x","rot_y","rot_z",
                "Angry","Disgust","Fear","Happy","Surpise","Unhappy","lavel"]

cols_left = ["left_eyebrow_18x","left_eyebrow_18y","left_eyebrow_20x","left_eyebrow_20y","left_eyebrow_22x","left_eyebrow_22y",
            "left_eye38_x","left_eye38_y",
            "left_lip49_x","left_lip49_y",
            "lip_center_upper52_x","lip_center_upper52_y",
            "lip_center_lower58_x","lip_center_lower58_y",

            "trans_x","trans_y","trans_z",
            "rot_x","rot_y","rot_z",
            "Angry","Disgust","Fear","Happy","Surpise","Unhappy","lavel"] 

rename_cols = [ "1","2","3","4","5","6",    #eyebrow
                "7","8",                    #eye   
                "9","10",                   #lip
                "11","12","13","14",        #lipcenter
                "trans_x","trans_y","trans_z",
                "rot_x","rot_y","rot_z",
                "Angry","Disgust","Fear","Happy","Surpise","Unhappy","lavel"] 

df_right = big_frame[big_frame["rot_y"] >= 0]
df_right = df_right[cols_right]
df_right.columns = rename_cols
df_left = big_frame[big_frame["rot_y"] < 0]
df_left = df_left[cols_left]
df_left.columns = rename_cols

big_frame     = pd.concat([df_right, df_left], ignore_index=True)

#説明変数と目的変数の設定
X = pd.DataFrame(big_frame.drop("lavel",axis=1))
y =  pd.DataFrame(big_frame["lavel"]
)
#トレーニングとテストデータを分離
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None )

# データの標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 線形SVMのインスタンスを生成
model = SVC(decision_function_shape='ovo')

# モデルの学習
history = model.fit(X_train_std, y_train)

# トレーニングデータに対する精度
pred_train = model.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

# テストデータに対する精度
pred_test = model.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)




