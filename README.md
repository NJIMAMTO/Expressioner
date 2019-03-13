Expressioner
==========================
表情・頭部姿勢変化を使った感情認識プログラムの流れ：顔画像・映像入力->特徴点検出->前処理->**機械学習**

#### 研究室メンバーへ
CSVファイルを読み込んでKerasで学習(バックエンドにtensorflow) ＋ optunaを使ったハイパーパラメータチューニング（といってもこのネットワーク構造ではさほど効果なし）をしています．いろいろソースコードがありますが，製作者側の都合なので中身に大差はないです．

あくまで中身は単純な多層パーセプトロンなので，CNN・DNN・LSTM等をやりたい人は各自でググってください．

# How to Use?
## Dependence
- tensorflow   1.12.0 
- scikit-learn 0.20.0 
- Keras        2.2.4 
- Python       3.5.2

The code is tested under Ubuntu 16.04.

## Installation and Usege
Move to working directory (The dependency is installed)

```
$ git clone <this repository>
$ python3 <code name>
  
  <result>
```

