# CodaLab | Animal Behavior Challenge (ABC2018) for understanding animal behavior
+ HWIPラボローテーション用デモ
+ GPSの軌跡からトリの性別を判定せよ!
+ [Reference - CodaLab](https://competitions.codalab.org/competitions/16283)



## Task
各ファイルのGPSの軌跡データについて, 2クラス分類(male,female)を行う. 
正解ラベルが学習データしか公開されていないので, 以下の形で学習と評価用に分ける.

+ `000~499.csv`: `TRAINING`セット
+ `500~631.csv`: `VALIDATION`セット


今回の課題では (1)学習器や (2)特徴量を変更しながら, `VALIDATION` セットのF値を最大にする.


### 評価指標
クラス分類問題における主な評価指標は以下の3つ. 今回は最終評価にF1-measure (F値)を使用する. 詳細は[ここ(大人になってからの再学習 - 検索結果の「再現率」と「適合率」)](http://zellij.hatenablog.com/entry/20120214/p1)を参照されたい.

1. **Precision**: 適合率, 予測結果が正解している割合
1. **Recall**: 再現率, 各クラスのサンプルをどのくらい拾えてこれているかを表す割合.
1. **F1-measure**: PrecisionとRecallの調和平均


評価は`code/utls.py: socres()`のAllのF値で評価します.


### サンプルコード
jupyter notebookのサンプルコードが, `code/Sample-{Method}.ipnb`に有ります. 適宜参考にしてください.

+ `code/Sample-RandomForest.ipynb`: RandomForest Classifire
+ `code/Sample-SVM.ipynb`: SVM (サポートベクターマシン)
+ `code/Sample-NN.ipynb`: Neural Network


### Input Files
1つのCSVファイル (`train/000.csv,630.csv`) に付き, 1トリップ分の軌跡が含まれている. CSV内の各行は各時刻のGPSのロケーション情報を記録している.
GPSロケーションの他に以下の情報が登録されている. データはおよそ1minごとに記録.

|#| Data type | Column Name |
|-|-----------|-------------|
|0| float     | longtitude  | 
|1| float     | latitude    | 
|2| float     | sun azimuth [degree] clockwise from the North |
|3| float     | sun elevation [degree] upward from the horizon |
|4| int       | daytime (betweem sunrise and sunset, `=1`), nighttime (`=0`) |
|5| int       | elapsed time [sec] after starting the trip |
|6| Time      | local time (format=`%H:%M:%S`) |
|7| int       | days (starts from 0, and increments by 1 when the local time passes 23:59:59) |

+ Example

```
139.29220,38.56632,76.42170,-4.45122,0,0,04:54:03,0
139.29300,38.56763,76.58196,-4.25726,0,60,04:55:03,0
139.29400,38.57053,76.73674,-4.06880,0,118,04:56:01,0
139.29620,38.57563,76.89729,-3.87201,0,178,04:57:01,0
...
```


### Labels: gender, or male/female
学習データに関しては正解データ有り. テストデータに関しては正解ラベルは公開されていない (2018.10.20現在).
Trainingデータの正解は, `train_labels.csv`にあり. 行番号 (0開始)とTrainingファイルの番号が一致している. 

| Gender | Binary |
|--------|--------|
| male   | 0      |
| female | 1      |


+ Example

```
1
1
1
0
1
0
0
...
```

### Stats: Number of the dataset
Trainingセットのデータ数は以下の通り. `TRAINIG`と`Test`は今回のデモ用に分割したものの値.


+ `TRAIN` (000~499)

| Trajectory | Number |
|------------|--------|
| Total      | 500    |
| - male     | 235    |
| - female   | 265    |


+ `TEST` (500~631)

| Trajectory | Number |
|------------|--------|
| Total      | 131    |
| - male     | 70     |
| - female   | 61     |



## Enviroment Setting
### Requirements
+ python3>=3.5
+ 


### Use Docker
local環境で行う場合は以下の設定は必要ありません. local環境を汚したくない場合は以下のコマンドでコンテナを立てましょう. 以下のコマンドは `docker/`内で実行してください.

```
# Move Directory
cd docker
# Build container
docker build -t <image name>:<tag> .
# Start Container
docker run -it -p 8888:8888  -v <absolute path to root directory>:/root/ --name <container name>  <image name>:<tag> jupyter notebook --allow-root
```

+ Example
```
docker build -t codalab:latest .
docker run -it --rm \
	-p 8888:8888 \
	-v /home/naoya/code708/codalab-abc2018-trajectory:/root/work \
	--name labrotation  codalab:latest jupyter notebook --allow-root
```

-----------------------------------------------------------
# 学習のヒント!
精度を高めていく上で使えるテクニック(??)をまとめました.

## 学習器
### Base Classifier
+ Linear Calssifier
+ Decision Tree
+ SVM


### Ensemble Classifier
複数の弱い学習器を複数作成して, 多数決をすることで性能を高める.

+ Random Forest
+ Gradient Boost
+ xGBoost
+ Light Gradient Boost


## 特徴量
+ window size: 特徴量をどの粒度で抽出するか?


## 学習アプローチ

### Basic Method [Level.1]
1つの軌跡全体から特徴量を計算して, そのベクトルを1 sampleとする.

### Ensemble Method 1 [Level.2]
同じアルゴリズムでパラメータを変えた学習器を作成する. 各学習器での予測結果を出す. この結果を使って多数決を行い, 最終的な予測結果を出す.


### Ensemble Method [Level.3]
複数のアルゴリズムで学習器を複数作成し、各学習器での予測結果を出す. この結果を使って多数決を行う. また各アルゴリズムの性能がわかっているなら, それを用いた重み付き平均で最終結果を出す.

### Ex.
分類器をC1(90%), C2(70%), C3(40%)とするならば.

```
prediction = 0.45*y_c1 + 0.35*y_c2 + 0.20*y_c3
```

### Window-wise Majority Vote [Level.5]
学習を軌跡単位ではなく, windowをサンプルとして学習する. 結果はwindow毎の予測結果を多数決して決める. 全体より局所的な特徴量を使うほうが, male/femaleの差を抽出できそう.  以下の追加テクニックは使えそう.

+ 学習器は2値分類ではなく, null(判定不能) を含めた3値分類にする.
+ 予測結果を確率で取得し, しきい値を超えたwindowの結果だけを使用する.

**注)** 思いつきなので本当にうまく行くかわからない....

