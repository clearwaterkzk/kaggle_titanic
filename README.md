![comp](./titanic.png)
# [kaggle-Titanic](https://www.kaggle.com/c/titanic/overview)
- directory tree
```
kaggle_titanic
├── README.md
├── input        <---- kaggle APIから取得したデータ
├── notebook     <---- jupyter lab で作業したEDA用ノートブック
└── script       <---- 実験用のPythonスクリプト(.py)
```


- environment
    - build kaggle-docker
    - export kaggle.json
    - run container 
    - in container, run jupyter server
        ```
        jupyter lab --port=XXXX --ip=0.0.0.0 --NotebookApp.token="XXX" --allow-root
        ```


## Description
Purpose of analysis : precdict whether they survived(1) or not(0) by usging passengers infomation.

|Variable|Definition|Key|
|--------|----------|---|
|survival|Survival|0 = No, 1 = Yes|
|pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd
|sex|Sex|
|Age|Age in years|
|sibsp|# of siblings / spouses aboard the Titanic|
|parch|# of parents / children aboard the Titanic|
|ticket|Ticket number|
|fare|Passenger fare|
|cabin|Cabin number|
|embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton


## Log
### 20220228
 - pandas profileから仮説をいくつか出そう
 - 仮説検証のためのDashBoardを作成しよう

### 20220301
 - DachBoardの作り方がわからん
 - EDAの仕方がわからん
     - 仮説を立ててデータをどのような角度から見たいかを検討すべき？

### 20220304
 - pandas profile
     - 欠損値あり、対処必要
     - 想像、仮説　-> 検証
         - Pclass(チケットクラス)、階級が高いほど助かる？
         - Sex(性別)、女性ほど助かる？
         - Age、若いほど助かる？
         - SibSp(兄弟姉妹、配偶者の数)、多いほど助かる？
         - Parch(子供、親の数)、多いほど助かる？
         - Fare(運賃)、Pclassと等しいのでは？高いほど助かる？
         - Cabin、救助用ボートから近いほど助かる？

### 20200305
 - まず、何も考えずにモデリングしてみて、そのあと特徴量エンジニアリングする。
 - ロジスティック回帰でモデリング
     - 欠損値処理、カテゴリー変数を変換
     - 変数選択
         - 目的変数
             - Survived
         - 説明変数
             - Pclass 数字の大小に意味があるので前処理不要
             - Sex male:0, female:1と変換する
             - Age 欠損値を正規分布仮定からのランダムサンプリングで埋める。
             - SibSp
             - Parch
             - Fare 欠損値を平均値で埋める
             - Embarked 欠損値をSで埋め、one hot encoding
           
     


## Reference
 - [Kaggle日記という戦い方](https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068)
 - [VSCodeでWSL2上のDockerコンテナ内コードをデバッグ](https://qiita.com/c60evaporator/items/fd019f5ac6eb4d612cd4)
 - [KagglerのためのGit入門](https://yutori-datascience.hatenablog.com/entry/2017/07/25/163702)
 - [GitHubを最強のToDo管理ツールにする](https://qiita.com/o_ob/items/fd45fba2a9af0ce963c3)
 - [Dockerコンテナ内でKaggle APIをつかう](https://qiita.com/komiya_____/items/88f08e1b7348d3a4cd5e)