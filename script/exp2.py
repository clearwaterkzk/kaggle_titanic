##### module #####
import random
import os
from tabnanny import verbose
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import torch


##### fix seed #####
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
# seed_everything()



##### load data #####
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

##### preprocess #####
# train, testに同じ加工を行うため、結合して1回で加工してしまう
data = pd.concat([train, test], axis=0, sort=False)
data = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data["Embarked"] = data["Embarked"].map({"S":0, "C":1, "Q":2}).astype(int)



categorical_features = ["Pclass", "Sex", "Embarked"]
train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


params = {
    'objective': 'binary',
    'max_bin': 300,
    'learning_rate': 0.01,
    'num_leaves': 40
}

oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True)

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    # modeling
    print("fold_id : ", fold_id)
    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature=categorical_features)
    model = lgb.train(params, lgb_train,
                        valid_sets=[lgb_train, lgb_eval],
                        verbose_eval=10,
                        num_boost_round=1000,
                        early_stopping_rounds=10)
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    print("##########")

y_pred_oof = (oof_train > 0.5).astype(int)
print("validation accuracy ", accuracy_score(y_train, y_pred_oof))