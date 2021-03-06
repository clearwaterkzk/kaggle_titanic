##### module #####
import random
import os
import pandas as pd
import numpy as np
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
seed_everything()



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

data_embarked = data['Embarked'].values.reshape(-1, 1)
oh_encoder = OneHotEncoder(sparse=False)
oh_encoder.fit(data_embarked)
onehot = pd.DataFrame(oh_encoder.transform(data_embarked), 
                        columns=oh_encoder.get_feature_names(),
                        index=data.index,
                        dtype=np.int8)
onehot.rename(columns={"x0_C":"Embarked_C", "x0_Q":"Embarked_Q", "x0_S":"Embarked_S"}, inplace=True)
data = data.drop(columns=["Embarked"])
data = pd.concat([data, onehot], axis=1)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


cv = KFold(n_splits=5, shuffle=True)
for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    # modeling
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_tr, y_tr)
    y_pred_train = clf.predict(X_tr)
    y_pred_val = clf.predict(X_val)

    print("fold_id : ", fold_id)
    print("train accuracy : ", accuracy_score(y_true=y_tr, y_pred=y_pred_train))
    print("validation accuracy : ", accuracy_score(y_true=y_val, y_pred=y_pred_val))
    print("##########")