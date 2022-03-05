##### module #####
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

# modeling
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_val = clf.predict(X_val)

print("train accuracy : ", accuracy_score(y_true=y_train, y_pred=y_pred_train))
print("validation accuracy : ", accuracy_score(y_true=y_val, y_pred=y_pred_val))
