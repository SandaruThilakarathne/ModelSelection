import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

le = LabelEncoder()

dataset = pd.read_csv('Python/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# encoding categorical data
# 1. encoding gender column
X[:, 2] = le.fit_transform(X[:, 2])

# 2. encoding geography column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                    test_size=0.20)

clasifier = XGBClassifier()
clasifier.fit(X_train, y_train)

y_pred = clasifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(cm)

accuracies = cross_val_score(estimator=clasifier, X=X_train, y=y_train, cv=10)
print("Accuracies {:.2f}%".format(accuracies.mean() * 100))
print("Standerd Deviation {:.2f}%".format(accuracies.std() * 100))
