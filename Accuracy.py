import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Pizza.csv', header = 0)

X = dataset.iloc[:,1:7].values
y = dataset['Brand'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_pred))