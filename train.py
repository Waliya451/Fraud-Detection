from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from DecisionTrees import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target
print(type(X))
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )

# classifier = DecisionTree(max_depth=10)
# print(type(X_train))
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)

# def accuracy(y_test, y_pred):
#     return np.sum(y_test == y_pred) / len(y_test)

# acc = accuracy(y_test, predictions)
# print(acc)

data = pd.read_csv("./twitter_human_bots_dataset.csv")
print(type(data))


