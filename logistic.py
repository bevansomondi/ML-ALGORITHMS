import logistic_visualization 
import logistic_regression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = datasets.make_blobs(n_samples=1000, n_features=1, centers=2)
X, y = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


lr = logistic_regression.LogisticRegression(1000, 0.01)
lr.fit(X_train, y_train)
predicted_cls, yhat= lr.predict(X_test)
print("My model")
print("++"*10)
print("Accuracy",np.mean(y_test==predicted_cls))
#print("Precision", lr.evaluate(y_test, predicted_cls)[0])
##print("recal", lr.evaluate(y_test, predicted_cls)[1])
#print("f1_score", lr.evaluate(y_test, predicted_cls)[2])
print("++"*10)

lr1 = LogisticRegression(penalty='none',max_iter=1000)
lr1.fit(X_train, y_train)
print("Sklearn Model")
print("++"*10)
print("Accuracy", np.mean(y_test==lr1.predict(X_test)))
#print("Precision", lr.evaluate(y_test, lr1.predict(X_test))[0])
#print("recal", lr.evaluate(y_test, lr1.predict(X_test))[1])
#print("f1_score", lr.evaluate(y_test, lr1.predict(X_test))[2])
print("++"*10)

print("\nWeights")

print(lr.weights)
print(lr1.coef_, lr1.intercept_,)


#logistic_visualization.plot_history(lr.loss_history)
logistic_visualization.plot_points(X_test, y_test, yhat)


