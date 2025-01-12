import random
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class PerceptronProsty(BaseEstimator, ClassifierMixin):
    def __init__(self, wspuczenia):
        self.wspuczenia = wspuczenia
        self.w = None
        self.iteracje = 0

    def decision_function(self, X):
        if np.dot(X, self.w) > 0:
            return 1
        else:
            return -1

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros(X.shape[1])
        k = 0
        bledy = []
        for i in range(X.shape[0]):
            fs = self.decision_function(X[i])
            if y[i] != fs:
                bledy.append(np.hstack((X[i], y[i])))
        E = np.array(bledy)

        maxiter = 0
        while E.shape[0] > 0 and maxiter < 1000:
            maxiter += 1
            indeks = random.randint(0, E.shape[0] - 1)
            wybor = E[indeks]
            self.w += self.wspuczenia * (wybor[-1] * wybor[:-1])
            k += 1

            bledy = []
            for i in range(X.shape[0]):
                fs = self.decision_function(X[i])
                if y[i] != fs:
                    bledy.append(np.hstack((X[i], y[i])))
                E = np.array(bledy)

        self.iteracje = maxiter

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.decision_function(X[i]))
        return np.array(predictions)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Losowanie danych
np.random.seed(42)
X = np.random.randn(300, 2)
y = np.array([1 if x[1] > 0.2 * x[0] else -1 for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = PerceptronProsty(wspuczenia=0.01)
perceptron.fit(X_train, y_train)

# Predykcja na danych testowych
predictions = perceptron.predict(X_test)
print(predictions)

# Obliczanie dokładności na danych testowych
accuracy = np.mean(predictions == y_test)
print(f"Dokładność modelu: {accuracy:.2f}")
print(f"Liczba iteracji: {perceptron.iteracje}")


# Wizualizacja danych i granicy decyzyjnej
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),  np.arange(y_min, y_max, .02))

Z = np.array([perceptron.decision_function(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Rysowanie granicy decyzyjnej
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.title("Perceptron Prosty - Decyzja i Dane")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()