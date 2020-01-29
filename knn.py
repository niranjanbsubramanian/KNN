
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score from numpy import arange

np.random.seed(2019)
X, y = make_classification(n_samples=300, n_features = 5, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f'Accuracy:', (accuracy_score(y_test, y_pred)))

# finding the best value for k
ks = list(arange(1, 50, 2))
scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    scores.append(score.mean())

# plot mse vs K
mse = [1 - x for x in scores]
import matplotlib.pyplot as plt
plt.plot(ks, mse)
plt.xlabel('K')
plt.ylabel('MSE')
plt.show()

#print the best value of K
best_k = ks[mse.index(min(mse))]
print(best_k)

#train the knn with best k
knn = KNeighborsClassifier(n_neighbors=best_k) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
print(f'Accuracy:', (accuracy_score(y_test, y_pred)))
