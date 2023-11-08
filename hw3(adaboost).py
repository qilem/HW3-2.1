import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

df = pd.read_csv(r'C:\Users\qilemeng\OneDrive\Desktop\spambase.data.shuffled', header=None, sep=',')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y[y == 0] = -1
X_train = X[:3450]
y_train = y[:3450]
X_test = X[3450:]
y_test = y[3450:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def AdaBoost(X, y, T):
    m = X.shape[0]
    D = np.ones(m) / m
    H = []
    alpha = []

    for t in range(T):
        h = DecisionTreeClassifier(max_depth=1)
        h.fit(X, y, sample_weight=D)
        pred = h.predict(X)
        err = np.sum(D * (pred != y))
        a = 0.5 * np.log((1 - err) / err)
        D *= np.exp(-a * y * pred)
        Z = np.sum(D)
        D /= Z
        H.append(h)
        alpha.append(a)

    def final_classifier(X):
        HX = np.array([a * h.predict(X) for h, a in zip(H, alpha)])
        return np.sign(np.sum(HX, axis=0))

    return final_classifier, H, alpha
k = 3
T_values = [10**i for i in range(1, k+1)]

cv_means = []
cv_stddevs = []

for T in T_values:
    kf = KFold(n_splits=10)
    cv_errors = []

    for train_index, val_index in kf.split(X_train_scaled):
        X_train_kf, X_val_kf = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        final_clf, H, alpha = AdaBoost(X_train_kf, y_train_kf, T)
        y_val_pred = final_clf(X_val_kf)
        val_error = 1 - accuracy_score(y_val_kf, y_val_pred)
        cv_errors.append(val_error)
    cv_means.append(np.mean(cv_errors))
    cv_stddevs.append(np.std(cv_errors))

cv_errors = np.array(cv_stddevs)

plt.errorbar(T_values, cv_means, yerr=cv_errors, fmt='-o')

plt.xlabel('Number of Rounds of Boosting (T)')
plt.ylabel('Cross-Validation Error')
plt.title('CV Error')
plt.show()


T = T_values[np.argmin(cv_errors)]
print(T)
final_clf, H, alpha = AdaBoost(X_train_scaled, y_train, T)

y_train_pred = final_clf(X_train_scaled)
y_test_pred = final_clf(X_test_scaled)

train_errors = [1 - accuracy_score(y_train, np.sign(
    np.sum([a * h.predict(X_train_scaled) for h, a in zip(H[:t + 1], alpha[:t + 1])], axis=0))) for t in range(T)]
test_errors = [1 - accuracy_score(y_test, np.sign(
    np.sum([a * h.predict(X_test_scaled) for h, a in zip(H[:t + 1], alpha[:t + 1])], axis=0))) for t in range(T)]

plt.plot(range(1, T + 1), train_errors, label='Training Error')
plt.plot(range(1, T + 1), test_errors, label='Test Error')
plt.xlabel('Number of Rounds of Boosting (t)')
plt.ylabel('Error')
plt.title('Training and Test Error')
plt.legend()
plt.show()
