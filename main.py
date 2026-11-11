from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fx_wb(w, x, b):
    return np.dot(x, w) + b


def sigmoid(fx):
    fx = np.asarray(fx, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(fx, -500, 500)))


def dj_calc(x, y, w, b, m=None):
    if m is None:
        m = x.shape[0]
    predictions = sigmoid(fx_wb(w, x, b))
    error = predictions - y
    return (x.T @ error) / m, np.mean(error)


def cost(m, y, probabilities):
    eps = np.finfo(float).eps
    probabilities = np.clip(probabilities, eps, 1 - eps)
    return (-1 / m) * np.sum(
        y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
    )


def gradient_descent(x, y, alpha=0.03, num_iters=1000):
    m, n = x.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        dj_dw, dj_db = dj_calc(x, y, w, b, m)
        cost_history[i] = cost(m, y, sigmoid(fx_wb(w, x, b)))
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b, cost_history


def prepare_data(dataset_path=Path(__file__).resolve().parent / "dataset.csv"):
    df = pd.read_csv(dataset_path).drop_duplicates().dropna(axis=0)
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 220)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 140)]
    df = df[df["ap_hi"] > df["ap_lo"]]

    df["age_years"] = (df["age"] / 365.25).round(1)
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["p_p"] = df["ap_hi"] - df["ap_lo"]
    df["MAP"] = df["ap_lo"] + (df["p_p"] / 3)
    df["BPR"] = df["ap_hi"] / df["ap_lo"].replace(0, 1)

    X = pd.get_dummies(
        df.drop(columns=["cardio", "age"]),
        columns=["cholesterol", "gluc", "gender"],
        drop_first=True,
        dtype=int,
    )
    y = df["cardio"].astype(int)
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    w, b, cost_history = gradient_descent(X_train_scaled, y_train.to_numpy())

    sklearn_model = LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_predictions = sklearn_model.predict(X_test_scaled)

    custom_probabilities = sigmoid(fx_wb(w, X_test_scaled, b))
    custom_predictions = (custom_probabilities >= 0.5).astype(int)

    print(f"Accuracy my own model: {accuracy_score(y_test, custom_predictions):.4f}")
    print(f"Accuracy sklearn model: {accuracy_score(y_test, sklearn_predictions):.4f}")

    plt.plot(range(len(cost_history)), cost_history, label="Cost")
    plt.xlabel("Iterations")
    plt.ylabel("Binary cross-entropy")
    plt.title("Logistic Regression Cost Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
