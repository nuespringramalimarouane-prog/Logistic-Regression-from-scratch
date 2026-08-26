import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# 1. Load Data
df = pd.read_csv("dataset.csv")

# drop duplicated data 
df = df.drop_duplicates()
# remove rows with null values
df = df.dropna(axis=0)

# 2. Clean Outliers (Crucial for Cardiovascular Dataset)
# Keep physiological boundaries
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 220)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 140)]
df = df[df["ap_hi"] > df["ap_lo"]]

# Convert Age from days to years
df["age_years"] = (df["age"] / 365.25).round(1)

# 3. Feature Engineering
df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
df["p_p"] = df["ap_hi"] - df["ap_lo"]
df["MAP"] = df["ap_lo"] + (df["p_p"] / 3)
df["BPR"] = df["ap_hi"] / df["ap_lo"].replace(0, 1)


# 4. One-Hot Encoding
X = pd.get_dummies(
    df.drop(columns=["cardio", "age"]),
    columns=["cholesterol", "gluc","gender"],
    drop_first=True,
    dtype=int
)
y = df["cardio"].astype(int)

# 5. Split First (Prevents Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Logistic Regression Functions
# ============================================================

def fx_wb(w, x, b):
    return np.dot(x, w) + b


def sigmoid(fx):
    fx = np.array(fx, dtype=float)

    return 1 / (1 + np.exp(-fx))


def dj_calc(x, y, w, b, m):

    # Predictions
    preds = sigmoid(fx_wb(w, x, b))

    # Error
    error = preds - y

    # Gradient with respect to weights
    dj_dw = (x.T @ error) / m

    # Gradient with respect to bias
    dj_db = np.mean(error)

    return dj_dw, dj_db


def cost(m, y, sig):

    return (-1 / m) * (
        np.sum(
            y * np.log(sig)
            + (1 - y) * np.log(1 - sig)
        )
    )


# ============================================================
# Gradient Descent
# ============================================================

def gradient_descent(
    x,
    y,
    alpha=0.03,
    num_iters=1000
):

    m = x.shape[0]
    n = x.shape[1]

    w = np.zeros(shape=(n,))
    b = 0

    cost_history = np.zeros(shape=(num_iters,))

    for i in range(num_iters):

        dj_dw, dj_db = dj_calc(
            x,
            y,
            w,
            b,
            m
        )

        cost_history[i] = cost(
            m,
            y,
            sigmoid(fx_wb(w, x, b))
        )

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b, cost_history


# ============================================================
# Train Custom Logistic Regression
# ============================================================

w, b, cost_history = gradient_descent(
    X_train_scaled,
    y_train
)


# ============================================================
# Scikit-learn Logistic Regression
# ============================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_model_preds = model.predict(X_test_scaled)

acc_model = accuracy_score(
    y_test,
    y_model_preds
)


# ============================================================
# Plot Cost Function Convergence
# ============================================================

plt.plot(
    range(len(cost_history)),
    cost_history,
    color="red"
)

plt.xlabel("iters")
plt.ylabel("j(w,b)")

plt.legend("cost function convergence")

plt.show()


# ============================================================
# Custom Model Predictions
# ============================================================

y_pred_probs = sigmoid(
    fx_wb(w, X_test_scaled, b)
)

y_pred_class = (
    y_pred_probs >= 0.5
).astype(int)


# ============================================================
# Accuracy
# ============================================================

acc = accuracy_score(
    y_test,
    y_pred_class
)


print(f"Accuracy: {acc}")
print(f"Accuracy model: {acc_model}")