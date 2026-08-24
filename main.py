import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#|| Load the dataset || 
dataset =  pd.read_csv("dataset.csv")

#||  Get features from the dataset ||
# One-hot encode categorical features
X_data = pd.get_dummies(dataset.drop(columns=["cardio"]),
                        columns=["cholesterol", "gluc"],
                        drop_first=True)
weight = X_data.get("weight")
height = X_data.get("height")
BMI_data = weight / (height/100)**2

# One-hot encode features in new one 
# Pulse Pressure = ap_hi - ap_lo
ap_hi = X_data.get("ap_hi")
ap_lo = X_data.get("ap_lo")
p_p = ap_hi - ap_lo

#Blood Pressure Ratio → ap_hi / ap_lo
bpr_aplo = ap_lo
bpr_aplo[bpr_aplo == 0] = 1
BPR = ap_hi / bpr_aplo
# Mean Arterial Pressure (MAP) → (ap_lo + (ap_hi - ap_lo)/3)
MAP = (ap_lo + (ap_hi - ap_lo)/3)

# Add BMI and p_p as a new column
X_data["BMI"] = BMI_data
X_data["p_p"] = p_p
X_data["MAP"] = MAP
X_data["BPR"] = BPR
# Separate continuous features
continuous_cols = ["age","height","weight", "BMI","BPR","p_p","MAP", "ap_hi", "ap_lo"]

# Normalize only continuous features
X_mean = X_data[continuous_cols].mean()
X_scale = X_data[continuous_cols].max() - X_data[continuous_cols].min()
X_data[continuous_cols] = (X_data[continuous_cols] - X_mean) / X_scale


#||  Get Target from the dataset ||
y_data = dataset.get("cardio")


#|| split test and train data ||
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.3,random_state=42)


X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)
y_train = y_train.to_numpy(dtype=float)
y_test = y_test.to_numpy(dtype=float)



def fx_wb(w,x,b):
    return (np.dot(x,w) + b)

def sigmoid(fx):
    fx = np.array(fx, dtype=float)
    return (1 / (1 + np.exp(-fx)))

def dj_calc(x,y,w,b,m):
    # predictions: shape (m,)
    preds = sigmoid(fx_wb(w, x, b))   # (m,)
    error = preds - y                 # (m,)

    # gradient w.r.t weights: shape (n,)
    dj_dw = (x.T @ error) / m         # (n,)

    # gradient w.r.t bias: scalar
    dj_db = np.mean(error)            # scalar

    return dj_dw, dj_db

def cost(m,y,sig):
    return (-1/m) * (np.sum((y * np.log(sig) + (1 - y) * np.log(1 - sig))))

def gradient_descent(x,y,alpha=0.99,num_iters=9000):

    m = x.shape[0]
    n = x.shape[1]
    w = np.zeros(shape=(n,))
    b=0
    cost_history = np.zeros(shape=(num_iters,))

    for i in range(num_iters):
        dj_dw, dj_db = dj_calc(x,y,w,b,m)
        cost_history[i] = cost(m,y,sigmoid(fx_wb(w,x,b)))
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b,cost_history

w, b,cost_history = gradient_descent(X_train,y_train)
model = LogisticRegression()

model.fit(X_train,y_train)

y_model_preds = model.predict(X_test)
acc_model = accuracy_score(y_test,y_model_preds)

plt.plot(range(len(cost_history)),cost_history,color="red")
plt.xlabel("iters")
plt.ylabel("j(w,b)")
plt.legend("cost function convergence")
plt.show()


y_pred_probs = sigmoid(fx_wb(w, X_test, b))          # probabilities
y_pred_class = (y_pred_probs >= 0.5).astype(int)     # convert to 0/1 labels

acc = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {acc}")
print(f"Accuracy model: {acc_model}")
print(f"w : {w}")
print(f'b:{b}')
