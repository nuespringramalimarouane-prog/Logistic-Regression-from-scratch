# Cardiovascular Disease Logistic Regression

This project predicts whether a person has cardiovascular disease using health and lifestyle measurements from `dataset.csv`.

The project demonstrates two approaches:

- A logistic regression model implemented manually with NumPy and gradient descent.
- A reference logistic regression model provided by scikit-learn.

The script compares the classification accuracy of both approaches and displays the training cost as it changes during gradient descent.

## Project Files

| File | Description |
| --- | --- |
| `main.py` | Loads the data, prepares features, trains both models, evaluates them, and plots training cost. |
| `dataset.csv` | The dataset used for training and testing. It contains 14,024 records. |
| `README.md` | Project documentation. |

## Dataset

The dataset contains the following columns:

- `age`: Age in days.
- `gender`: Encoded gender value.
- `height`: Height in centimeters.
- `weight`: Weight in kilograms.
- `ap_hi`: Systolic blood pressure.
- `ap_lo`: Diastolic blood pressure.
- `cholesterol`: Cholesterol level category.
- `gluc`: Glucose level category.
- `smoke`: Smoking status.
- `alco`: Alcohol consumption status.
- `active`: Physical activity status.
- `cardio`: Target label. `0` means no cardiovascular disease and `1` means cardiovascular disease.

## How It Works

### 1. Load and clean the data

The script reads the CSV with pandas, removes duplicate rows and rows containing null values, and filters out implausible blood-pressure measurements. It keeps rows where:

- `ap_hi` is between `80` and `220`.
- `ap_lo` is between `40` and `140`.
- `ap_hi` is greater than `ap_lo`.

### 2. Engineer additional features

Five features are calculated from the existing measurements:

- **Age in years**: `age / 365.25`, rounded to one decimal place.
- **BMI**: `weight / (height / 100)²`
- **Pulse pressure**: `ap_hi - ap_lo`
- **Mean arterial pressure**: `ap_lo + (ap_hi - ap_lo) / 3`
- **Blood pressure ratio**: `ap_hi / ap_lo`

If a diastolic pressure value is zero, it is replaced with `1` before calculating the blood-pressure ratio to avoid division by zero. The original `age` column is removed after `age_years` is calculated.

### 3. Encode and split the data

The categorical `cholesterol`, `gluc`, and `gender` columns are converted to numeric one-hot encoded columns with `pd.get_dummies()`. The first category is dropped from each encoded column group to avoid redundant features.

The data is split into 80% training data and 20% test data. The split uses `random_state=42` and `stratify=y`, preserving the class distribution in both sets.

### 4. Standardize the features

`StandardScaler` is fitted only on the training features. The same fitted scaler transforms the test features, preventing test-set information from influencing preprocessing.

### 5. Train and evaluate both models

The custom model uses the sigmoid function, binary cross-entropy cost, and gradient descent. It runs with a learning rate of `0.03` for `1,000` iterations.

The scikit-learn model uses `sklearn.linear_model.LogisticRegression` with `max_iter=1000`.

Both models predict the test set, and their accuracy is calculated with `accuracy_score`.

## Requirements

- Python 3.9 or newer is recommended.
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Installation

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install pandas numpy matplotlib scikit-learn
```

If PowerShell does not recognize `python`, use the Python launcher instead:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install pandas numpy matplotlib scikit-learn
```

## Run the Project

Make sure `main.py` and `dataset.csv` are in the same directory, then run:

```powershell
python main.py
```

The program will:

1. Train the custom logistic regression model.
2. Train the scikit-learn logistic regression model.
3. Print both accuracy scores.
4. Open a plot showing cost-function convergence over the gradient-descent iterations.

Example output format:

```text
Accuracy my own model: <custom model accuracy>
Accuracy sklearn model: <scikit-learn accuracy>
```

The exact accuracy values can vary if the dataset or preprocessing changes.

## Notes

- This is an educational implementation, not a clinical diagnostic tool.
- The script currently displays the convergence plot with `plt.show()`, so execution may wait until the plot window is closed.
- The train/test split is performed before fitting `StandardScaler`; the scaler is fitted on `X_train` and then applied to `X_test`. This avoids leaking test-set scaling statistics into training.
- The dataset is loaded with a relative path, so run the script from the project directory or update the path in `main.py`.

