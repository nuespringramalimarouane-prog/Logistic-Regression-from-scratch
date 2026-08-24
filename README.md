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

### 1. Load and encode the data

The script reads the CSV with pandas. The categorical `cholesterol` and `gluc` columns are converted into numeric one-hot encoded columns with `pd.get_dummies()`.

### 2. Engineer additional features

Four features are calculated from the existing measurements:

- **BMI**: `weight / (height / 100)²`
- **Pulse pressure**: `ap_hi - ap_lo`
- **Blood pressure ratio**: `ap_hi / ap_lo`
- **Mean arterial pressure**: `ap_lo + (ap_hi - ap_lo) / 3`

If a diastolic pressure value is zero, it is replaced with `1` before calculating the blood-pressure ratio to avoid division by zero.

### 3. Normalize continuous features

Selected continuous columns are scaled using:

```text
(value - column_mean) / (column_max - column_min)
```

This helps gradient descent optimize the model more consistently across features with different units and ranges.

### 4. Split the data

The data is split into:

- 70% training data
- 30% test data

The split uses `random_state=42`, making the result reproducible.

### 5. Train and evaluate both models

The custom model uses the sigmoid function, binary cross-entropy cost, and gradient descent. It runs with a learning rate of `0.99` for `9,000` iterations.

The scikit-learn model uses `sklearn.linear_model.LogisticRegression` with its default settings.

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
3. Print both accuracy scores, the custom model weights, and its bias.
4. Open a plot showing cost-function convergence over the gradient-descent iterations.

Example output format:

```text
Accuracy: <custom model accuracy>
Accuracy model: <scikit-learn accuracy>
w : [<weights>]
b:<bias>
```

The exact accuracy values can vary if the dataset or preprocessing changes.

## Notes

- This is an educational implementation, not a clinical diagnostic tool.
- The script currently displays the convergence plot with `plt.show()`, so execution may wait until the plot window is closed.
- The preprocessing statistics are calculated before the train/test split. For a production machine-learning workflow, calculate them from the training set only to avoid data leakage.
- The dataset is loaded with a relative path, so run the script from the project directory or update the path in `main.py`.

## License

No license has been specified for this project.