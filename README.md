# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yuvaram S
RegisterNumber:  212224230315
*/
```
```
# ==========================
# Employee Churn Prediction
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv("C:/Users/admin/Downloads/Employee.csv")

# Basic checks
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

# Encode salary column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features
X = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

y = data["left"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

# Create Decision Tree model (prevent overfitting)
dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

# Train model
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
feature_importance = pd.Series(dt.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

# Example Prediction
sample_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample_employee)

if prediction[0] == 1:
    print("Prediction: Employee will leave")
else:
    print("Prediction: Employee will stay")

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()

```

## Output:
<img width="1120" height="405" alt="image" src="https://github.com/user-attachments/assets/5658014d-8587-4c94-aff4-06cfa2b003db" />

<img width="1118" height="680" alt="image" src="https://github.com/user-attachments/assets/c0da9845-9771-44c5-9072-08f7c687a771" />

<img width="1114" height="691" alt="image" src="https://github.com/user-attachments/assets/e1b4421e-029b-44c9-a74d-968610c0d51f" />

<img width="1806" height="1018" alt="image" src="https://github.com/user-attachments/assets/47d920e8-ecc4-4364-8e47-23f084d6bb27" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
