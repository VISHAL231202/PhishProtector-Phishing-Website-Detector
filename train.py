import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import warnings

warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('phishing.csv')

# Display the first few rows and column names
print(data.head())
print(data.columns)

# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop(["class", "Index"], axis=1)
y = data["class"]

# Visualize correlation between features
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.title('Correlation between different features', fontsize=15, c='black')
plt.show()

# Random Forest Classifier function
def RandomForestModel(X):
    rf_train = []
    rf_test = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_train_rf = rf.predict(X_train)
    y_test_rf = rf.predict(X_test)
    
    # Calculate accuracy scores
    acc_train_rf = metrics.accuracy_score(y_train, y_train_rf)
    acc_test_rf = metrics.accuracy_score(y_test, y_test_rf)
    
    print("Random Forest Classifier: Accuracy on training Data: {:.3f}".format(acc_train_rf))
    print("Random Forest Classifier: Accuracy on test Data: {:.3f}".format(acc_test_rf))
    rf_train.append(acc_train_rf)
    rf_test.append(acc_test_rf)
    
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(rf_train) + 1), rf_train, label="Train accuracy")
    plt.plot(range(1, len(rf_test) + 1), rf_test, label="Test accuracy")
    plt.legend()
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Classifier Performance")
    plt.show()

# Logistic Regression function
def LogisticRegressionModel(X):
    lr_train = []
    lr_test = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Logistic Regression Classifier
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_train_lr = lr.predict(X_train)
    y_test_lr = lr.predict(X_test)
    
    # Calculate accuracy scores
    acc_train_lr = metrics.accuracy_score(y_train, y_train_lr)
    acc_test_lr = metrics.accuracy_score(y_test, y_test_lr)
    
    print("Logistic Regression Classifier: Accuracy on training Data: {:.3f}".format(acc_train_lr))
    print("Logistic Regression Classifier: Accuracy on test Data: {:.3f}".format(acc_test_lr))
    lr_train.append(acc_train_lr)
    lr_test.append(acc_test_lr)
    
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(lr_train) + 1), lr_train, label="Train accuracy")
    plt.plot(range(1, len(lr_test) + 1), lr_test, label="Test accuracy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression Classifier Performance")
    plt.show()

# Call Random Forest and Logistic Regression functions with different feature sets
RandomForestModel(X)
LogisticRegressionModel(X)
