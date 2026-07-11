from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("Random Forest Classifier")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.2f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))












