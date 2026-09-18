import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from gesture_detector import normalize_landmarks

def row_to_points(row):
    """A CSV row is 63 flat numbers. Regroup them into 21 (x, y, z) tuples."""
    return [(row[i], row[i+1], row[i+2]) for i in range(0, len(row), 3)]

# 1. load
df = pd.read_csv("gesture_data.csv", header=None)
X_raw = df.iloc[:, :-1].to_numpy()        # all columns except the last = 63 numbers
y = df.iloc[:, -1].to_numpy()             # last column = the label

# 2. normalize every row with the SAME function the live app uses
X = np.array([
    np.array(normalize_landmarks(row_to_points(row))).flatten()
    for row in X_raw
])
print("feature matrix:", X.shape, "(rows, 63)")

# 3. split, keeping the class balance the same in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. a deliberately small network: one hidden layer of 32 neurons is plenty for 63 inputs and 3 classes
clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 5. evaluate on data the model never saw
y_pred = clf.predict(X_test)
print("\ntest accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nper-class report:")
print(classification_report(y_test, y_pred))
print("confusion matrix (rows = true label, columns = predicted), label order:", clf.classes_)
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

# 6. save the trained model for main.py to load
joblib.dump(clf, "gesture_classifier.pkl")
print("\nsaved gesture_classifier.pkl")