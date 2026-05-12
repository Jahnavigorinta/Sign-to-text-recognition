import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# LOAD DATA
# =========================
dataset = "dataset/landmarks"

data = []
labels = []

for file in os.listdir(dataset):

    label = file.split(".")[0]

    with open(os.path.join(dataset, file)) as f:
        lines = f.readlines()

    for line in lines:
        values = list(map(float, line.strip().split(",")))

        if len(values) == 63:
            data.append(values)
            labels.append(label)

print("Total samples:", len(data))

# Convert to numpy
X = np.array(data)
y = np.array(labels)

# =========================
# ENCODE LABELS
# =========================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", list(encoder.classes_))

# =========================
# STRATIFIED TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 🔥 IMPORTANT
)

# Shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# =========================
# MODELS
# =========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel='rbf', probability=True)
}

results = {}

print("\n===== MODEL PERFORMANCE =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

# =========================
# CROSS VALIDATION
# =========================
print("\n===== CROSS VALIDATION =====")

rf = RandomForestClassifier(n_estimators=200)

scores = cross_val_score(rf, X, y, cv=5)

print("Cross Validation Accuracy:", scores.mean()*100)

# =========================
# BEST MODEL
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name]*100)

# =========================
# SAVE MODEL
# =========================
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("\nModel saved as best_model.pkl")