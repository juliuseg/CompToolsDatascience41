import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.pipeline import Pipeline
from pathlib import Path

# --- Your mapping function ---
def map_sentiment(rating):
    if rating <= 2:
        return -1  # negative
    elif rating == 3:
        return 0   # neutral
    else:
        return 1   # positive

# --- Load your cleaned dataset ---
script_dir = Path(__file__).parent
data_path = script_dir / 'data' / 'reviews_hotel1_clean.csv'
df = pd.read_csv(data_path)
X = df['Review']
y = df['Rating'].apply(map_sentiment)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Models to try ---
configs = [
    {"name": "tfidf_logreg", "vectorizer": TfidfVectorizer, "classifier": LogisticRegression, "clf_params": {"max_iter":500}},
    {"name": "count_svm", "vectorizer": CountVectorizer, "classifier": LinearSVC, "clf_params": {"max_iter":1000}}
]

results = []

for cfg in configs:
    name = cfg["name"]
    vec_cls, clf_cls = cfg["vectorizer"], cfg["classifier"]
    clf_params = cfg.get("clf_params", {})

    pipe = Pipeline([
        ("vect", vec_cls()),
        ("clf", clf_cls(**clf_params))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()  # convert to list for JSON

    results.append({
        "model": name,
        "accuracy": acc,
        "mae": mae,
        "y_true": y_test.tolist(),
        "y_pred": preds.tolist(),
        "confusion_matrix": cm
    })

# --- Save to JSON ---
out_path = script_dir / 'data' / "train_test_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {out_path}")
