import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.utils import check_random_state
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



# === More realistic toy data ===
def make_noisy_reviews(n=1000, random_state=42):
    rng = check_random_state(random_state)

    positive = ["good", "great", "nice", "pleasant", "friendly", "clean", "comfortable", "helpful", "fast", "amazing"]
    neutral = ["ok", "average", "fine", "decent", "simple", "regular", "standard"]
    negative = ["bad", "terrible", "dirty", "slow", "rude", "noisy", "broken", "horrible"]

    common = ["room", "hotel", "staff", "service", "food", "location", "price", "breakfast", "wifi", "bed"]
    fillers = ["but", "and", "the", "very", "really", "quite", "a", "too", "for", "with"]

    def random_typo(word):
        if rng.rand() < 0.1:
            i = rng.randint(len(word))
            return word[:i] + rng.choice(list("abcdefghijklmnopqrstuvwxyz")) + word[i:]
        return word

    X, y = [], []
    for _ in range(n):
        rating = rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.25, 0.15])
        length = rng.randint(8, 20)
        text = []

        # choose sentiment-laden words (with some noise)
        if rating <= 2:
            sent_words = rng.choice(negative + neutral, size=3, replace=True)
        elif rating == 3:
            sent_words = rng.choice(neutral + positive + negative, size=3, replace=True)
        else:
            sent_words = rng.choice(positive + neutral, size=3, replace=True)

        # add some noise (e.g. positive words in bad review)
        if rng.rand() < 0.2:
            sent_words[rng.randint(3)] = rng.choice(positive + negative)

        all_words = list(sent_words) + rng.choice(common + fillers, size=length - 3, replace=True).tolist()
        rng.shuffle(all_words)
        text = " ".join(random_typo(w) for w in all_words)
        X.append(text)
        y.append(rating)
    return X, y


# === Nested CV ===
def nested_cross_validation(X, y, configs, outer_splits=5, inner_splits=5, random_state=42):
    rng = check_random_state(random_state)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    results = []

    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train_outer, X_test_outer = np.array(X)[train_outer_idx], np.array(X)[test_outer_idx]
        y_train_outer, y_test_outer = np.array(y)[train_outer_idx], np.array(y)[test_outer_idx]

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + outer_fold)
        best_score, best_cfg = -np.inf, None

        # inner CV loop
        for cfg in configs:
            name = cfg["name"]
            vec_cls, clf_cls = cfg["vectorizer"], cfg["classifier"]
            for vp in ParameterGrid(cfg.get("vect_params", [{}])):
                for cp in ParameterGrid(cfg.get("clf_params", [{}])):
                    pipe = Pipeline([("vect", vec_cls(**vp)), ("clf", clf_cls(**cp))])
                    f1_scores = []
                    for tr_in, te_in in inner_cv.split(X_train_outer, y_train_outer):
                        pipe.fit(np.array(X_train_outer)[tr_in], np.array(y_train_outer)[tr_in])
                        preds = pipe.predict(np.array(X_train_outer)[te_in])
                        f1_scores.append(f1_score(np.array(y_train_outer)[te_in], preds, average="macro"))
                    score = np.mean(f1_scores)
                    if score > best_score:
                        best_score = score
                        best_cfg = (name, vp, cp)

        # evaluate best model on outer test
        name, vp, cp = best_cfg
        pipe = Pipeline([("vect", configs[0]["vectorizer"](**vp)), ("clf", configs[0]["classifier"](**cp))])
        pipe.fit(X_train_outer, y_train_outer)
        preds = pipe.predict(X_test_outer)

        acc = accuracy_score(y_test_outer, preds)
        f1m = f1_score(y_test_outer, preds, average="macro")
        mae = mean_absolute_error(y_test_outer, preds)
        misses = int(np.sum(y_test_outer != preds))

        print(f"Fold {outer_fold}: {name} acc={acc:.3f} f1={f1m:.3f} mae={mae:.3f}")
        results.append({
            "fold": outer_fold,
            "chosen_model": name,
            "vect_params": vp,
            "clf_params": cp,
            "accuracy": acc,
            "f1_macro": f1m,
            "mae": mae,
            "misclassifications": misses
        })

    df = pd.DataFrame(results)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)
    return df


# === Run example ===
if __name__ == "__main__":
    #X, y = make_noisy_reviews(600)
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / "reviews_hotel1_clean.csv"
    df = pd.read_csv(data_path)
    X = df['Review']  # text input
    y = df['Rating']  # target star rating


    configs = [
        {
            "name": "tfidf_logreg",
            "vectorizer": TfidfVectorizer,
            "vect_params": {"ngram_range": [(1, 1), (1, 2)], "min_df": [1, 2]},
            "classifier": LogisticRegression,
            "clf_params": {"C": [1.0, 0.5], "max_iter": [500]},
        },
        {
            "name": "count_svm",
            "vectorizer": CountVectorizer,
            "vect_params": {"ngram_range": [(1, 2)], "min_df": [2]},
            "classifier": LinearSVC,
            "clf_params": {"C": [0.5, 1.0], "max_iter": [1000]},
        },
        {
            "name": "tfidf_nb",
            "vectorizer": TfidfVectorizer,
            "vect_params": {"ngram_range": [(1, 1)], "min_df": [1, 2]},
            "classifier": MultinomialNB,
            "clf_params": {"alpha": [0.5, 1.0]},
        },
        #{
        #    "name": "count_knn",
        #    "vectorizer": CountVectorizer,
        #    "vect_params": {"ngram_range": [(1, 1)], "min_df": [1]},
        #    "classifier": KNeighborsClassifier,
        #    "clf_params": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
        #},
        #{
        #    "name": "tfidf_rf",
        #    "vectorizer": TfidfVectorizer,
        ##    "vect_params": {"ngram_range": [(1, 1), (1, 2)], "min_df": [1]},
        #    "classifier": RandomForestClassifier,
        #    "clf_params": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        #},
    ]


    df = nested_cross_validation(X, y, configs, 10, 10)
    print("\n=== Summary ===")
    print(df)
