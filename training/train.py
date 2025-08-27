import random, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parents[1]
data_path = BASE / "data" / "sms_synthetic.csv"

def main():
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    out = BASE / "app" / "model" / "sms_spam_pipeline.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Saved model to: {out}")

if __name__ == "__main__":
    main()