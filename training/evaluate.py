import joblib, pandas as pd, json, sys
from pathlib import Path

def main():
    model = joblib.load(Path(__file__).resolve().parents[1] / "app" / "model" / "sms_spam_pipeline.joblib")
    samples = [
        "WIN 1000$ CASH now! Reply YES to claim.",
        "Hey, are we still meeting at 6pm today?",
        "Congratulations! You won a gift card. Click http://bit.ly/reward",
        "Mom asked if you're coming home this weekend."
    ]
    preds = model.predict(samples)
    for s, p in zip(samples, preds):
        print(f"{p.upper():>4} | {s}")
if __name__ == "__main__":
    main()