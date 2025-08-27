import joblib
from pathlib import Path

def load_model():
    return joblib.load(Path(__file__).resolve().parents[1] / "app" / "model" / "sms_spam_pipeline.joblib")

def test_spam():
    m = load_model()
    text = "WIN 500$ CASH now! Reply YES to claim."
    assert m.predict([text])[0] == "spam"

def test_ham():
    m = load_model()
    text = "Hey, are we still meeting at 6pm today?"
    assert m.predict([text])[0] == "ham"