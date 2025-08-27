# 📱 SMS Spam Detector — Classic ML (TF-IDF + Logistic Regression)

Production-ready beginner project you can **push to GitHub** and show on **LinkedIn**.

### 🔥 Highlights
- End-to-end: data ➜ training ➜ metrics ➜ saved model ➜ API ➜ UI.
- Clean repo with tests, CI, Docker, Makefile.
- No API keys, works fully offline (uses synthetic dataset).

---

## 🚀 Quickstart

```bash
git clone YOUR_REPO_URL
cd ai-sms-spam-detector
python -m venv .venv && . .venv/bin/activate   # or: py -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python training/train.py                       # (optional) retrain model
streamlit run app/streamlit_app.py             # open UI at http://localhost:8501
# or run API:
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 🧪 Tests
```bash
pytest -q
```

### 🐳 Docker
```bash
docker build -t sms-spam-detector .
docker run -p 8501:8501 -p 8000:8000 sms-spam-detector
```

---

## 🧠 Model
- **Pipeline:** `TfidfVectorizer(1-2 grams)` + `LogisticRegression`
- **Data:** synthetic SMS dataset (`data/sms_synthetic.csv`)
- **Artifacts:** `app/model/sms_spam_pipeline.joblib`

See [`training/METRICS.md`](training/METRICS.md) for current scores.

---

## 🖥 UI (Streamlit)
- Input any SMS text and get **SPAM/HAM** prediction with confidence.

## 🌐 API (FastAPI)
- `POST /predict` with JSON: `{ "text": "your message" }`

---

## 📂 Repo Structure
```
ai-sms-spam-detector/
├─ app/
│  ├─ model/sms_spam_pipeline.joblib
│  ├─ api.py               # FastAPI service
│  └─ streamlit_app.py     # Streamlit UI
├─ data/
│  └─ sms_synthetic.csv    # synthetic dataset (no keys, offline)
├─ training/
│  ├─ train.py             # (re)train the model
│  ├─ evaluate.py          # quick sample predictions
│  └─ METRICS.md
├─ tests/
│  └─ test_inference.py    # sanity tests
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ Dockerfile
├─ Makefile
├─ LICENSE
├─ .gitignore
└─ README.md
```

---

## 📝 LinkedIn Post (copy-paste)
> Shipped a tiny end-to-end **AI project**: SMS Spam Detector 📱🤖  
> Built a classic ML pipeline (TF-IDF + Logistic Regression), added a **Streamlit UI** + **FastAPI API**, tests, CI and Docker.  
> No keys, runs locally, clear code. Repo: <your GitHub link>  
> #machinelearning #nlp #python #fastapi #streamlit #datascience

---

## 📣 Notes
- This project is educational and uses synthetic messages; **do not** deploy to production without further evaluation on real data.
- MIT licensed — feel free to fork and improve.# AI_Spam_Cheker
