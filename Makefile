.PHONY: venv install train app api test

venv:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

train:
	python training/train.py

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q