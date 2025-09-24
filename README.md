# Chest Pain PoC

Minimal demo:
- Weak labels from MDM text (ACS workup cues)
- Curated HPI keywords with basic negation
- Logistic Regression
- FastAPI endpoint + simple web UI

Run:
1) Train: `python scripts/poc.py --csv data/toy_data.csv`
2) API: `uvicorn server.api:app --reload --port 8000`
3) Web: `python3 -m http.server 5500` in `web/` then open http://127.0.0.1:5500
