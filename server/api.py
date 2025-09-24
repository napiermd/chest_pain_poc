from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from joblib import load
import yaml, re, numpy as np

HERE = Path(__file__).resolve().parent.parent  # project root
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InText(BaseModel):
    text: str

# --- load model + config once ---
MODEL = load(HERE / "models" / "acs_lr_model.joblib")

with open(HERE / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

POS = [re.compile(p, re.IGNORECASE) for p in CFG["features"]["keywords"]["positive"]]
NEG = [re.compile(p, re.IGNORECASE) for p in CFG["features"]["keywords"]["negative"]]

ASK_CHECKS = [
    ("exertional trigger", re.compile(r"exertional", re.I)),
    ("radiation (arm/jaw/neck)", re.compile(r"radiat", re.I)),
    ("diaphoresis", re.compile(r"diaphoresis|sweating", re.I)),
    ("nausea/vomit", re.compile(r"nausea|vomit", re.I)),
    ("dyspnea", re.compile(r"short(ness)?\s*of\s*breath|dyspnea", re.I)),
    ("syncope", re.compile(r"syncope|fainted|passed out", re.I)),
    ("cardiac risk factors", re.compile(r"htn|hypertension|hld|hyperlip|dm|diabetes|smoker|smoking|cad|mi|stent|cabg", re.I)),
]

def negated(text: str, start: int) -> bool:
    window = text[max(0, start-20):start]
    return re.search(r'(?i)\b(no|denies|denied|without)\b', window) is not None

def kw_hits_with_negation(text: str, regs) -> int:
    if not isinstance(text, str): return 0
    cnt = 0
    for r in regs:
        seen = False
        for m in r.finditer(text):
            if not negated(text, m.start()):
                seen = True
                break
        cnt += 1 if seen else 0
    return cnt

def band(p: float) -> str:
    if p >= 0.75: return "HIGH"
    if p >= 0.40: return "MED"
    return "LOW"

def ask_next_list(text: str):
    missing = []
    for label, rgx in ASK_CHECKS:
        m = rgx.search(text or "")
        if not m or negated(text, m.start()):
            missing.append(label)
    return missing[:3]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(inp: InText):
    t = inp.text or ""
    pos = kw_hits_with_negation(t, POS)
    neg = kw_hits_with_negation(t, NEG)
    X = np.array([[pos, neg]])
    prob = float(MODEL.predict_proba(X)[:,1][0])
    return {
        "pos_kw_hits": int(pos),
        "neg_kw_hits": int(neg),
        "prob": prob,
        "band": band(prob),
        "ask_next": ask_next_list(t),
    }
