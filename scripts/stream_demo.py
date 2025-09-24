import argparse, re, time
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import yaml

HERE = Path(__file__).resolve().parent.parent

def load_model():
    return load(HERE / "models" / "acs_lr_model.joblib")

def load_config():
    with open(HERE / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_keyword_regexps(config):
    pos = [re.compile(p, re.IGNORECASE) for p in config["features"]["keywords"]["positive"]]
    neg = [re.compile(p, re.IGNORECASE) for p in config["features"]["keywords"]["negative"]]
    return pos, neg

def negated(text: str, start_idx: int) -> bool:
    window = text[max(0, start_idx - 20):start_idx]
    return re.search(r'(?i)\b(no|denies|denied|without)\b', window) is not None

def kw_hits_with_negation(text: str, regs) -> int:
    if not isinstance(text, str):
        return 0
    cnt = 0
    for r in regs:
        hit = 0
        for m in r.finditer(text):
            if not negated(text, m.start()):
                hit = 1
                break  # count at most once per concept
        cnt += hit
    return cnt

# Simple evidence checks for “what to ask next”
ASK_CHECKS = [
    ("exertional trigger", re.compile(r"exertional", re.I)),
    ("radiation (arm/jaw/neck)", re.compile(r"radiat", re.I)),
    ("diaphoresis", re.compile(r"diaphoresis|sweating", re.I)),
    ("nausea/vomit", re.compile(r"nausea|vomit", re.I)),
    ("dyspnea", re.compile(r"short(ness)?\s*of\s*breath|dyspnea", re.I)),
    ("syncope", re.compile(r"syncope|fainted|passed out", re.I)),
    ("cardiac risk factors", re.compile(r"htn|hypertension|hld|hyperlip|dm|diabetes|smoker|smoking|cad|mi|stent|cabg", re.I)),
]

def present_without_negation(text: str, regex: re.Pattern) -> bool:
    m = regex.search(text or "")
    if not m:
        return False
    return not negated(text, m.start())

def risk_band(p: float) -> str:
    if p >= 0.75:
        return "HIGH"
    if p >= 0.40:
        return "MED"
    return "LOW"

# --- simple ANSI colors for terminal ---
def color(text, c):
    codes = {"red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m", "reset": "\033[0m"}
    return f"{codes.get(c,'')}{text}{codes['reset']}"

def color_band(band: str) -> str:
    return {
        "HIGH": color("[HIGH]", "red"),
        "MED":  color("[MED]", "yellow"),
        "LOW":  color("[LOW]", "green"),
    }[band]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with hpi_text")
    ap.add_argument("--row", type=int, default=0, help="Row index to stream")
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds between chunks")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    text = str(df.loc[args.row, "hpi_text"])

    model = load_model()
    cfg = load_config()
    pos_regs, neg_regs = build_keyword_regexps(cfg)

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    so_far = ""

    print("\nStreaming HPI with risk + prompts:\n")
    for i, s in enumerate(sentences, 1):
        so_far = (so_far + " " + s).strip()

        pos = kw_hits_with_negation(so_far, pos_regs)
        neg = kw_hits_with_negation(so_far, neg_regs)
        X = np.array([[pos, neg]])
        prob = model.predict_proba(X)[:, 1][0]
        band = risk_band(prob)

        # What-to-ask-next prompts (only when not already HIGH)
        missing = [label for (label, rgx) in ASK_CHECKS if not present_without_negation(so_far, rgx)]
        prompts = missing[:3] if band != "HIGH" else []

        # If negatives present, show a short note
        neg_note = " | lowering features noted" if neg > 0 else ""

        print(f"[{i:02d}] {s}")
        print(f"  -> pos_kw={pos}  neg_kw={neg}  |  P(ACS workup)={prob:.3f}  {color_band(band)}{neg_note}")
        if prompts:
            print(f"     ask next: {', '.join(prompts)}")
        print()

        if args.delay > 0:
            time.sleep(args.delay)

if __name__ == "__main__":
    main()