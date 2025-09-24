import argparse, re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
from collections import Counter
import yaml

HERE = Path(__file__).resolve().parent.parent

def load_config():
    with open(HERE / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def weak_label_acs_workup(mdm_text, patterns):
    if not isinstance(mdm_text, str):
        return 0
    for pat in patterns:
        if re.search(pat, mdm_text):
            return 1
    return 0

def build_keyword_regexps(config):
    pos = [re.compile(p, re.IGNORECASE) for p in config["features"]["keywords"]["positive"]]
    neg = [re.compile(p, re.IGNORECASE) for p in config["features"]["keywords"]["negative"]]
    return pos, neg

def keyword_counts(text, regs):
    if not isinstance(text, str):
        return 0
    text = text or ""
    count = 0
    for r in regs:
        m = r.search(text)
        if not m:
            continue
        start = m.start()
        # Look back a bit for negation words like "no", "denies", "without"
        window = text[max(0, start - 20):start]
        if re.search(r'(?i)\b(no|denies|denied|without)\b', window):
            continue
        count += 1
    return count

def make_feature_frame(df, pos_regs, neg_regs):
    df = df.copy()
    df["pos_kw_hits"] = df["hpi_text"].apply(lambda t: keyword_counts(t, pos_regs))
    df["neg_kw_hits"] = df["hpi_text"].apply(lambda t: keyword_counts(t, neg_regs))
    X = df[["pos_kw_hits", "neg_kw_hits"]].values.astype(float)
    return X, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: encounter_id,hpi_text,mdm_text")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.csv)
    need = {"encounter_id", "hpi_text", "mdm_text"}
    if not need.issubset(df.columns):
        raise SystemExit("CSV must include encounter_id, hpi_text, mdm_text")

    # Weak labels from MDM
    pats = cfg["label_rules"]["acs_workup_any"]["patterns"]
    df["acs_workup"] = df["mdm_text"].apply(lambda t: weak_label_acs_workup(t, pats))

    # Features from HPI
    pos_regs, neg_regs = build_keyword_regexps(cfg)
    X, df_feat = make_feature_frame(df, pos_regs, neg_regs)
    y = df_feat["acs_workup"].values

    # --- Split & train (robust to tiny datasets) ---
    counts = Counter(y)
    small = (len(counts) < 2) or (min(counts.values()) < 2) or (len(y) < 8)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")

    if small:
        clf.fit(X, y)
        print("\n[Info] Small or single-class dataset detected — trained on all rows; skipping test metrics.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=args.seed, stratify=y
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        y_hat = (probs >= 0.5).astype(int)

        print(f"Test AUROC {auroc:.3f}")
        print(f"Test AUPRC {auprc:.3f}")
        print("\nConfusion matrix @0.5")
        print(confusion_matrix(y_test, y_hat))
        print("\nClassification report @0.5")
        # zero_division=0 prevents warnings when a class has no predicted samples
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))

    # Fit on full data and save
    clf.fit(X, y)
    model_dir = HERE / "models"
    model_dir.mkdir(exist_ok=True)
    dump(clf, model_dir / "acs_lr_model.joblib")
    print(f"\nSaved model to {model_dir}")
    # ---- Export per-encounter predictions to CSV ----
    probs_full = clf.predict_proba(X)[:, 1]

    HI_T = 0.75
    MED_T = 0.40
    def band(p):
        return "HIGH" if p >= HI_T else ("MED" if p >= MED_T else "LOW")

    out = df_feat[["encounter_id", "hpi_text", "mdm_text"]].copy()
    out["pos_kw_hits"] = df_feat["pos_kw_hits"]
    out["neg_kw_hits"] = df_feat["neg_kw_hits"]
    out["p_acs_workup"] = probs_full
    out["risk_band"] = [band(p) for p in probs_full]

    out_path = Path(args.csv).with_name("predictions_" + Path(args.csv).name)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    
    # Simple interpretability
    w = clf.coef_.ravel()
    print("\nFeature weights (positive pushes toward ACS workup)")
    print(f"  pos_kw_hits = {w[0]:+.3f}")
    print(f"  neg_kw_hits = {w[1]:+.3f}")

if __name__ == "__main__":
    main()