#!/usr/bin/env python
"""Step 1: sample canonical expressions + spoken realisations and assign leak-free splits."""
import argparse, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT
from src.data.grammar import generate, TEMPLATES


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=cfg["synthetic"]["n_expressions"]); ap.add_argument("--out", default=str(DATA_DIR / "synthetic/expressions.jsonl"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[skip] {out} exists"); return
    seed = cfg["seed"]
    recs = generate(a.n, seed=seed)
    sc = cfg["synthetic"]
    for r in recs:
        h = int(hashlib.md5(r["latex"].replace(" ", "").encode()).hexdigest(), 16) % 10000 / 10000   # split by LaTeX hash => no expression leakage
        r["split"] = "test" if h < sc["test_frac"] else "validation" if h < sc["test_frac"] + sc["val_frac"] else "train"
    hard = generate(500, seed=seed + 1, only_heldout=True)
    seen = set()
    for r in hard:
        if r["latex"] in seen: continue
        seen.add(r["latex"]); r["split"] = "test_hard"; recs.append(r)
    # cap repeats of the same LaTeX in eval splits (train may contain several spoken variants of one expression)
    cnt = {}
    kept = []
    for r in recs:
        k = (r["split"], r["latex"].replace(" ", ""))
        cnt[k] = cnt.get(k, 0) + 1
        if r["split"] in ("train",) or cnt[k] <= 2:
            kept.append(r)
    recs = kept
    # leak check: no test/val latex in train
    tr = {r["latex"].replace(" ", "") for r in recs if r["split"] == "train"}
    recs = [r for r in recs if r["split"] == "train" or r["latex"].replace(" ", "") not in tr]
    for i, r in enumerate(recs):
        r["expr_id"] = f"e{i:06d}"
    with out.open("w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter(r["split"] for r in recs), Counter(r["domain"] for r in recs))
    print(f"unique templates: {len({r['template'] for r in recs})}/{len(TEMPLATES)}  -> {out}")


if __name__ == "__main__":
    main()
