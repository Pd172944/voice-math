#!/usr/bin/env python
"""Step 1: sample canonical expressions + spoken realisations and assign leak-free splits.
 * expressions.jsonl       : 68 hand-written templates (train/validation/test + held-out-template test_hard)   [v1 corpus]
 * expressions_comp.jsonl  : compositional random trees (train/validation/test_comp)                          [v2: generalisation]
 * expressions_ood.jsonl   : hand-written out-of-distribution set (test_ood; never trained on)               [v2]
Every file is generated once (idempotent) and splits are by hash of the canonical LaTeX => no expression leakage."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT
from src.data.grammar import generate, TEMPLATES
from src.data.compose import compose
from src.data.ood import OOD
from src.evaluation.metrics import norm_latex


def h(s):
    return int(hashlib.md5(s.replace(" ", "").encode()).hexdigest(), 16) % 10000 / 10000


def write(path, recs):
    with path.open("w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text()); sc = cfg["synthetic"]; seed = cfg["seed"]
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=sc["n_expressions"]); ap.add_argument("--n_comp", type=int, default=sc["n_compositional"])
    a = ap.parse_args()
    out = DATA_DIR / "synthetic"; out.mkdir(parents=True, exist_ok=True)

    # ---------------- v1 templates
    f1 = out / "expressions.jsonl"
    if not f1.exists():
        recs = generate(a.n, seed=seed)
        for r in recs:
            x = h(r["latex"]); r["split"] = "test" if x < sc["test_frac"] else "validation" if x < sc["test_frac"] + sc["val_frac"] else "train"
        hard, seen = generate(500, seed=seed + 1, only_heldout=True), set()
        for r in hard:
            if r["latex"] in seen: continue
            seen.add(r["latex"]); r["split"] = "test_hard"; recs.append(r)
        cnt, kept = {}, []
        for r in recs:
            k = (r["split"], r["latex"].replace(" ", "")); cnt[k] = cnt.get(k, 0) + 1
            if r["split"] == "train" or cnt[k] <= 2: kept.append(r)
        recs = kept
        tr = {r["latex"].replace(" ", "") for r in recs if r["split"] == "train"}
        recs = [r for r in recs if r["split"] == "train" or r["latex"].replace(" ", "") not in tr]
        for i, r in enumerate(recs): r["expr_id"] = f"e{i:06d}"
        write(f1, recs); print("templates:", len(recs))
    else:
        print(f"[skip] {f1}")
    base = [json.loads(l) for l in open(f1)]

    # ---------------- OOD (hand-written)
    f3 = out / "expressions_ood.jsonl"
    if not f3.exists():
        ood = [dict(latex=l, spoken_text=s, domain=d, difficulty=4, template="ood", heldout=True, split="test_ood", expr_id=f"o{i:04d}") for i, (d, l, s) in enumerate(OOD)]
        write(f3, ood); print("ood:", len(ood))
    ood = [json.loads(l) for l in open(f3)]

    # ---------------- compositional trees
    f2 = out / "expressions_comp.jsonl"
    if not f2.exists():
        banned = {norm_latex(r["latex"]) for r in base if r["split"] != "train"} | {norm_latex(r["latex"]) for r in ood}
        banned_all = {norm_latex(r["latex"]) for r in base}                      # also avoid duplicating v1 expressions anywhere
        recs = []
        for r in compose(int(a.n_comp * 1.25), seed=seed + 11):
            nl = norm_latex(r["latex"])
            if nl in banned or nl in banned_all: continue
            x = h(r["latex"]); r["split"] = "test_comp" if x < sc["comp_test_frac"] else "validation" if x < sc["comp_test_frac"] + sc["comp_val_frac"] else "train"
            recs.append(r)
        recs = recs[: a.n_comp]
        for i, r in enumerate(recs): r["expr_id"] = f"c{i:06d}"
        write(f2, recs)
        from collections import Counter
        print("compositional:", Counter(r["split"] for r in recs), Counter(r["domain"] for r in recs))
    else:
        print(f"[skip] {f2}")


if __name__ == "__main__":
    main()
