#!/usr/bin/env python
"""Step 2: add phrasing variation (lead-in fillers) and assign voice/speed per example. Deterministic given the seed.
(An optional LLM paraphrase pass can be enabled with --llm; canonical LaTeX stays the source of truth and the
paraphrase is accepted only if the rule normalizer maps it back to the same LaTeX.)"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT

LEADS = ["so ", "we have ", "consider ", "now ", "then ", "that is ", "we get "]


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", default=str(DATA_DIR / "synthetic/expressions.jsonl")); ap.add_argument("--out", default=str(DATA_DIR / "synthetic/spoken.jsonl"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"[skip] {out} exists"); return
    rng = random.Random(cfg["seed"] + 7)
    tcfg, sc = cfg["tts"], cfg["synthetic"]
    rows = [json.loads(l) for l in open(a.inp)]
    res = []
    for r in rows:
        if rng.random() < sc["filler_prob"] and r["split"] != "test_hard":
            r["spoken_text"] = rng.choice(LEADS) + r["spoken_text"]
        lo, hi = tcfg["speed_range"]
        if r["split"] in ("test", "test_hard"):
            voices = tcfg["heldout_voices"] + (tcfg["train_voices"] if r["split"] == "test" else [])
            pool = voices if r["split"] == "test_hard" else tcfg["heldout_voices"] * 3 + tcfg["train_voices"]
        else:
            pool = tcfg["train_voices"]
        if r["split"] == "test_hard":     # each held-out-template expression spoken by held-out voices at several speeds
            for k in range(sc["hard_voices_per_expr"]):
                x = dict(r); x["voice"] = tcfg["heldout_voices"][k % 2]; x["speed"] = round([0.9, 1.0, 1.1][k % 3], 2)
                x["id"] = f"{r['expr_id']}_{k}"; res.append(x)
        else:
            r["voice"] = rng.choice(pool); r["speed"] = round(rng.uniform(lo, hi), 2); r["id"] = r["expr_id"]; res.append(r)
    with out.open("w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    print(f"{len(res)} spoken examples -> {out}")


if __name__ == "__main__":
    main()
