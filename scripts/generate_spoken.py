#!/usr/bin/env python
"""Step 2: add phrasing variation (lead-in fillers) and assign voice/speed per example. Deterministic given the seed.
Processes every data/synthetic/expressions*.jsonl -> spoken*.jsonl (skipping ones that already exist)."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT

LEADS = ["so ", "we have ", "consider ", "now ", "then ", "that is ", "we get "]


def process(inp, out, cfg):
    rng = random.Random(cfg["seed"] + 7 + sum(map(ord, inp.name)))
    tcfg, sc = cfg["tts"], cfg["synthetic"]
    is_v1 = inp.name == "expressions.jsonl"
    train_voices = tcfg["train_voices"] if is_v1 else tcfg["train_voices"] + tcfg.get("extra_train_voices", [])
    rows, res = [json.loads(l) for l in open(inp)], []
    for r in rows:
        if rng.random() < sc["filler_prob"] and r["split"] not in ("test_hard", "test_ood"):
            r["spoken_text"] = rng.choice(LEADS) + r["spoken_text"]
        lo, hi = tcfg["speed_range"]
        if r["split"] in ("test_hard", "test_ood"):         # held-out voices at several speeds
            n = sc["hard_voices_per_expr"] if r["split"] == "test_hard" else 2
            for k in range(n):
                x = dict(r); x["voice"] = tcfg["heldout_voices"][k % 2]; x["speed"] = round([0.9, 1.0, 1.1][k % 3], 2); x["id"] = f"{r['expr_id']}_{k}"; res.append(x)
            continue
        pool = (tcfg["heldout_voices"] * 3 + train_voices) if r["split"] in ("test", "test_comp") else train_voices
        r["voice"] = rng.choice(pool); r["speed"] = round(rng.uniform(lo, hi), 2); r["id"] = r["expr_id"]; res.append(r)
    with out.open("w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    print(f"{len(res)} spoken examples -> {out}")


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    for inp in sorted((DATA_DIR / "synthetic").glob("expressions*.jsonl")):
        out = inp.with_name(inp.name.replace("expressions", "spoken"))
        if out.exists(): print(f"[skip] {out}"); continue
        process(inp, out, cfg)


if __name__ == "__main__":
    main()
