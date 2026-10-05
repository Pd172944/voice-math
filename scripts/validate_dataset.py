#!/usr/bin/env python
"""Step 6: sanity checks - audio exists/duration, label validity, no train/test LaTeX leakage, licences present."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR, ROOT
from src.normalization.canonicalize import valid_latex
from src.evaluation.metrics import norm_latex


def validate():
    man = DATA_DIR / "manifests"
    splits = {n: [json.loads(l) for l in open(man / f"mathspeech_{n}.jsonl")] for n in ("train", "validation", "test", "test_hard", "test_comp", "test_ood") if (man / f"mathspeech_{n}.jsonl").exists()}
    errs, report = [], {}
    for n, rows in splits.items():
        bad_audio = sum(1 for r in rows if not (ROOT / r["audio"]).exists())
        bad_dur = sum(1 for r in rows if not (0.3 < r["duration"] < 30))
        bad_latex = sum(1 for r in rows if r["synthetic"] and not valid_latex(r["target_text"]))
        no_lic = sum(1 for r in rows if not r.get("license") or not r.get("source"))
        report[n] = dict(n=len(rows), hours=round(sum(r["duration"] for r in rows) / 3600, 2), bad_audio=bad_audio, bad_duration=bad_dur, invalid_latex=bad_latex,
                         missing_license_or_source=no_lic, synthetic=sum(r["synthetic"] for r in rows), real=sum(not r["synthetic"] for r in rows))
        for k in ("bad_audio", "bad_duration", "invalid_latex", "missing_license_or_source"):
            if report[n][k]: errs.append(f"{n}: {k}={report[n][k]}")
    tr = {norm_latex(r["target_text"]) for r in splits["train"]}
    for n in ("validation", "test", "test_hard", "test_comp", "test_ood"):
        if n in splits:
            leak = sum(norm_latex(r["target_text"]) in tr for r in splits[n]); report[n]["train_overlap"] = leak
            if leak: errs.append(f"{n}: {leak} examples share canonical LaTeX with train")
    (OUTPUT_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "datasets/validation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if errs:
        print("VALIDATION ERRORS:", errs); sys.exit(1)
    print("dataset OK")


if __name__ == "__main__":
    validate()
