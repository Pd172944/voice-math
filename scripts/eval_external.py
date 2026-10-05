#!/usr/bin/env python
"""EVALUATION-ONLY probe on real human math speech: AAAI2025/MathSpeech (1,101 clips, audio+LaTeX, YouTube-sourced).
Its licence is undeclared => per project policy it is NEVER used for training, never copied into the repo/artifacts (downloaded to the git-ignored data/raw/),
and only aggregate metrics are reported. Reference LaTeX is cleaned (\\displaystyle, empty groups, $ ) because the labels use a different style than ours."""
import argparse, io, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, soundfile as sf
from src.env import DATA_DIR, OUTPUT_DIR, MODEL_ID
from src.normalization.canonicalize import canonicalize
from src.evaluation import metrics as M


def clean_ref(s):
    s = s.strip().strip("$")
    s = re.sub(r"\\(displaystyle|textstyle|operatorname\{=\}|,|;|!|quad|left|right)", lambda m: "=" if "operatorname" in m.group(0) else "", s)
    s = s.replace("{}", "").replace("\\prime", "'").replace("\\cdot", "\\cdot")
    s = re.sub(r"\^\{\\prime\}", "'", s)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--adapter", default=str(OUTPUT_DIR / "model/mathspeech-canonical/best"))
    a = ap.parse_args()
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("AAAI2025/MathSpeech", "data/train-00000-of-00001.parquet", repo_type="dataset", local_dir=str(DATA_DIR / "raw/aaai_mathspeech"))
    df = pd.read_parquet(p)
    if a.n: df = df.iloc[: a.n]
    audio = [(sf.read(io.BytesIO(r["bytes"]), dtype="float32")) for r in df.audio]
    audio = [(x.mean(1) if x.ndim > 1 else x, sr) for x, sr in audio]
    refs = [clean_ref(x) for x in df.LaTeX]
    from src.inference.asr import ASR
    import torch
    g = ASR(MODEL_ID).transcribe(audio, batch_size=16); torch.cuda.empty_cache()
    m = ASR(MODEL_ID, adapter=a.adapter).transcribe(audio, batch_size=16)
    systems = {"zeroshot": g, "rules": [canonicalize(x)["latex"] for x in g], "mathspeech": m}
    res = {k: M.aggregate([M.score_example(h, r) for h, r in zip(v, refs)]) for k, v in systems.items()}
    res["zeroshot"]["wer_vs_spoken"] = float(np.mean([M.wer(h, t) for h, t in zip(g, df.transcription)]))
    out = OUTPUT_DIR / "evaluation/external_aaai_mathspeech.json"; out.write_text(json.dumps(res, indent=1))
    for k, v in res.items(): print(f"{k:11s}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if not x.endswith("support")})
    # a few qualitative lines (aggregate-only policy: ids, not audio)
    for i in range(0, min(len(df), 600), 75):
        print(f"[{i}] spoken: {df.transcription.iloc[i].strip()!r}\n   generic: {g[i]!r}\n   mathspeech: {m[i]!r}\n   ref: {refs[i]!r}")


if __name__ == "__main__":
    main()
