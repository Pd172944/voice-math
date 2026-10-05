#!/usr/bin/env python
"""Step 4: real-audio corpus. ingest (HF streaming, licensed) -> chunk -> generic ASR -> math/science detection -> normalise -> filter.
Every record keeps: source, license, ASR transcript, canonical target, confidence, method, prompt/version. Rejected rows are kept (accepted=false)
with a reason, so the filter is auditable. Skips gracefully if the source is unavailable (pipeline still yields the synthetic dataset)."""
import argparse, hashlib, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT
from src.normalization.canonicalize import canonicalize, valid_latex
from src.normalization.rules import tokenize

KEYWORDS = set("""equation equations derivative integral integrals variable variables theorem lemma matrix vector probability variance standard deviation
gaussian distribution function functions exponent squared cubed square root logarithm sine cosine tangent infinity sigma lambda theta alpha beta
epsilon delta gamma omega derivative gradient limit sum fraction numerator denominator polynomial quadratic calculus algebra geometry triangle
angle velocity acceleration momentum energy force voltage current resistance frequency wavelength entropy molecule atom electron proton
neutron hypothesis regression neural network algorithm parameter""".split())
MATH_CORE = set("squared cubed integral derivative sigma lambda theta alpha beta epsilon delta gamma omega infinity equals plus minus over times".split())


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())["real"]
    ap = argparse.ArgumentParser(); ap.add_argument("--max_scan", type=int, default=60000); ap.add_argument("--max_candidates", type=int, default=cfg["max_clips"])
    ap.add_argument("--out", default=str(DATA_DIR / "real/real_manifest.jsonl")); ap.add_argument("--use_llm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} exists"); return
    try:
        from datasets import load_dataset
        from datasets import Audio
        ds = load_dataset(cfg["source"], cfg["config"], split="train", streaming=True).cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"[warn] real source unavailable ({e}); continuing with synthetic only"); out.write_text(""); return
    import soundfile as sf, numpy as np
    cands, scanned = [], 0
    audio_dir = DATA_DIR / "real/audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        for ex in ds:
            scanned += 1
            if scanned > a.max_scan or len(cands) >= a.max_candidates: break
            words = set(re.findall(r"[a-z]+", ex["text"].lower()))
            if not (len(words & KEYWORDS) >= 2 or (words & MATH_CORE)): continue
            import io
            arr0, sr0 = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            if arr0.ndim > 1: arr0 = arr0.mean(1)
            ex["audio"] = {"array": arr0, "sampling_rate": sr0}
            hits = words & KEYWORDS
            if len(hits) >= 2 or (words & MATH_CORE):          # cheap prefilter on the corpus' own text; the label itself comes from ASR below
                dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
                if not (2.0 <= dur <= 30.0): continue
                p = audio_dir / (hashlib.md5(ex["id"].encode()).hexdigest()[:12] + ".flac")
                if not p.exists():
                    sr = ex["audio"]["sampling_rate"]; arr = np.asarray(ex["audio"]["array"], dtype="float32")
                    if sr != 16000:
                        import librosa; arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                    sf.write(p, arr, 16000, format="FLAC")
                cands.append(dict(id="real_" + p.stem, audio=str(p), duration=dur, source=f"{cfg['source']}:{ex['id']}", license=cfg["license"], keywords=sorted(hits)))
            if scanned % 5000 == 0: print(f"scanned {scanned}, candidates {len(cands)}", flush=True)
    except Exception as e:
        print(f"[warn] streaming interrupted: {e}")
    print(f"scanned {scanned}; {len(cands)} science/math candidates -> ASR")
    if not cands:
        out.write_text(""); return
    from src.inference.asr import ASR
    asr = ASR()
    hyps = asr.transcribe([c["audio"] for c in cands], batch_size=24)
    n_acc = 0
    with out.open("w") as f:
        for c, h in zip(cands, hyps):
            r = canonicalize(h, use_llm=a.use_llm)
            toks = tokenize(h)
            reason = None
            if not r["is_math"]: reason = "not_math"
            elif r["confidence"] < cfg["min_confidence"]: reason = "low_confidence"
            elif not valid_latex(r["latex"]): reason = "invalid_latex"
            elif r.get("unknown"): reason = "unknown_words"
            elif not (re.search(r"\\[a-zA-Z]+|[\^_=]", r["latex"])): reason = "no_math_structure"
            elif len(toks) > 40: reason = "too_long"
            c.update(asr_transcript=h, canonical=r["latex"], confidence=r["confidence"], method=r["method"], math_score=r.get("math_score"),
                     transform="src/normalization/canonicalize.py:rules" + (f" +llm({r['llm']['model']},{r['llm']['prompt_version']})" if r.get("llm") else ""),
                     accepted=reason is None, reject_reason=reason, domain="real_science", speaker_id="unknown",
                     split="test" if int(hashlib.md5(c["id"].encode()).hexdigest(), 16) % 5 == 0 else "train")
            n_acc += reason is None
            f.write(json.dumps(c) + "\n")
    print(f"real corpus: {n_acc}/{len(cands)} accepted -> {out}")


if __name__ == "__main__":
    main()
