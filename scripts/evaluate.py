#!/usr/bin/env python
"""Evaluate systems on the test splits. Systems:
  A zeroshot   : generic Whisper -> raw transcript
  C rules      : generic Whisper -> rule-based math normaliser
  B canonical  : MathSpeech LoRA (target_mode=canonical) -> LaTeX directly
  D spoken_ft  : LoRA trained on spoken targets -> rule normaliser (ablation: 'ASR fine-tune + text normalisation')
Predictions are cached per (split,id,system) in artifacts/evaluation/predictions_<split>.jsonl; re-running only fills what is missing."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR, MODEL_ID
from src.normalization.canonicalize import canonicalize
from src.evaluation import metrics as M

EVAL = OUTPUT_DIR / "evaluation"
ADAPTERS = {"canonical": OUTPUT_DIR / "model/mathspeech-canonical/best", "spoken_ft": OUTPUT_DIR / "model/mathspeech-spoken/best"}


def load_rows(split, limit=0):
    rows = [json.loads(l) for l in open(DATA_DIR / f"manifests/mathspeech_{split}.jsonl")]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test,test_hard"); ap.add_argument("--systems", default="zeroshot,canonical,spoken_ft"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--batch_size", type=int, default=24); ap.add_argument("--beams", type=int, default=1)
    a = ap.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    systems = a.systems.split(",")
    for split in a.splits.split(","):
        if not (DATA_DIR / f"manifests/mathspeech_{split}.jsonl").exists(): continue
        rows = load_rows(split, a.limit)
        pf = EVAL / f"predictions_{split}.jsonl"
        preds = {}
        if pf.exists():
            for l in open(pf): d = json.loads(l); preds[d["id"]] = d
        for sysname in systems:
            todo = [r for r in rows if sysname not in preds.get(r["id"], {})]
            if not todo: continue
            ad = ADAPTERS.get(sysname)
            if sysname != "zeroshot" and not (ad and ad.exists()):
                print(f"[skip] {sysname}: no adapter at {ad}"); continue
            from src.inference.asr import ASR
            import torch
            asr = ASR(a.model, adapter=str(ad) if ad else None)
            hyp = asr.transcribe([r["audio"] for r in todo], batch_size=a.batch_size, num_beams=a.beams)
            for r, h in zip(todo, hyp):
                d = preds.setdefault(r["id"], {"id": r["id"]}); d[sysname] = h
            del asr; torch.cuda.empty_cache()
            print(f"[{split}] {sysname}: {len(todo)} clips")
        with pf.open("w") as f:
            for d in preds.values(): f.write(json.dumps(d) + "\n")
        # ---- metrics
        by = {r["id"]: r for r in rows}
        res = {}
        def sysout(d, name):
            if name == "zeroshot": return d.get("zeroshot")
            if name == "rules": return canonicalize(d["zeroshot"])["latex"] if "zeroshot" in d else None
            if name == "spoken_ft_rules": return canonicalize(d["spoken_ft"])["latex"] if "spoken_ft" in d else None
            return d.get(name)
        for name in ["zeroshot", "rules", "canonical", "spoken_ft_rules"]:
            sc = []; wers = []
            for i, d in preds.items():
                if i not in by: continue
                h = sysout(d, name)
                if h is None: continue
                ref = by[i]["target_text"]
                sc.append(M.score_example(h, ref))
                src = {"zeroshot": "zeroshot", "rules": "zeroshot", "canonical": "canonical", "spoken_ft_rules": "spoken_ft"}[name]
                if name in ("zeroshot",) or (name == "spoken_ft_rules"):
                    wers.append(M.wer(d[src], by[i]["spoken_transcript"]))
            if sc:
                res[name] = M.aggregate(sc)
                if wers: res[name]["wer_vs_spoken"] = sum(wers) / len(wers)
        # per-domain exact match for the main systems
        dom = {}
        for name in ("zeroshot", "rules", "canonical"):
            for i, d in preds.items():
                if i in by and sysout(d, name) is not None:
                    dom.setdefault(by[i]["domain"], {}).setdefault(name, []).append(M.score_example(sysout(d, name), by[i]["target_text"])["exact"])
        res["per_domain_exact"] = {k: {n: sum(v) / len(v) for n, v in x.items()} | {"n": len(next(iter(x.values())))} for k, x in dom.items()}
        (EVAL / f"metrics_{split}.json").write_text(json.dumps(res, indent=1))
        print(f"== {split}"); 
        for k, v in res.items():
            if k != "per_domain_exact": print(f"{k:16s}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if not x.endswith("support")})


if __name__ == "__main__":
    main()
