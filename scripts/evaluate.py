#!/usr/bin/env python
"""Evaluate systems on the test splits. Systems (suffix _rules = followed by the rule-based normaliser):
  zeroshot / rules   : generic Qwen3-ASR (raw)  /  generic + rule normaliser           (baselines A, C)
  canonical          : MathSpeech = Qwen3-ASR + LoRA trained on canonical LaTeX          (B)
  whisper_*          : previous-generation Whisper large-v3-turbo results (cached; test + test_hard only)
Splits: test (unseen expressions) · test_hard (unseen templates) · test_comp (unseen compositions) · test_ood (hand-written, never trained on).
Predictions are cached per (split,id,system) in artifacts/evaluation/predictions_<split>.jsonl; re-running only fills what is missing."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR, MODEL_ID
from src.normalization.canonicalize import canonicalize
from src.evaluation import metrics as M

EVAL = OUTPUT_DIR / "evaluation"
ADAPTERS = {"canonical": OUTPUT_DIR / "model/qwen3asr-canonical/best"}
# metric name -> (prediction key, apply_rules?)
VIEWS = {"zeroshot": ("zeroshot", False), "rules": ("zeroshot", True), "canonical": ("canonical", False),
         "whisper_zeroshot": ("whisper_zeroshot", False), "whisper_rules": ("whisper_zeroshot", True), "whisper_canonical": ("whisper_canonical", False),
         "whisper_spoken_ft_rules": ("whisper_spoken_ft", True)}
WER_KEYS = {"zeroshot", "whisper_zeroshot", "whisper_spoken_ft_rules"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test,test_hard,test_comp,test_ood"); ap.add_argument("--systems", default="zeroshot,canonical"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--beams", type=int, default=1)
    a = ap.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    from src.inference.asr import make_asr
    import torch
    for split in a.splits.split(","):
        mpath = DATA_DIR / f"manifests/mathspeech_{split}.jsonl"
        if not mpath.exists(): continue
        rows = [json.loads(l) for l in open(mpath)]; rows = rows[: a.limit] if a.limit else rows
        pf = EVAL / f"predictions_{split}.jsonl"; preds = {}
        if pf.exists():
            for l in open(pf): d = json.loads(l); preds[d["id"]] = d
        for sysname in a.systems.split(","):
            todo = [r for r in rows if sysname not in preds.get(r["id"], {})]
            if not todo: continue
            ad = ADAPTERS.get(sysname)
            if sysname != "zeroshot" and not (ad and ad.exists()):
                print(f"[skip] {sysname}: no adapter at {ad}"); continue
            asr = make_asr(a.model, adapter=str(ad) if ad else None)
            hyp = asr.transcribe([r["audio"] for r in todo], batch_size=a.batch_size, num_beams=a.beams)
            for r, h in zip(todo, hyp): preds.setdefault(r["id"], {"id": r["id"]})[sysname] = h
            del asr; torch.cuda.empty_cache(); print(f"[{split}] {sysname}: {len(todo)} clips")
        with pf.open("w") as f:
            for d in preds.values(): f.write(json.dumps(d) + "\n")
        by = {r["id"]: r for r in rows}; res = {}
        def view(d, name):
            key, rules = VIEWS[name]
            if key not in d: return None
            return canonicalize(d[key])["latex"] if rules else d[key]
        for name in VIEWS:
            sc, wers = [], []
            for i, d in preds.items():
                if i not in by: continue
                h = view(d, name)
                if h is None: continue
                sc.append(M.score_example(h, by[i]["target_text"]))
                if name in WER_KEYS: wers.append(M.wer(d[VIEWS[name][0]], by[i]["spoken_transcript"]))
            if sc:
                res[name] = M.aggregate(sc)
                if wers: res[name]["wer_vs_spoken"] = sum(wers) / len(wers)
        dom = {}
        for name in ("zeroshot", "rules", "canonical"):
            for i, d in preds.items():
                if i in by and view(d, name) is not None:
                    dom.setdefault(by[i]["domain"], {}).setdefault(name, []).append(M.score_example(view(d, name), by[i]["target_text"])["exact"])
        res["per_domain_exact"] = {k: {n: sum(v) / len(v) for n, v in x.items()} | {"n": len(next(iter(x.values())))} for k, x in dom.items()}
        (EVAL / f"metrics_{split}.json").write_text(json.dumps(res, indent=1))
        print(f"== {split}")
        for k, v in res.items():
            if k != "per_domain_exact": print(f"{k:24s}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if not x.endswith("support") and x in ("n", "exact_match", "latex_token_edit", "wer_vs_spoken")})

    log_wandb(a)


def log_wandb(a):
    """Log every metrics_<split>.json as one W&B eval run (best-effort)."""
    import os
    if not os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "disabled": return
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "voiceTrain"), entity=os.environ.get("WANDB_ENTITY"), name=f"eval-{Path(a.model).name}", job_type="eval", reinit=True)
        flat = {}
        for f in EVAL.glob("metrics_test*.json"):
            if "canonical_v1" in f.name: continue
            split = f.stem.replace("metrics_", "")
            for sysname, m in json.loads(f.read_text()).items():
                if sysname == "per_domain_exact": continue
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not k.endswith("support"): flat[f"{split}/{sysname}/{k}"] = v
        run.summary.update(flat); run.finish()
    except Exception as e:
        print(f"[warn] wandb eval logging failed: {e}")


if __name__ == "__main__":
    main()
