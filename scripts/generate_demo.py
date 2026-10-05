#!/usr/bin/env python
"""Step 11: pick the best before/after examples (generic ASR wrong, MathSpeech exactly right; diverse domains) -> artifacts/demo/."""
import json, shutil, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR
from src.normalization.canonicalize import canonicalize
from src.evaluation.metrics import norm_latex


def main(n=18):
    ev = OUTPUT_DIR / "evaluation"; demo = OUTPUT_DIR / "demo"; (demo / "audio").mkdir(parents=True, exist_ok=True)
    cands = []
    for split in ("test", "test_hard"):
        mf = {json.loads(l)["id"]: json.loads(l) for l in open(DATA_DIR / f"manifests/mathspeech_{split}.jsonl")}
        pf = ev / f"predictions_{split}.jsonl"
        if not pf.exists(): continue
        for l in open(pf):
            d = json.loads(l); r = mf.get(d["id"])
            if not r or "canonical" not in d: continue
            ok_b = norm_latex(d["canonical"]) == norm_latex(r["target_text"])
            rules = canonicalize(d["zeroshot"])["latex"]
            ok_c = norm_latex(rules) == norm_latex(r["target_text"])
            cands.append(dict(id=d["id"], split=split, domain=r["domain"], difficulty=r["difficulty"], audio_src=r["audio"], spoken=r["spoken_transcript"], generic=d["zeroshot"],
                              rules=rules, mathspeech=d["canonical"], truth=r["target_text"], ok_b=ok_b, ok_c=ok_c, speaker=r["speaker_id"]))
    good = [c for c in cands if c["ok_b"] and len(c["truth"]) < 90]
    good.sort(key=lambda c: (c["ok_c"], -c["difficulty"], -len(c["truth"])))     # prefer cases where even generic+rules fails, hard, long
    rng = random.Random(0); picked, seen_dom = [], {}
    for c in good:
        if seen_dom.get(c["domain"], 0) >= 3: continue
        seen_dom[c["domain"]] = seen_dom.get(c["domain"], 0) + 1; picked.append(c)
        if len(picked) >= n: break
    for c in picked:
        dst = demo / "audio" / (c["id"] + ".flac"); shutil.copy(c["audio_src"], dst); c["audio"] = f"audio/{c['id']}.flac"
    (demo / "examples.json").write_text(json.dumps(picked, indent=1))
    print(f"{len(picked)} demo examples; domains {seen_dom}")


if __name__ == "__main__":
    main()
