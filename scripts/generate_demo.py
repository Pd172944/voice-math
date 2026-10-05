#!/usr/bin/env python
"""Pick the best before/after examples (generic ASR wrong, MathSpeech exactly right; diverse domains/splits) -> artifacts/demo/."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR, ROOT
from src.normalization.canonicalize import canonicalize
from src.evaluation.metrics import norm_latex
from src.rendering.latex_to_text import latex_to_text


def main(n=24):
    ev = OUTPUT_DIR / "evaluation"; demo = OUTPUT_DIR / "demo"; (demo / "audio").mkdir(parents=True, exist_ok=True)
    cands = []
    for split in ("test_ood", "test_comp", "test", "test_hard"):
        mp, pf = DATA_DIR / f"manifests/mathspeech_{split}.jsonl", ev / f"predictions_{split}.jsonl"
        if not (mp.exists() and pf.exists()): continue
        mf = {json.loads(l)["id"]: json.loads(l) for l in open(mp)}
        for l in open(pf):
            d = json.loads(l); r = mf.get(d["id"])
            if not r or "canonical" not in d or "zeroshot" not in d: continue
            rules = canonicalize(d["zeroshot"])["latex"]
            cands.append(dict(id=d["id"], split=split, domain=r["domain"], difficulty=r["difficulty"], audio_src=str(ROOT / r["audio"]), spoken=r["spoken_transcript"], generic=d["zeroshot"],
                              rules=rules, mathspeech=d["canonical"], truth=r["target_text"], text=latex_to_text(d["canonical"]), ok_b=norm_latex(d["canonical"]) == norm_latex(r["target_text"]),
                              ok_c=norm_latex(rules) == norm_latex(r["target_text"]), speaker=r["speaker_id"]))
    good = [c for c in cands if c["ok_b"] and 12 < len(c["truth"]) < 90]
    good.sort(key=lambda c: (c["ok_c"], c["split"] != "test_ood", -c["difficulty"], -len(c["truth"])))
    picked, dom, spl, seen = [], {}, {}, set()
    for c in good:
        key = c["truth"].replace(" ", "")
        if key in seen or dom.get(c["domain"], 0) >= 3 or spl.get(c["split"], 0) >= 10: continue
        seen.add(key); dom[c["domain"]] = dom.get(c["domain"], 0) + 1; spl[c["split"]] = spl.get(c["split"], 0) + 1; picked.append(c)
        if len(picked) >= n: break
    for c in picked:
        dst = demo / "audio" / (c["id"] + ".flac"); shutil.copy(c["audio_src"], dst); c["audio"] = f"audio/{c['id']}.flac"; c.pop("audio_src")
    (demo / "examples.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    print(f"{len(picked)} demo examples; domains {dom}; splits {spl}")


if __name__ == "__main__":
    main()
