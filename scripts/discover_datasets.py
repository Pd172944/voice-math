#!/usr/bin/env python
"""Query the Hugging Face Hub for speech/lecture/math datasets and write data/dataset_candidates.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR
from huggingface_hub import HfApi

KEYWORDS = ["math speech", "mathematical speech", "spoken math", "lecture audio", "lecture transcripts",
            "university lectures asr", "physics lecture", "statistics lecture", "machine learning lecture",
            "scientific talk", "latex speech", "audio latex", "mathematical formulas audio", "educational speech",
            "academic speech", "technical terminology asr", "mit opencourseware", "ted talks audio",
            "peoples_speech", "spoken equations", "math word problems audio"]
OK_LICENSES = ("cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "cc-by-3.0", "cc-by-2.0", "apache-2.0", "mit", "odc-by", "cc-by-sa-3.0")


def main(out=None):
    api = HfApi()
    out = Path(out or DATA_DIR / "dataset_candidates.json")
    seen = {}
    for kw in KEYWORDS:
        try:
            for d in api.list_datasets(search=kw, limit=25, full=True):
                if d.id in seen:
                    seen[d.id]["matched_keywords"].append(kw); continue
                tags = d.tags or []
                lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
                mods = [t.split(":", 1)[1] for t in tags if t.startswith("modality:")]
                tasks = [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")]
                size = [t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")]
                seen[d.id] = dict(id=d.id, license=lic, modality=mods, tasks=tasks, size_category=size,
                                  downloads=d.downloads, likes=d.likes, gated=bool(d.gated),
                                  is_audio=("audio" in mods) or any("speech" in t or "audio" in t for t in tasks),
                                  permissive=lic in OK_LICENSES, matched_keywords=[kw])
        except Exception as e:  # skip and continue
            print(f"[warn] search '{kw}' failed: {e}")
        time.sleep(0.2)
    cands = sorted(seen.values(), key=lambda r: (not r["permissive"], not r["is_audio"], -(r["downloads"] or 0)))
    usable = [c for c in cands if c["is_audio"] and c["permissive"] and not c["gated"]]
    for c in usable[:15]:  # splits via the dataset-viewer API-free route: card metadata only
        try:
            info = api.dataset_info(c["id"]); c["splits_hint"] = [s.get("name") for s in (info.card_data.get("dataset_info", {}) or {}).get("splits", [])] if isinstance(info.card_data.get("dataset_info", {}), dict) else None
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(n_candidates=len(cands), n_usable_audio_permissive=len(usable), candidates=cands), indent=1))
    print(f"{len(cands)} candidates, {len(usable)} permissive+audio+ungated -> {out}")
    for c in usable[:25]:
        print(f"  {c['id']:55s} {c['license']:12s} dl={c['downloads']}")


if __name__ == "__main__":
    main()
